resource "aws_ecs_cluster" "main" {
  name = "${var.app_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "${var.app_name}-cluster"
  }
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.app_name}/api"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_group" "training" {
  name              = "/ecs/${var.app_name}/training"
  retention_in_days = var.log_retention_days
}

locals {
  image_uri = "${aws_ecr_repository.app.repository_url}:${var.app_image_tag}"

  # Configuração do coletor ADOT, passada inline por variável de ambiente para
  # não exigir um parâmetro no SSM só por causa de um arquivo YAML. O coletor
  # raspa o /metrics da própria task via localhost e faz remote write para o
  # workspace gerenciado do Prometheus.
  otel_config = yamlencode({
    receivers = {
      prometheus = {
        config = {
          scrape_configs = [{
            job_name        = "triage-api"
            scrape_interval = "15s"
            static_configs  = [{ targets = ["localhost:${var.container_port}"] }]
          }]
        }
      }
    }
    processors = {
      batch = { timeout = "30s" }
    }
    exporters = {
      prometheusremotewrite = {
        endpoint = "${aws_prometheus_workspace.main.prometheus_endpoint}api/v1/remote_write"
        auth     = { authenticator = "sigv4auth" }
      }
    }
    extensions = {
      sigv4auth = { region = var.aws_region }
    }
    service = {
      extensions = ["sigv4auth"]
      pipelines = {
        metrics = {
          receivers  = ["prometheus"]
          processors = ["batch"]
          exporters  = ["prometheusremotewrite"]
        }
      }
    }
  })
}

# A imagem da API não embarca modelo: o artefato promovido é sincronizado do
# S3 por um container de inicialização, que precisa terminar com sucesso antes
# de a API subir. É o mesmo contrato do compose local — o serving lê um
# diretório de modelos montado somente-leitura —, apenas com o S3 no lugar do
# bind mount.
resource "aws_ecs_task_definition" "api" {
  family                   = "${var.app_name}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.api_task.arn

  volume {
    name = "models"
  }

  container_definitions = jsonencode([
    {
      name      = "model-sync"
      image     = "public.ecr.aws/aws-cli/aws-cli:latest"
      essential = false
      command   = ["s3", "sync", "s3://${aws_s3_bucket.models.id}/models", "/models", "--only-show-errors"]

      mountPoints = [{
        sourceVolume  = "models"
        containerPath = "/models"
        readOnly      = false
      }]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.api.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "model-sync"
        }
      }
    },
    {
      name      = "api"
      image     = local.image_uri
      essential = true

      dependsOn = [{
        containerName = "model-sync"
        condition     = "SUCCESS"
      }]

      portMappings = [{
        containerPort = var.container_port
        protocol      = "tcp"
      }]

      mountPoints = [{
        sourceVolume  = "models"
        containerPath = "/app/models"
        readOnly      = true
      }]

      environment = [
        { name = "INFERENCE_BACKEND", value = var.inference_backend },
        { name = "MODELS_DIR", value = "/app/models" },
        { name = "PROJECT_ROOT", value = "/app" }
      ]

      healthCheck = {
        command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:${var.container_port}/health')\""]
        interval    = 15
        timeout     = 5
        retries     = 3
        startPeriod = 30
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.api.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "api"
        }
      }
    },
    {
      name      = "otel-collector"
      image     = "public.ecr.aws/aws-observability/aws-otel-collector:latest"
      essential = false

      environment = [
        { name = "AOT_CONFIG_CONTENT", value = local.otel_config }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.api.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "otel"
        }
      }
    }
  ])

  tags = {
    Name = "${var.app_name}-api-task"
  }
}

resource "aws_ecs_service" "api" {
  name            = "${var.app_name}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.service_desired_count
  launch_type     = "FARGATE"

  # Sem downtime na troca de versão: o ECS sobe as novas tasks antes de
  # derrubar as antigas.
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  health_check_grace_period_seconds  = 60

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "api"
    container_port   = var.container_port
  }

  depends_on = [aws_lb_listener.app]

  # A tag da imagem é gerenciada pelo pipeline de deploy, que registra uma nova
  # revisão a cada merge. Ignorar aqui evita que o Terraform reverta o deploy.
  lifecycle {
    ignore_changes = [task_definition]
  }

  tags = {
    Name = "${var.app_name}-api-service"
  }
}

# Escala por CPU. A inferência é curta e barata, então o sinal que antecipa
# saturação é a fila de requisições concorrentes, refletida na CPU — não a
# memória, que fica praticamente constante.
resource "aws_appautoscaling_target" "api" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.service_min_capacity
  max_capacity       = var.service_max_capacity
}

resource "aws_appautoscaling_policy" "api_cpu" {
  name               = "${var.app_name}-cpu-target"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.api.service_namespace
  resource_id        = aws_appautoscaling_target.api.resource_id
  scalable_dimension = aws_appautoscaling_target.api.scalable_dimension

  target_tracking_scaling_policy_configuration {
    target_value       = 65
    scale_in_cooldown  = 300
    scale_out_cooldown = 60

    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}
