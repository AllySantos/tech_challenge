# Equivalente gerenciado da DAG do Airflow. O MWAA seria a tradução literal,
# mas cobra por ambiente ligado 24/7 (algo em torno de 350 USD/mês) para rodar
# um pipeline de um minuto por semana. Uma task agendada no Fargate executa o
# mesmo `python -m src.pipeline`, com o mesmo portão de qualidade, e só é
# cobrada enquanto roda.

resource "aws_ecs_task_definition" "training" {
  family                   = "${var.app_name}-training"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.training_cpu
  memory                   = var.training_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.training_task.arn

  volume {
    name = "workspace"
  }

  container_definitions = jsonencode([
    {
      name      = "train"
      image     = local.image_uri
      essential = false
      command   = ["python", "-m", "src.pipeline"]

      mountPoints = [{
        sourceVolume  = "workspace"
        containerPath = "/app/models"
        readOnly      = false
      }]

      environment = [
        { name = "INFERENCE_BACKEND", value = var.inference_backend },
        { name = "MODELS_DIR", value = "/app/models" },
        { name = "PROJECT_ROOT", value = "/app" }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.training.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "train"
        }
      }
    },
    # Só publica se o treino sair com código zero. Como o portão de qualidade
    # levanta exceção quando o F1 ou a latência regridem, uma versão reprovada
    # nunca chega ao S3 — e o serving continua na anterior.
    {
      name      = "publish"
      image     = "public.ecr.aws/aws-cli/aws-cli:latest"
      essential = true
      command   = ["s3", "sync", "/models", "s3://${aws_s3_bucket.models.id}/models", "--only-show-errors"]

      dependsOn = [{
        containerName = "train"
        condition     = "SUCCESS"
      }]

      mountPoints = [{
        sourceVolume  = "workspace"
        containerPath = "/models"
        readOnly      = true
      }]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.training.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "publish"
        }
      }
    }
  ])

  tags = {
    Name = "${var.app_name}-training-task"
  }
}

resource "aws_scheduler_schedule" "retraining" {
  name                = "${var.app_name}-retraining"
  description         = "Retreino semanal do classificador de urgencia"
  schedule_expression = var.retraining_schedule
  group_name          = "default"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_ecs_cluster.main.arn
    role_arn = aws_iam_role.scheduler.arn

    ecs_parameters {
      task_definition_arn = aws_ecs_task_definition.training.arn_without_revision
      launch_type         = "FARGATE"
      task_count          = 1

      network_configuration {
        subnets          = aws_subnet.private[*].id
        security_groups  = [aws_security_group.training.id]
        assign_public_ip = false
      }
    }

    retry_policy {
      maximum_retry_attempts       = 1
      maximum_event_age_in_seconds = 3600
    }
  }
}

# Uma versão promovida só entra em serving quando as tasks reiniciam, porque a
# API carrega o modelo na subida. O alarme abaixo sinaliza que o retreino
# falhou; a reciclagem após um retreino bem-sucedido fica a cargo do operador
# (ou de um `ecs update-service --force-new-deployment` encadeado).
resource "aws_cloudwatch_metric_alarm" "retraining_failed" {
  alarm_name          = "${var.app_name}-retraining-failed"
  alarm_description   = "A task de retreino terminou com erro"
  namespace           = "AWS/Events"
  metric_name         = "FailedInvocations"
  statistic           = "Sum"
  period              = 3600
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  dimensions = {
    RuleName = aws_scheduler_schedule.retraining.name
  }
}
