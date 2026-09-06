resource "aws_lb" "app" {
  name               = "${var.app_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = false
  idle_timeout               = 30

  tags = {
    Name = "${var.app_name}-alb"
  }
}

resource "aws_lb_target_group" "app" {
  name        = "${var.app_name}-tg"
  port        = var.container_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  # A API responde /health mesmo em modo degradado (sem modelo carregado), o
  # que é proposital: o alvo continua saudável para o ALB e o alarme de modelo
  # ausente vem das métricas, não de um 503 em massa no load balancer.
  health_check {
    path                = "/health"
    matcher             = "200"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 15
  }

  # A inferência leva menos de 1 ms; drenar por 30 s já cobre qualquer
  # requisição em voo durante um deploy.
  deregistration_delay = 30

  tags = {
    Name = "${var.app_name}-tg"
  }
}

resource "aws_lb_listener" "app" {
  load_balancer_arn = aws_lb.app.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

# /metrics fica fora do alcance da internet: a exposição Prometheus revela
# volumetria e distribuição das predições, e quem raspa é o coletor que roda
# como sidecar dentro da própria task.
resource "aws_lb_listener_rule" "block_metrics" {
  listener_arn = aws_lb_listener.app.arn
  priority     = 10

  action {
    type = "fixed-response"

    fixed_response {
      content_type = "application/json"
      message_body = jsonencode({ detail = "Not found" })
      status_code  = "404"
    }
  }

  condition {
    path_pattern {
      values = ["/metrics"]
    }
  }
}
