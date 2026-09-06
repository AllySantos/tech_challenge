# Equivalente gerenciado do par Prometheus + Grafana do compose local. O
# coletor ADOT que roda como sidecar na task da API raspa o mesmo /metrics e
# faz remote write para cá, então as mesmas queries e o mesmo dashboard JSON
# continuam valendo.

resource "aws_prometheus_workspace" "main" {
  alias = "${var.app_name}-metrics"

  tags = {
    Name = "${var.app_name}-metrics"
  }
}

resource "aws_grafana_workspace" "main" {
  name                     = var.app_name
  account_access_type      = "CURRENT_ACCOUNT"
  authentication_providers = ["AWS_SSO"]
  permission_type          = "SERVICE_MANAGED"
  data_sources             = ["PROMETHEUS"]
  role_arn                 = aws_iam_role.grafana.arn

  tags = {
    Name = "${var.app_name}-grafana"
  }
}

resource "aws_iam_role" "grafana" {
  name = "${var.app_name}-grafana-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "grafana.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "grafana_prometheus" {
  name = "${var.app_name}-grafana-prometheus"
  role = aws_iam_role.grafana.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "aps:QueryMetrics",
        "aps:GetLabels",
        "aps:GetSeries",
        "aps:GetMetricMetadata",
        "aps:ListWorkspaces",
        "aps:DescribeWorkspace"
      ]
      Resource = "*"
    }]
  })
}

# --- Alarmes ---------------------------------------------------------------
# Cobrem o que o dashboard mostra mas ninguém fica olhando de madrugada.

resource "aws_sns_topic" "alerts" {
  name = "${var.app_name}-alerts"
}

resource "aws_cloudwatch_metric_alarm" "target_5xx" {
  alarm_name          = "${var.app_name}-target-5xx"
  alarm_description   = "A API esta devolvendo erro de servidor"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 2
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    LoadBalancer = aws_lb.app.arn_suffix
    TargetGroup  = aws_lb_target_group.app.arn_suffix
  }
}

# O p95 medido ponta a ponta no ambiente local fica em torno de 4,8 ms. O
# limite de 500 ms é folgado de propósito: ele existe para pegar degradação
# estrutural, não oscilação normal.
resource "aws_cloudwatch_metric_alarm" "target_latency" {
  alarm_name          = "${var.app_name}-target-latency-p95"
  alarm_description   = "p95 de resposta acima do orcamento"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "TargetResponseTime"
  extended_statistic  = "p95"
  period              = 300
  evaluation_periods  = 2
  threshold           = var.alarm_p95_latency_seconds
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    LoadBalancer = aws_lb.app.arn_suffix
    TargetGroup  = aws_lb_target_group.app.arn_suffix
  }
}

resource "aws_cloudwatch_metric_alarm" "unhealthy_hosts" {
  alarm_name          = "${var.app_name}-unhealthy-hosts"
  alarm_description   = "Ha tasks fora do target group"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "UnHealthyHostCount"
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 3
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    LoadBalancer = aws_lb.app.arn_suffix
    TargetGroup  = aws_lb_target_group.app.arn_suffix
  }
}
