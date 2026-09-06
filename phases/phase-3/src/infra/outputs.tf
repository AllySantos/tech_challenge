output "api_url" {
  description = "URL publica da API de triagem"
  value       = "http://${aws_lb.app.dns_name}"
}

output "api_docs_url" {
  description = "Swagger UI"
  value       = "http://${aws_lb.app.dns_name}/docs"
}

output "ecr_repository_url" {
  description = "Repositorio ECR da imagem da API"
  value       = aws_ecr_repository.app.repository_url
}

output "models_bucket" {
  description = "Bucket S3 que guarda o registry de modelos"
  value       = aws_s3_bucket.models.id
}

output "ecs_cluster_name" {
  description = "Nome do cluster ECS"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "Nome do servico da API"
  value       = aws_ecs_service.api.name
}

output "training_task_family" {
  description = "Familia da task definition de retreino"
  value       = aws_ecs_task_definition.training.family
}

output "github_deploy_role_arn" {
  description = "Role assumida pelo GitHub Actions via OIDC"
  value       = aws_iam_role.github_deploy.arn
}

output "prometheus_endpoint" {
  description = "Endpoint de remote write do Amazon Managed Prometheus"
  value       = aws_prometheus_workspace.main.prometheus_endpoint
}

output "grafana_endpoint" {
  description = "URL do workspace do Amazon Managed Grafana"
  value       = aws_grafana_workspace.main.endpoint
}

output "alerts_topic_arn" {
  description = "Topico SNS que recebe os alarmes"
  value       = aws_sns_topic.alerts.arn
}
