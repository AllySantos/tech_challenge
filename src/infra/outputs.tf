output "ecr_repository_url" {
  value       = aws_ecr_repository.app.repository_url
  description = "ECR repository URL"
}

output "sagemaker_role_arn" {
  value       = aws_iam_role.sagemaker_role.arn
  description = "SageMaker IAM role ARN"
}

output "sagemaker_bucket_name" {
  value       = aws_s3_bucket.sagemaker.id
  description = "The name of the SageMaker bucket"
}

output "sagemaker_bucket_arn" {
  value       = aws_s3_bucket.sagemaker.arn
  description = "The ARN of the SageMaker bucket"
}

output "alb_dns_name" {
  description = "The DNS name of the load balancer"
  value       = aws_lb.app.dns_name
}

output "mlflow_artifacts_bucket" {
  value       = aws_s3_bucket.mlflow_artifacts.id
  description = "S3 bucket for MLflow artifacts"
}

output "sagemaker_experiment_name" {
  description = "The name of the SageMaker Experiment (managed via SDK)"
  value       = "${var.app_name}-experiment"
}

output "sagemaker_mlflow_tracking_uri" {
  description = "The tracking URI for the MLflow server"
  value       = aws_sagemaker_mlflow_tracking_server.churn_mlflow.tracking_server_url
}

output "sagemaker_mlflow_tracking_server_name" {
  description = "The name of the SageMaker MLflow tracking server"
  value       = aws_sagemaker_mlflow_tracking_server.churn_mlflow.tracking_server_name
}

output "sagemaker_mlflow_tracking_server_arn" {
  description = "The tracking ARN for the MLflow server"
  value       = aws_sagemaker_mlflow_tracking_server.churn_mlflow.arn
}
