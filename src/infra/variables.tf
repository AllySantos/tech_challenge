variable "aws_region" {
  default = "us-east-1"
}

variable "app_name" {
  default = "churn-prediction"
}

variable "environment" {
  default = "prod"
}

variable "ecr_repo_name" {
  default = "churn-prediction"
}

variable "app_image_tag" {
  description = "Container image tag deployed to ECS"
  default     = "app-latest"
}

variable "container_port" {
  description = "Container port exposed by the ECS service and ALB target group"
  default     = 80
}

variable "sagemaker_training_instance_type" {
  description = "EC2 instance type for SageMaker training"
  default     = "ml.t2.micro"
}
