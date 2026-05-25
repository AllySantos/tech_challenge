# Main SageMaker bucket for model artifacts and training data
resource "aws_s3_bucket" "sagemaker" {
  bucket = "${var.app_name}-sagemaker-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name = "${var.app_name}-sagemaker"
  }
}

# MLflow artifacts bucket for experiment tracking
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "${var.app_name}-mlflow-artifacts-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name = "${var.app_name}-mlflow-artifacts"
  }
}

# Versioning configuration for all buckets
resource "aws_s3_bucket_versioning" "all" {
  for_each = {
    sagemaker        = aws_s3_bucket.sagemaker.id
    mlflow_artifacts = aws_s3_bucket.mlflow_artifacts.id
  }
  bucket = each.value
  versioning_configuration {
    status = "Enabled"
  }
}

# Server-side encryption for all buckets
resource "aws_s3_bucket_server_side_encryption_configuration" "all" {
  for_each = {
    sagemaker        = aws_s3_bucket.sagemaker.id
    mlflow_artifacts = aws_s3_bucket.mlflow_artifacts.id
  }
  bucket = each.value

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Public access block for all buckets
resource "aws_s3_bucket_public_access_block" "all" {
  for_each = {
    sagemaker        = aws_s3_bucket.sagemaker.id
    mlflow_artifacts = aws_s3_bucket.mlflow_artifacts.id
  }
  bucket = each.value

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
