resource "aws_iam_role_policy" "sagemaker_mlflow_s3" {
  name = "${var.app_name}-sagemaker-mlflow-s3"
  role = aws_iam_role.sagemaker_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ]
      Resource = [
        aws_s3_bucket.mlflow_artifacts.arn,
        "${aws_s3_bucket.mlflow_artifacts.arn}/*"
      ]
    }]
  })
}

resource "aws_iam_role_policy" "sagemaker_experiments" {
  name = "${var.app_name}-sagemaker-experiments"
  role = aws_iam_role.sagemaker_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker-mlflow:*",
          "sagemaker:CreateExperiment",
          "sagemaker:DescribeExperiment",
          "sagemaker:UpdateExperiment",
          "sagemaker:CreateTrial",
          "sagemaker:DescribeTrial",
          "sagemaker:CreateTrialComponent",
          "sagemaker:DescribeTrialComponent",
          "sagemaker:UpdateTrialComponent",
          "sagemaker:AddTags",
          "sagemaker:DescribeMlflowTrackingServer"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:DescribeLogStreams",
          "logs:GetLogEvents",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

resource "aws_sagemaker_mlflow_tracking_server" "churn_mlflow" {
  tracking_server_name = "${var.app_name}-tracking-server"
  role_arn             = aws_iam_role.mlflow_tracking_role.arn
  artifact_store_uri   = "s3://${aws_s3_bucket.mlflow_artifacts.bucket}"
  tracking_server_size = "Small"

  tags = {
    Name = "${var.app_name}-mlflow-server"
  }
}
