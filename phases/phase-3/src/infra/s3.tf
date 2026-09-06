# O registry de modelos migra do disco local para o S3, mantendo o mesmo
# contrato: versões em models/<timestamp>/ e o ponteiro models/current.json.
# A task de retreino escreve, as tasks da API leem na subida.

resource "aws_s3_bucket" "models" {
  bucket        = "${var.app_name}-models-${data.aws_caller_identity.current.account_id}"
  force_destroy = true

  tags = {
    Name = "${var.app_name}-models"
  }
}

# Versionamento é o que torna o rollback recuperável mesmo se uma versão ruim
# sobrescrever o ponteiro: dá para restaurar a revisão anterior de
# current.json sem reprocessar nada.
resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    id     = "expira-versoes-antigas"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}
