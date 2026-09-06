variable "aws_region" {
  description = "Região onde toda a stack é provisionada"
  type        = string
  default     = "us-east-1"
}

variable "app_name" {
  description = "Prefixo aplicado ao nome de todos os recursos"
  type        = string
  default     = "medical-triage"
}

variable "environment" {
  description = "Ambiente lógico, usado em tags"
  type        = string
  default     = "prod"
}

variable "ecr_repo_name" {
  description = "Repositório ECR que guarda a imagem da API"
  type        = string
  default     = "medical-triage"
}

variable "app_image_tag" {
  description = "Tag da imagem servida pelo serviço ECS"
  type        = string
  default     = "app-latest"
}

variable "container_port" {
  description = "Porta exposta pelo container da API"
  type        = number
  default     = 8000
}

variable "inference_backend" {
  description = "Backend de inferência servido: sklearn, onnx, onnx-int8 ou onnx-pruned"
  type        = string
  default     = "onnx-pruned"

  validation {
    condition     = contains(["sklearn", "onnx", "onnx-int8", "onnx-pruned"], var.inference_backend)
    error_message = "Backend inválido. Use sklearn, onnx, onnx-int8 ou onnx-pruned."
  }
}

# 0,5 vCPU e 1 GB sustentam a carga medida com folga: o artefato servido tem
# 0,42 MB e a inferência leva 0,10 ms no p95. O gargalo é rede, não CPU.
variable "task_cpu" {
  description = "Unidades de CPU da task da API (1024 = 1 vCPU)"
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Memória da task da API, em MiB"
  type        = number
  default     = 1024
}

variable "service_desired_count" {
  description = "Réplicas iniciais do serviço"
  type        = number
  default     = 2
}

variable "service_min_capacity" {
  description = "Piso do autoscaling"
  type        = number
  default     = 2
}

variable "service_max_capacity" {
  description = "Teto do autoscaling"
  type        = number
  default     = 6
}

# O retreino roda em lote e não tem restrição de latência, só de tempo total —
# por isso recebe mais recursos que o serving.
variable "training_cpu" {
  description = "Unidades de CPU da task de retreino"
  type        = number
  default     = 2048
}

variable "training_memory" {
  description = "Memória da task de retreino, em MiB"
  type        = number
  default     = 4096
}

variable "retraining_schedule" {
  description = "Agendamento do retreino, no formato do EventBridge"
  type        = string
  default     = "cron(0 4 ? * SUN *)"
}

variable "log_retention_days" {
  description = "Retenção dos logs no CloudWatch"
  type        = number
  default     = 14
}

variable "alarm_p95_latency_seconds" {
  description = "Limite de p95 no ALB que dispara alarme"
  type        = number
  default     = 0.5
}

# Só pode existir um provider OIDC por URL em cada conta AWS. A Fase 1 cria o
# dela; se as duas fases forem para a mesma conta, desligue esta flag e o
# provider existente será reaproveitado.
variable "create_github_oidc_provider" {
  description = "Cria o provider OIDC do GitHub. Desligue se a conta já tiver um."
  type        = bool
  default     = true
}

variable "github_repository" {
  description = "Repositório autorizado a assumir a role de deploy via OIDC"
  type        = string
  default     = "AllySantos/tech_challenge"
}
