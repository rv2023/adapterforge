# M7 — Terraform/provider version pins. (IaC for the GPU EKS platform.)
terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # State: local for now (simplest; apply/destroy clean per session).
  # TODO(later): move to an S3 backend (you have the DVC bucket) for durable state:
  # backend "s3" { bucket = "adapterforge-dvc-073053153137" key = "infra/terraform.tfstate" region = "us-east-1" }
}
