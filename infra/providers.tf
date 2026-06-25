# AWS provider. Region + default tags (so every resource is attributable / easy to find).
provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "adapterforge"
      Milestone = "m7"
      ManagedBy = "terraform"
    }
  }
}

# NOTE: the kubernetes + helm providers (to install the GPU Operator / Kueue via Terraform)
# get added once the cluster exists — they authenticate against the EKS endpoint. We'll
# wire them in the Operator step, not here.
