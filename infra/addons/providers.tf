provider "aws" {
  region = var.region
}

# Read the existing cluster (must already be applied via ../). Data source, not a resource.
data "aws_eks_cluster" "this" {
  name = var.cluster_name
}

# Auth via the `exec` plugin (aws eks get-token) rather than a token data source: exec
# fetches a FRESH token at apply time, so it can't expire mid-apply.
locals {
  cluster_host = data.aws_eks_cluster.this.endpoint
  cluster_ca   = base64decode(data.aws_eks_cluster.this.certificate_authority[0].data)
  token_exec = {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", var.cluster_name, "--region", var.region]
  }
}

provider "kubernetes" {
  host                   = local.cluster_host
  cluster_ca_certificate = local.cluster_ca
  exec {
    api_version = local.token_exec.api_version
    command     = local.token_exec.command
    args        = local.token_exec.args
  }
}

provider "helm" {
  kubernetes {
    host                   = local.cluster_host
    cluster_ca_certificate = local.cluster_ca
    exec {
      api_version = local.token_exec.api_version
      command     = local.token_exec.command
      args        = local.token_exec.args
    }
  }
}
