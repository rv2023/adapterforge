# EKS cluster + two managed node groups (system CPU, GPU).
# (Node groups live inside the eks module call — that's the module's API, so both groups
# are here rather than a separate gpu_nodes.tf.)
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = var.eks_version

  # public endpoint so you can kubectl from your laptop. (Lock down CIDRs for real prod.)
  cluster_endpoint_public_access = true
  # gives the identity that runs `apply` admin on the cluster (so kubectl works immediately)
  enable_cluster_creator_admin_permissions = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    # System pods + the GPU Operator's controllers run here. Always on (cheap).
    system = {
      instance_types = ["t3.medium"]
      ami_type       = "AL2023_x86_64_STANDARD"
      min_size       = 1
      max_size       = 2
      desired_size   = 1
    }

    # GPU workloads. desired_size=0 by default => $0 between sessions; scale to 1 to test.
    gpu = {
      instance_types = [var.gpu_instance_type]
      # BASE AMI (no preinstalled NVIDIA driver) on purpose — we want to WATCH the GPU
      # Operator install driver/toolkit/device-plugin/DCGM. (Use *_NVIDIA_* for prebaked.)
      ami_type     = "AL2023_x86_64_STANDARD"
      min_size     = 0
      max_size     = var.gpu_max_size
      desired_size = var.gpu_desired_size

      labels = { "workload" = "gpu" }
      # Taint so ONLY GPU workloads (that tolerate it) land here — keeps system pods off
      # the expensive node. The GPU Operator tolerates this taint.
      taints = {
        gpu = {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
    }
  }
}
