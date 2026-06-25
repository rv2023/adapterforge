# NVIDIA GPU Operator. driver.enabled=false because the GPU node uses the EKS
# AL2023_x86_64_NVIDIA AMI (driver + runtime prebaked). The Operator then manages
# device-plugin + container-toolkit + DCGM + MIG-manager + NFD. docs/m7-concepts §7.
resource "helm_release" "gpu_operator" {
  name             = "gpu-operator"
  repository       = "https://helm.ngc.nvidia.com/nvidia"
  chart            = "gpu-operator"
  version          = var.gpu_operator_version # null => latest (pin via tfvars for repro)
  namespace        = "gpu-operator"
  create_namespace = true

  set {
    name  = "driver.enabled"
    value = "false"
  }
}

# TODO(next M7 pieces): add as we reach them, in this same addons module —
#   - Kueue            (helm_release: oci://registry.k8s.io/kueue/charts/kueue, or manifests)
#   - kube-prometheus-stack (helm_release; values: ephemeral storage per the storage decision)
#   - (M8) KServe
