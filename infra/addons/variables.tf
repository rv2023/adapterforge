variable "region" {
  type    = string
  default = "us-east-1"
}

variable "cluster_name" {
  type    = string
  default = "adapterforge"
}

variable "gpu_operator_version" {
  description = "NVIDIA GPU Operator chart version. null => latest. TODO: pin for reproducibility (e.g. \"v24.9.2\")."
  type        = string
  default     = null
}

variable "kube_prometheus_stack_version" {
  description = "kube-prometheus-stack chart version. null => latest. TODO: pin for reproducibility."
  type        = string
  default     = null
}
