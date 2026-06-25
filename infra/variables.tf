# Inputs. Defaults are sane for the M7 lab; override in terraform.tfvars if you like.
variable "region" {
  type    = string
  default = "us-east-1"
}

variable "cluster_name" {
  type    = string
  default = "adapterforge"
}

variable "eks_version" {
  type    = string
  default = "1.31"
}

variable "vpc_cidr" {
  type    = string
  default = "10.0.0.0/16"
  # TODO: confirm this doesn't collide with anything you peer with later.
}

variable "az_count" {
  description = "How many AZs to spread subnets across."
  type        = number
  default     = 2 # 2 keeps NAT/subnet cost down; bump to 3 for more HA.
}

# --- GPU node group cost lever ---------------------------------------------
variable "gpu_instance_type" {
  type    = string
  default = "g5.xlarge" # 1x A10G, ~$1/hr on-demand
}

variable "gpu_desired_size" {
  description = "GPU nodes to run. KEEP 0 between sessions; scale to 1 only when testing."
  type        = number
  default     = 0
}

variable "gpu_max_size" {
  type    = number
  default = 2
}
