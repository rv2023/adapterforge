# M7 addons — SEPARATE root module from ../ (its own state) so the helm/kubernetes
# providers authenticate to an ALREADY-EXISTING cluster. This avoids the provider
# chicken-and-egg you hit when managing a cluster + its in-cluster charts in one state.
# See docs/m7-concepts.md (KServe/addons discussion) + m7-session-notes.
terraform {
  required_version = ">= 1.5"

  required_providers {
    aws        = { source = "hashicorp/aws", version = "~> 5.0" }
    helm       = { source = "hashicorp/helm", version = "~> 2.12" }
    kubernetes = { source = "hashicorp/kubernetes", version = "~> 2.30" }
  }
  # local state for now (separate file from ../). Apply ../ FIRST (cluster), then here.
}
