# Kueue — quota-aware batch queueing + gang scheduling for the training plane.
# (Self-manages its webhook certs; no cert-manager needed.) The OCI helm chart REQUIRES
# an explicit version — there is no "latest" for OCI registries.
resource "helm_release" "kueue" {
  name             = "kueue"
  repository       = "oci://registry.k8s.io/kueue/charts"
  chart            = "kueue"
  version          = var.kueue_version
  namespace        = "kueue-system"
  create_namespace = true
}
