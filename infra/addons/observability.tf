# kube-prometheus-stack (Prometheus + Grafana + Alertmanager + Prometheus Operator CRDs).
# Ephemeral storage (storage decision = Option B): the chart defaults to emptyDir when no
# storageSpec/persistence is configured, so we just don't set PVCs — data is lost on pod
# restart, fine for a destroy-per-session lab.
resource "helm_release" "kube_prometheus_stack" {
  name             = "kube-prometheus-stack"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  version          = var.kube_prometheus_stack_version # null => latest (pin via tfvars)
  namespace        = "monitoring"
  create_namespace = true

  # By default Prometheus only selects ServiceMonitors labelled with THIS release. Set
  # false so it scrapes ServiceMonitors in ALL namespaces — including the GPU Operator's
  # dcgm-exporter ServiceMonitor (in the gpu-operator namespace).
  set {
    name  = "prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues"
    value = "false"
  }
  set {
    name  = "prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues"
    value = "false"
  }
}
