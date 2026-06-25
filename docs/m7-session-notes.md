# M7 — Kubernetes GPU Platform: Session Notes

Running log of M7. Concepts in `docs/m7-concepts.md`; IaC in `infra/`.

## Plan (decided at kickoff)

Free-first: build/rehearse on local `kind` → one tight paid EKS+GPU session → destroy.
- **Calibration:** Karthik is **solid on core K8s** → assume it, focus on GPU/MLOps parts.
- **Start:** write the Terraform first (free).
- Pieces: Terraform EKS+GPU (IaC) · GPU Operator · MIG + time-slicing (carried from M6) ·
  Kueue (quota + gang) · in-place resize (<1 min) · kube-prometheus-stack + DCGM ·
  RCA bot (<10 min) · >98% SLO dashboard.
- Free on kind: Kueue, in-place resize, kube-prometheus-stack/Grafana/SLO, RCA bot logic.
  Paid (EKS+GPU): apply itself, GPU Operator, MIG/time-slicing, DCGM-on-GPU.

## Session 1 — 2026-06-25 (M7 kickoff + Terraform authored + plan verified)

**Built `infra/` (Terraform EKS + GPU node group):** versions/providers/variables/vpc/
eks/outputs + terraform.tfvars.example. Community modules; system + GPU node groups; GPU
**desired=0** cost lever; base AL2023 AMI (Operator installs drivers); on-demand GPU;
single NAT; local state; us-east-1, EKS 1.31; IRSA on; nvidia.com/gpu taint.
`terraform validate` → valid. Commits: 5888e6d (module), dc2321c (tfvars.example),
2821584 (gitignore tfplan).

**`terraform plan` reviewed (free, creates nothing):** **61 to add, 0/0**; 1 NAT;
GPU node group desired=0/min=0/max=1, g5.xlarge, base AMI; system desired=1. tfvars
copied (max_size=1 applied). Cost to apply w/ GPU=0 ≈ ~$0.20/hr; +~$1/hr per GPU.

**Decision: APPLY (Option A)** — Karthik running `terraform apply tfplan` (his AWS
creds; I don't run spend commands). Teardown ready: `cd infra && terraform destroy`.

**Concepts taught + saved** (`docs/m7-concepts.md`): Terraform design rationale; **Kueue**
(why vanilla scheduler fails, object model, suspend→admit flow, gang scheduling, quota/
borrow/preempt, single-cluster vs MultiKueue); **Kueue vs KEDA**; **NCCL** (all-reduce,
intra/inter-node, tie to gang); **namespaces** (multi-tenancy; not strictly needed solo);
**KServe** (serverless inference for K8s; InferenceService; wraps engines; scale-to-zero +
canary = M8 hot-swap; v2 protocol = the one triton_client already speaks).

**Next:** finish apply → `aws eks update-kubeconfig` → `kubectl get nodes` (1 system node,
no GPU yet). Then GPU Operator (scale GPU to 1 when ready) OR start the free kind pieces.
**Remember to `terraform destroy` at end of session.** git push pending (M6+M7 commits).

**Apply DONE** (2026-06-25): cluster `adapterforge` up, 1 t3.medium system node Ready,
core pods (aws-node/coredns/kube-proxy) Running. GPU node desired=0 (none yet).
**Addon decisions:** storage = **ephemeral/emptyDir** for observability (Option B — no
EBS CSI; cluster is destroy-per-session). **ALB controller + ExternalDNS skipped** →
deferred to M8 (external serving exposure); M7 uses `kubectl port-forward`.
**DONE this session (2026-06-25):**
- `infra/` applied (cluster `adapterforge` live; GPU node group ami_type → AL2023_x86_64_NVIDIA).
- Scaled GPU node to 1 **via AWS CLI** (`aws eks update-nodegroup-config ... desiredSize=1`) —
  the EKS module **ignore_changes** on `scaling_config.desired_size` (autoscaler coexistence),
  so tfvars `gpu_desired_size` does nothing after create; desired is managed out-of-band.
- **GPU Operator provisioned the node** (Opt 2): device-plugin + container-toolkit +
  **dcgm-exporter** + gpu-feature-discovery + validators all Running; cuda-validator
  Completed (driver+CUDA work). **No mig-manager** — A10G (g5.xlarge) doesn't support MIG.
- **Piece 4 DONE**: `infra/addons/` now has kube-prometheus-stack (ephemeral) +
  dcgm ServiceMonitor → Prometheus scrapes GPU metrics → Grafana. (Optional load-graph
  via a gpu-burn pod.)
- **⚠️ MIG (Piece 3) blocked on hardware:** needs A100/H100 (p4d/p5) — expensive + AWS
  quota-gated. Decide at Piece 3: short A100-node session vs theory; time-slicing (no-iso
  half) works on A10G.

**Remaining M7 (all non-GPU → do with GPU=0 or on kind):** Kueue (quota+gang), in-place
resize (<1 min), RCA bot (<10 min), >98% SLO dashboard.

**Driver decision RESOLVED → Opt 2** (see m7-concepts §7): GPU node
`ami_type = AL2023_x86_64_NVIDIA` (driver+runtime prebaked by AWS) + GPU Operator
`--set driver.enabled=false` (Operator supplies device-plugin + DCGM + MIG-manager + NFD).
Why: the GPU-node stack = driver/runtime (AMI) + device-plugin + DCGM + MIG (Operator);
bare device-plugin alone (AWS default) has no DCGM/MIG so fails M7; Operator-managed
driver needs Ubuntu (CUSTOM AMI) — Opt 2 is the reliable, representative EKS pattern and
covers Pieces 3+4. **TODO:** edit `infra/eks.tf` gpu ami_type STANDARD→NVIDIA + re-apply
(GPU still desired=0, just a launch-template change), then `helm install gpu-operator
... --set driver.enabled=false`, then scale GPU to 1 and watch operands roll out.
