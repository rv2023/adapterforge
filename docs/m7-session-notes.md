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
