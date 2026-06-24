# M6 Piece 1 — Serving Benchmark: vLLM vs naïve FastAPI

**Same model, two serving stacks.** Both serve the Qwen2.5-1.5B-Instruct base + the
`fpb` LoRA adapter, both in **bf16**, on the same GPU. The only variable is the serving
engine. Concepts: `docs/m6-serving-concepts.md` §4–5, §10.

## Setup
- **GPU:** RunPod **L4** (24 GB). (4090 was out of capacity; the vLLM-vs-naïve *gap* is
  hardware-independent.)
- **Precision:** bf16 both sides.
- **Naïve stack:** `serving/bench_naive.py` — FastAPI + HF `generate()` per request, no
  batching.
- **vLLM stack:** `scripts/runpod_vllm.sh` — vLLM 0.6.6, `--enable-lora`, continuous
  batching + PagedAttention.
- **Load:** `pipelines/benchmark_serving.py`, 200 requests per concurrency level, greedy
  decoding (temperature 0), `max_tokens 5` (classification). Sweep: 1/4/8/16/32.
- **Raw data:** `results/benchmark_serving_naive.{json,md}`,
  `results/benchmark_serving_vllm.{json,md}`.

## Results

### Naïve FastAPI
| concurrency | p50 ms | p95 ms | p99 ms | req/s |
|---:|---:|---:|---:|---:|
| 1 | 108.0 | 160.8 | 163.2 | 8.01 |
| 4 | 608.1 | 837.0 | 1222.3 | 6.40 |
| 8 | 1294.4 | 1941.2 | 1987.3 | 5.32 |
| 16 | 2589.8 | 3939.6 | 4039.2 | 5.20 |
| 32 | 5895.6 | 10329.3 | 10666.6 | 4.71 |

### vLLM
| concurrency | p50 ms | p95 ms | p99 ms | req/s |
|---:|---:|---:|---:|---:|
| 1 | 95.3 | 113.5 | 115.2 | 10.62 |
| 4 | 138.4 | 206.0 | 215.8 | 25.57 |
| 8 | 141.8 | 210.6 | 214.0 | 48.29 |
| 16 | 162.0 | 228.6 | 268.2 | 90.28 |
| 32 | 247.3 | 292.1 | 301.6 | 128.21 |

### Head-to-head (at concurrency 32)
| metric | naïve | vLLM | vLLM advantage |
|---|---:|---:|---:|
| throughput (req/s) | 4.71 | 128.21 | **~27×** |
| p99 latency (ms) | 10,666.6 | 301.6 | **~35× lower** |

Trend: as concurrency rises 1→32, naïve **throughput drops** (8.0→4.7) and **p99
explodes ~65×** (163→10,667 ms); vLLM **throughput climbs** (10.6→128.2) and **p99 stays
roughly flat** (115→302 ms).

## Peak GPU memory (KV-cache observation)
**Not captured** during this run (we focused on latency/throughput). Proper GPU-memory
observability — including watching KV-cache VRAM grow with concurrency — lands in
**M6 Piece 4 (DCGM → Prometheus)**.

## Why vLLM wins
<!-- TODO(Karthik): YOUR words (rule 5). One short paragraph. Hit: prefill vs decode,
     why the naïve server leaves the GPU idle / serializes, what continuous batching
     does differently, and (briefly) PagedAttention's role in fitting more concurrent
     requests. See docs/m6-serving-concepts.md §4–5 if you need to refresh. -->

## Reproduce
See `docs/runpod-m6-benchmark.md` (pod setup → run both → pull results → teardown).
