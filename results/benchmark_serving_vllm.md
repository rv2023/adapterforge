# Serving Benchmark: vllm

- URL: `http://localhost:8000`
- Model: `fpb`
- Requests per concurrency level: `200`

| concurrency | p50 ms | p95 ms | p99 ms | req/s |
|---:|---:|---:|---:|---:|
| 1 | 95.3 | 113.5 | 115.2 | 10.62 |
| 4 | 138.4 | 206.0 | 215.8 | 25.57 |
| 8 | 141.8 | 210.6 | 214.0 | 48.29 |
| 16 | 162.0 | 228.6 | 268.2 | 90.28 |
| 32 | 247.3 | 292.1 | 301.6 | 128.21 |
