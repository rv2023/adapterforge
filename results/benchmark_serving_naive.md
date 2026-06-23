# Serving Benchmark: naive

- URL: `http://localhost:8002`
- Model: `None`
- Requests per concurrency level: `200`

| concurrency | p50 ms | p95 ms | p99 ms | req/s |
|---:|---:|---:|---:|---:|
| 1 | 108.0 | 160.8 | 163.2 | 8.01 |
| 4 | 608.1 | 837.0 | 1222.3 | 6.40 |
| 8 | 1294.4 | 1941.2 | 1987.3 | 5.32 |
| 16 | 2589.8 | 3939.6 | 4039.2 | 5.20 |
| 32 | 5895.6 | 10329.3 | 10666.6 | 4.71 |
