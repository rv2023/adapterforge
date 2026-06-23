"""M6 Piece 1 — serving load-test harness: naïve FastAPI vs vLLM, SAME model.

Fires a fixed prompt set at a server under increasing concurrency and records
latency percentiles (p50/p95/p99) + throughput (req/s). Run it twice — once at the
naïve FastAPI, once at vLLM — and compare. The whole deliverable is the contrast:
naïve p99 balloons with concurrency; vLLM's stays flat (continuous batching).
See docs/m6-serving-concepts.md §4.

    # naïve server (our FastAPI + generate per request):
    python -m pipelines.benchmark_serving --server naive --url http://localhost:8001
    # vLLM (OpenAI-compatible):
    python -m pipelines.benchmark_serving --server vllm  --url http://localhost:8000 --model fpb
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

CONCURRENCY_LEVELS = [1, 4, 8, 16, 32]
N_REQUESTS = 200            # requests fired per concurrency level
TEST_FILE = "data/instruction/test.jsonl"
INSTRUCTION_PREFIX = (
    "Classify the financial sentiment of the following statement as exactly one "
    "of bullish, bearish, or neutral.\n\nStatement: "
)


def load_prompts(n: int) -> list[str]:
    """Return n raw statement strings to classify (reuse the frozen test set).

    Each test.jsonl line is {"messages":[system, user, assistant]}. We want the bare
    statement the user asks about — strip the INSTRUCTION prefix back off, OR just feed
    the whole user content (decide: the two servers must receive the SAME text).
    """
    prompts = []
    with open(TEST_FILE, encoding="utf-8") as f:
        for line in f:
            user_content = json.loads(line)["messages"][1]["content"]
            if user_content.startswith(INSTRUCTION_PREFIX):
                user_content = user_content[len(INSTRUCTION_PREFIX) :]
            prompts.append(user_content)
            if len(prompts) == n:
                break
    return prompts


# ---------------------------------------------------------------------------
# Per-server request senders.  Each returns a zero-arg callable send() that does ONE
# request for the given prompt and raises on a bad response (so a failure isn't timed
# as a success).  Same model, two wire formats.
# ---------------------------------------------------------------------------
def naive_sender(url: str, prompt: str):
    """Build a send() that POSTs to our FastAPI /predict {"text": prompt}."""
    def send():
        response = requests.post(f"{url}/predict", json={"text": prompt})
        response.raise_for_status()
        return response.json()["label"]

    return send


def vllm_sender(url: str, model: str, prompt: str):
    """Build a send() that POSTs to vLLM's OpenAI-compatible completion endpoint.

    Must mirror the naïve path's generation: same prompt text, greedy (temperature=0),
    small max_tokens (this is classification — a handful of tokens).
    """
    endpoint = f"{url}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 5,
        "temperature": 0,
    }

    def send():
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["text"]

    return send


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def time_one(send) -> float:
    """Run one request, return its latency in seconds."""
    t0 = time.perf_counter()
    send()
    return time.perf_counter() - t0


def run_level(make_send, concurrency: int, n_requests: int) -> dict:
    """Fire n_requests at the given concurrency; return latencies + wall time.

    make_send(i) -> a zero-arg send() for request i (lets you vary the prompt per call).
    Wall time is measured around the WHOLE batch (that's what throughput divides by).
    """
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        latencies = list(pool.map(lambda i: time_one(make_send(i)), range(n_requests)))
    wall = time.perf_counter() - t0
    return {"latencies": latencies, "wall_s": wall}


def summarize(latencies: list[float], wall_s: float) -> dict:
    """p50/p95/p99 (ms) + throughput (req/s) for one level."""
    sorted_latencies = sorted(latencies)
    last = len(sorted_latencies) - 1

    def percentile(p: float) -> float:
        return sorted_latencies[int(p * last)] * 1000

    return {
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "req_s": len(latencies) / wall_s,
    }


def main() -> None:
    """Parse --server/--url/--model, sweep CONCURRENCY_LEVELS, print + persist a table."""
    parser = argparse.ArgumentParser(description="Benchmark serving latency and throughput.")
    parser.add_argument("--server", choices=("naive", "vllm"), required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.server == "vllm" and not args.model:
        parser.error("--model is required when --server vllm")
    if args.server == "naive" and args.model:
        parser.error("--model is only valid when --server vllm")
    prompts = load_prompts(N_REQUESTS)
    prompt_count = len(prompts)
    url = args.url
    if args.server == "naive":
        def make_send(i, _prompts=prompts, _prompt_count=prompt_count, _url=url):
            return naive_sender(_url, _prompts[i % _prompt_count])
    elif args.server == "vllm":
        model = args.model

        def make_send(i, _prompts=prompts, _prompt_count=prompt_count, _url=url, _model=model):
            return vllm_sender(_url, _model, _prompts[i % _prompt_count])
    else:
        raise ValueError(f"unsupported server: {args.server}")
    rows = []
    for c in CONCURRENCY_LEVELS:
        result = run_level(make_send, c, N_REQUESTS)
        summary = summarize(result["latencies"], result["wall_s"])
        rows.append({"concurrency": c, **summary})

    header = ("concurrency", "p50", "p95", "p99", "req/s")
    table_rows = [
        (
            str(row["concurrency"]),
            f"{row['p50_ms']:.1f}",
            f"{row['p95_ms']:.1f}",
            f"{row['p99_ms']:.1f}",
            f"{row['req_s']:.2f}",
        )
        for row in rows
    ]
    widths = [
        max(len(header[i]), *(len(row[i]) for row in table_rows))
        for i in range(len(header))
    ]
    lines = [
        " | ".join(header[i].rjust(widths[i]) for i in range(len(header))),
        "-|-".join("-" * widths[i] for i in range(len(header))),
    ]
    lines.extend(
        " | ".join(row[i].rjust(widths[i]) for i in range(len(row)))
        for row in table_rows
    )
    table = "\n".join(lines)
    print(table)

    output_stem = f"benchmark_serving_{args.server}"
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    payload = {
        "server": args.server,
        "url": args.url,
        "model": args.model,
        "n_requests": N_REQUESTS,
        "concurrency_levels": CONCURRENCY_LEVELS,
        "rows": rows,
    }
    json_path = output_dir / f"{output_stem}.json"
    md_path = output_dir / f"{output_stem}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(
        f"# Serving Benchmark: {args.server}\n\n"
        f"- URL: `{args.url}`\n"
        f"- Model: `{args.model}`\n"
        f"- Requests per concurrency level: `{N_REQUESTS}`\n\n"
        "| concurrency | p50 ms | p95 ms | p99 ms | req/s |\n"
        "|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {row['concurrency']} | {row['p50_ms']:.1f} | {row['p95_ms']:.1f} | "
            f"{row['p99_ms']:.1f} | {row['req_s']:.2f} |"
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
