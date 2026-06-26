"""M8 — evaluate the summarization adapter on ECTSum: ROUGE-L vs base zero-shot.

The scope bar: ROUGE-L(adapter) > ROUGE-L(base zero-shot). Mirrors eval_adapter.py but
the task is generation (longer outputs) and the metric is ROUGE-L, not macro-F1.

    python -m pipelines.eval_summarizer        # (on the GPU pod, after training)
"""

import json
from pathlib import Path

import torch

ADAPTER_DIR = "models/fpb-summarizer"
TEST_FILE = "data/instruction_summ/test.jsonl"
MAX_NEW_TOKENS = 160  # summaries are multi-sentence, not a single label word
_ROUGE_SCORER = None


def generate_summary(model, tokenizer, messages) -> str:
    """[system, user] -> the model's generated summary (greedy)."""
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    prompt_len = ids["input_ids"].shape[-1]
    out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure between one prediction and its reference summary."""
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        from rouge_score import rouge_scorer

        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _ROUGE_SCORER.score(reference, prediction)["rougeL"].fmeasure


def main() -> float:
    """Score the adapter on ECTSum test; ALSO score base zero-shot (the bar). Persist + return.

    The bar = ROUGE-L(adapter) must beat ROUGE-L(base, no adapter). Print both.
    """
    from pipelines import eval_adapter, summarize_format

    model, tokenizer = eval_adapter.load_model_and_tokenizer(adapter_dir=ADAPTER_DIR)

    adapter_scores = []
    base_scores = []
    with torch.inference_mode(), open(TEST_FILE, encoding="utf-8") as f:
        for line in f:
            messages = json.loads(line)["messages"]
            reference = messages[-1]["content"].strip()
            user_content = messages[-2]["content"]
            if user_content.startswith(summarize_format.INSTRUCTION):
                transcript = user_content[len(summarize_format.INSTRUCTION) :]
            else:
                transcript = user_content
            prompt = summarize_format.build_chat_messages(transcript)

            adapter_scores.append(rouge_l(generate_summary(model, tokenizer, prompt), reference))
            with model.disable_adapter():
                base_scores.append(rouge_l(generate_summary(model, tokenizer, prompt), reference))

    n = len(adapter_scores)
    rouge_adapter = sum(adapter_scores) / n
    rouge_base = sum(base_scores) / n
    metrics_path = Path(ADAPTER_DIR) / "eval_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "test_f1": rouge_adapter,
                "n_test": n,
                "metric": "rougeL",
                "base_rougeL": rouge_base,
            }
        ),
        encoding="utf-8",
    )
    print(
        f"rougeL adapter={rouge_adapter:.4f} base={rouge_base:.4f} "
        f"beat_bar={rouge_adapter > rouge_base}"
    )
    return rouge_adapter


if __name__ == "__main__":
    main()
