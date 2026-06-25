"""M8 — evaluate the summarization adapter on ECTSum: ROUGE-L vs base zero-shot.

The scope bar: ROUGE-L(adapter) > ROUGE-L(base zero-shot). Mirrors eval_adapter.py but
the task is generation (longer outputs) and the metric is ROUGE-L, not macro-F1.

    python -m pipelines.eval_summarizer        # (on the GPU pod, after training)
"""

import json
from pathlib import Path

ADAPTER_DIR = "models/fpb-summarizer"
TEST_FILE = "data/instruction_summ/test.jsonl"
MAX_NEW_TOKENS = 160  # summaries are multi-sentence, not a single label word


def generate_summary(model, tokenizer, messages) -> str:
    """[system, user] -> the model's generated summary (greedy)."""
    # TODO: reuse the eval_adapter generate pattern but with MAX_NEW_TOKENS, decode the
    # NEW tokens only (skip the prompt), return the text. (No label-parsing — it's free text.)
    raise NotImplementedError


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure between one prediction and its reference summary."""
    # TODO: from rouge_score import rouge_scorer   (pip install rouge-score)
    #       scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    #       return scorer.score(reference, prediction)["rougeL"].fmeasure
    raise NotImplementedError


def main() -> float:
    """Score the adapter on ECTSum test; ALSO score base zero-shot (the bar). Persist + return.

    The bar = ROUGE-L(adapter) must beat ROUGE-L(base, no adapter). Print both.
    """
    # TODO: load adapter:  eval_adapter.load_model_and_tokenizer(adapter_dir=ADAPTER_DIR)
    # TODO: load base (no adapter) for the zero-shot bar (same base, no PeftModel)
    # TODO: for each test line: build [system,user] (summarize_format.build_chat_messages),
    #       generate with each model, accumulate rouge_l vs the reference (messages[-1] summary)
    # TODO: rouge_adapter = mean(...); rouge_base = mean(...)
    # TODO: write {"test_f1": rouge_adapter, "n_test": n, "metric": "rougeL",
    #              "base_rougeL": rouge_base} to ADAPTER_DIR/eval_metrics.json
    #       (register_adapter reads "test_f1" as the gate score — store ROUGE-L there;
    #        see the gate-is-sentiment-pinned note before promoting.)
    # TODO: print(f"rougeL adapter={...} base={...} beat_bar={adapter>base}"); return rouge_adapter
    raise NotImplementedError


if __name__ == "__main__":
    main()
