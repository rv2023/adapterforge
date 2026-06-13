"""M2 baseline — TF-IDF + Logistic Regression for financial sentiment.

Trains the cheapest model that could possibly work on Financial PhraseBank and
reports its macro-F1 on a FROZEN test set. That number is "the bar" every later
model (the M5 LLM, the M5 student) must beat on this same sealed data.

Pipeline:  load silver parquet -> split 70/15/15 -> TF-IDF + LogReg -> score test

This module is a plain script for now. In M4 it gets wrapped in a Dagster DAG.
"""

from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn

# The SDK lands the validated (silver) PhraseBank here. The baseline reads silver,
# NOT raw — we want the clean, schema-checked rows.
DATA_PATH = Path("data/validated/financial_phrasebank.parquet")

# Fixed seed so the split is identical every run. This is what makes the frozen
# test set actually frozen: same seed -> same rows land in test every single time.
RANDOM_SEED = 42


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the validated PhraseBank parquet into a DataFrame.

    Returns a table with columns ``text`` and ``label`` (bullish/bearish/neutral).
    """
    return pd.read_parquet(path)


def split_data(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / val / test = 70 / 15 / 15, returned in that order.

    Two properties matter and you must get BOTH right:
      - STRATIFY by label, so each split keeps the same bullish/bearish/neutral
        proportions as the whole dataset (don't let all the rare bearish rows
        accidentally pile into one split).
      - DETERMINISTIC via `seed`, so test is the same sealed rows every run.
    """
    train_df, holdout_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=0.50,
        stratify=holdout_df["label"],
        random_state=seed,
    )
    return train_df, val_df, test_df


def build_model(c: float = 1.0):
    """Build the text-classifier: TF-IDF vectorizer -> Logistic Regression.

    Return a single sklearn object that does text -> label end to end, so callers
    never touch the two stages separately.
    """
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            (
                "logreg",
                LogisticRegression(C=c, class_weight="balanced", max_iter=1000),
            ),
        ]
    )


def evaluate(model, texts, labels) -> float:
    """Score a fitted model on the given texts/labels. Returns macro-F1.

    Use MACRO averaging (compute F1 per class, then average the three equally).
    Why macro and not the default: macro refuses to let the big neutral class
    drown out bullish/bearish — every class counts the same. That's the honest
    bar for an imbalanced task.
    """
    return f1_score(labels, model.predict(texts), average="macro")


def main() -> None:
    """Run the C-sweep tuning runs and log each to MLflow.

    Train on TRAIN, score on VALIDATION. Test stays sealed (bar already locked).
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("m2-baseline")
    df = load_data()
    train_df, val_df, test_df = split_data(df)

    with mlflow.start_run():
        c = 10
        model = build_model(c)
        fit_start = perf_counter()
        model.fit(train_df["text"], train_df["label"])   # ← THE FIX: train on TRAIN
        step_time_sec = perf_counter() - fit_start
        samples_per_sec = len(train_df) / step_time_sec
        val_f1 = evaluate(model, val_df["text"], val_df["label"])   # score on VALIDATION

        mlflow.log_param("C", c)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_metric("val_f1", val_f1)
        mlflow.log_metric("step_time_sec", step_time_sec)
        mlflow.log_metric("samples_per_sec", samples_per_sec)
        mlflow.sklearn.log_model(model, name="model")

        print(f"C={c}  validation macro-F1: {val_f1:.4f}")   # ← label fixed: validation, not "LOCKED BAR/TEST"


if __name__ == "__main__":
    main()
