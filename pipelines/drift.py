"""M4.1 — drift detection on the production model via OOV rate.

Signal: out-of-vocabulary (OOV) rate per headline — the fraction of a headline's
words the production model's TF-IDF vocabulary has never seen. When a new market
regime floods in (hawkish/QT/crypto vocabulary), OOV spikes.

Tests: PSI (magnitude, hand-rolled to learn it) + KS (significance, via scipy).
We compare an in-distribution REFERENCE (PhraseBank test set) to a CURRENT batch
(the regime fixture). PSI > 0.2 = drift.
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from baseline import load_data, split_data

MODEL_NAME = "fpb-sentiment"
REGIME_CSV = Path(__file__).resolve().parent / "regime_headlines.csv"
PSI_THRESHOLD = 0.2

_DB = Path(__file__).resolve().parent.parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{_DB}")


def oov_rate(headline: str, analyzer, vocab: set) -> float:
    """Fraction of a headline's tokens NOT in the TF-IDF vocabulary.

    `analyzer` = vectorizer.build_analyzer() (tokenizes exactly like TF-IDF did).
    `vocab`    = set(vectorizer.vocabulary_).
    """
    tokens = analyzer(headline)
    if not tokens:
        return 0.0
    return sum(token not in vocab for token in tokens) / len(tokens)


def psi(reference: list[float], current: list[float], bins: int = 10) -> float:
    """Population Stability Index between two distributions of a [0,1] value.

    PSI = sum over bins of (cur_pct - ref_pct) * ln(cur_pct / ref_pct).
    """
    edges = np.linspace(0, 1, bins + 1)
    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]
    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()
    np.maximum(ref_pct, 1e-6, out=ref_pct)
    np.maximum(cur_pct, 1e-6, out=cur_pct)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_drift() -> None:
    """Compute OOV distributions (reference vs regime), run PSI + KS, print the verdict."""
    # load the production model and pull its fitted TF-IDF out of the pipeline
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")
    vectorizer = model.named_steps["tfidf"]
    analyzer = vectorizer.build_analyzer()
    vocab = set(vectorizer.vocabulary_)

    # REFERENCE = in-distribution (PhraseBank test set) ; CURRENT = regime fixture
    _, _, test_df = split_data(load_data())
    regime_df = pd.read_csv(REGIME_CSV)

    reference_oov = [oov_rate(t, analyzer, vocab) for t in test_df["text"]]
    current_oov = [oov_rate(t, analyzer, vocab) for t in regime_df["text"]]

    psi_val = psi(reference_oov, current_oov)
    ks_stat, ks_p = ks_2samp(reference_oov, current_oov)
    drift = psi_val > PSI_THRESHOLD
    print(
        f"mean_oov_ref={np.mean(reference_oov):.4f} "
        f"mean_oov_current={np.mean(current_oov):.4f} "
        f"psi={psi_val:.4f} "
        f"ks_stat={ks_stat:.4f} "
        f"ks_p={ks_p:.4g} "
        f"DRIFT: {'yes' if drift else 'no'}"
    )


def detect_drift_evidently() -> None:
    """Same OOV drift check, via Evidently: build datasets, run DataDriftPreset, save report."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")
    vectorizer = model.named_steps["tfidf"]
    analyzer = vectorizer.build_analyzer()
    vocab = set(vectorizer.vocabulary_)

    _, _, test_df = split_data(load_data())
    regime_df = pd.read_csv(REGIME_CSV)

    ref_df = pd.DataFrame(
        {"oov_rate": [oov_rate(t, analyzer, vocab) for t in test_df["text"]]}
    )
    cur_df = pd.DataFrame(
        {"oov_rate": [oov_rate(t, analyzer, vocab) for t in regime_df["text"]]}
    )

    data_definition = DataDefinition(numerical_columns=["oov_rate"])
    ref = Dataset.from_pandas(ref_df, data_definition=data_definition)
    cur = Dataset.from_pandas(cur_df, data_definition=data_definition)

    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=cur, reference_data=ref)

    snapshot.save_html("docs/drift_report.html")
    print(snapshot.dict())


if __name__ == "__main__":
    detect_drift()
    detect_drift_evidently()
