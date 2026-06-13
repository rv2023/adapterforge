"""M2 Step 5 — overfit on purpose, plot the train-vs-val loss curve.

The recipe for guaranteed overfitting: a TINY training set + MANY epochs. Starve
the model of data and let it stare at the same few rows over and over — it will
memorize them (train loss -> ~0) while doing worse on validation (val loss turns
up). We run the epochs ourselves with partial_fit so we can measure BETWEEN steps
and draw the curve a single fit() would hide.

Reuses load_data / split_data from the baseline.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

from baseline import RANDOM_SEED, load_data, split_data

N_EPOCHS = 50      # how many rounds (steps downhill) to run and measure
TRAIN_ROWS = 500   # starve the model: only this many training rows -> overfits


def run_overfit_demo() -> None:
    """Train round-by-round on a tiny set; record train/val loss each epoch; plot."""
    df = load_data()
    train_df, val_df, _test_df = split_data(df)   # test stays sealed, unused here

    train_df = train_df.head(TRAIN_ROWS)

    vectorizer = TfidfVectorizer(dtype=np.float32)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    y_train = train_df["label"]
    y_val = val_df["label"]

    #model = SGDClassifier(loss="log_loss", random_state=RANDOM_SEED)
    model = SGDClassifier(loss="log_loss", penalty=None, random_state=RANDOM_SEED)

    classes = np.unique(y_train)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for _epoch in range(N_EPOCHS):
        model.partial_fit(X_train, y_train, classes=classes)
        train_losses.append(log_loss(y_train, model.predict_proba(X_train), labels=classes))
        val_losses.append(log_loss(y_val, model.predict_proba(X_val), labels=classes))

    epochs = range(1, N_EPOCHS + 1)
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("log loss")
    plt.title("Train vs validation loss while overfitting")
    plt.legend()
    plt.tight_layout()
    plt.savefig("docs/overfit_curve.png")

    best_epoch = int(np.argmin(val_losses)) + 1
    print(f"Lowest val_loss at epoch {best_epoch}")


if __name__ == "__main__":
    run_overfit_demo()
