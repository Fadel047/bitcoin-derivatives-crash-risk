
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/processed/BTCUSDT_features_8h.csv")
OUTPUT = Path("results/models")
OUTPUT.mkdir(parents=True, exist_ok=True)

TARGET = "crash_72h"
HORIZON = pd.Timedelta(hours=72)
TEST_WINDOW = pd.Timedelta(days=90)
INITIAL_TRAIN_RATIO = 0.7

price_features = [
    "return_8h",
    "return_24h",
    "return_72h",
    "realized_vol_72h",
]

positioning_features = [
    "oi_change_24h",
    "oi_change_3d",
    "oi_zscore",
    "top_trader_ratio",
    "taker_ratio",
    "count_long_short_ratio",
]

models = {
    "Price only": price_features,
    "Price + positioning": price_features + positioning_features,
}

df = pd.read_csv(DATA, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

required = list(set(
    [TARGET, "timestamp"]
    + price_features
    + positioning_features
))

df = df.dropna(subset=required).reset_index(drop=True)

if df.empty or df["timestamp"].duplicated().any():
    raise ValueError("Dataset is empty or contains duplicate timestamps.")

initial_split = int(len(df) * INITIAL_TRAIN_RATIO)
start = df.loc[initial_split, "timestamp"]
end = df["timestamp"].max()

results = []
predictions = []

fold = 0

while start <= end:
    test_end = start + TEST_WINDOW

    train = df[df["timestamp"] + HORIZON < start]
    test = df[
        (df["timestamp"] >= start)
        & (df["timestamp"] < test_end)
    ]

    if train.empty or test.empty:
        break

    y_train = train[TARGET]
    y_test = test[TARGET]

    if y_train.nunique() < 2:
        raise ValueError(f"Fold {fold}: training has only one class.")

    baseline = np.full(len(test), y_train.mean())

    fold_predictions = pd.DataFrame({
        "timestamp": test["timestamp"].to_numpy(),
        "actual": y_test.to_numpy(),
        "fold": fold,
        "Historical baseline": baseline,
    })

    for name, features in models.items():
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )

        model.fit(train[features], y_train)

        fold_predictions[name] = model.predict_proba(
            test[features]
        )[:, 1]

    predictions.append(fold_predictions)

    for name in ["Historical baseline", *models]:
        probabilities = fold_predictions[name].to_numpy()

        metrics = {
            "fold": fold,
            "model": name,
            "test_start": start,
            "n_train": len(train),
            "n_test": len(test),
            "crash_rate": y_test.mean(),
            "brier": brier_score_loss(y_test, probabilities),
            "log_loss": log_loss(y_test, probabilities, labels=[0, 1]),
            "roc_auc": np.nan,
            "pr_auc": np.nan,
        }

        if y_test.nunique() == 2:
            metrics["roc_auc"] = roc_auc_score(
                y_test, probabilities
            )
            metrics["pr_auc"] = average_precision_score(
                y_test, probabilities
            )

        results.append(metrics)

    fold += 1
    start = test_end

if not predictions:
    raise ValueError("No valid walk-forward folds.")

predictions = pd.concat(predictions, ignore_index=True)
results = pd.DataFrame(results)

summary = []

for name in ["Historical baseline", *models]:
    probabilities = predictions[name].to_numpy()
    actual = predictions["actual"]

    summary.append({
        "model": name,
        "roc_auc": roc_auc_score(actual, probabilities)
        if actual.nunique() == 2 else np.nan,
        "pr_auc": average_precision_score(actual, probabilities)
        if actual.nunique() == 2 else np.nan,
        "brier": brier_score_loss(actual, probabilities),
        "log_loss": log_loss(actual, probabilities, labels=[0, 1]),
    })

summary = pd.DataFrame(summary).set_index("model")

results.to_csv(OUTPUT / "walk_forward_folds.csv", index=False)
predictions.to_csv(
    OUTPUT / "walk_forward_predictions.csv", index=False
)
summary.to_csv(OUTPUT / "walk_forward_summary.csv")

print(f"Walk-forward folds: {fold}")
print(f"Out-of-sample observations: {len(predictions)}")
print(f"Crash rate: {predictions['actual'].mean():.2%}")
print("\nWalk-forward performance:")
print(summary.round(4).to_string())
print(f"\nResults saved to {OUTPUT}")
