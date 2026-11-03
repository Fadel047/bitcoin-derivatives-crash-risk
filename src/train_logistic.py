
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

df = pd.read_csv(DATA, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

target = "crash_72h"

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

# Chronological split with a 72-hour purge
split = int(len(df) * 0.7)
test_start = df.loc[split, "timestamp"]
train = df[
    df["timestamp"] + pd.Timedelta(hours=72) < test_start
].copy()
test = df.iloc[split:].copy()

y_train = train[target]
y_test = test[target]

print(f"Train: {len(train)} observations, {y_train.mean():.2%} crashes")
print(f"Test:  {len(test)} observations, {y_test.mean():.2%} crashes")

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError("Both train and test must contain both classes.")

def evaluate(name, probabilities):
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "brier": brier_score_loss(y_test, probabilities),
        "log_loss": log_loss(y_test, probabilities),
    }

# Constant-probability benchmark
baseline = np.full(len(test), y_train.mean())
results = [evaluate("Historical baseline", baseline)]

predictions = pd.DataFrame({
    "timestamp": test["timestamp"],
    "actual": y_test,
    "baseline": baseline,
})

for name, features in models.items():
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )

    model.fit(train[features], y_train)
    probabilities = model.predict_proba(test[features])[:, 1]

    results.append(evaluate(name, probabilities))
    predictions[name] = probabilities

    coefficients = pd.Series(
        model.named_steps["logisticregression"].coef_[0],
        index=features,
        name="coefficient",
    )
    coefficients.to_csv(
        OUTPUT / f"{name.lower().replace(' ', '_').replace('+', 'plus')}_coefficients.csv"
    )

summary = pd.DataFrame(results).set_index("model")

summary.to_csv(OUTPUT / "logistic_results.csv")
predictions.to_csv(OUTPUT / "logistic_predictions.csv", index=False)

print("\nOut-of-sample performance:")
print(summary.round(4).to_string())
print(f"\nResults saved to {OUTPUT}")
