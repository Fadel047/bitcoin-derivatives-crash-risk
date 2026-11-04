
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTIONS = Path("results/models/walk_forward_predictions.csv")
PRICES = Path("data/processed/BTCUSDT_research_8h.csv")
OUTPUT = Path("results/analysis")
OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL = "Price + positioning"
HORIZON = 9
N_GROUPS = 5

pred = pd.read_csv(PREDICTIONS, parse_dates=["timestamp"])
prices = pd.read_csv(PRICES, parse_dates=["timestamp"])

prices = prices.sort_values("timestamp").reset_index(drop=True)

if prices["timestamp"].duplicated().any():
    raise ValueError("Duplicate price timestamps.")

if not prices["timestamp"].diff().iloc[1:].eq(
    pd.Timedelta(hours=8)
).all():
    raise ValueError("Non-consecutive price timestamps.")

prices["forward_return_72h"] = (
    prices["close"].shift(-HORIZON) / prices["close"] - 1
)

df = pred.merge(
    prices[["timestamp", "forward_return_72h"]],
    on="timestamp",
    how="left",
    validate="one_to_one",
)

if df["forward_return_72h"].isna().any():
    raise ValueError("Missing future returns.")

if df[MODEL].isna().any():
    raise ValueError("Missing model predictions.")

# Risk groups based only on predicted probabilities
df["risk_group"] = pd.qcut(
    df[MODEL].rank(method="first"),
    q=N_GROUPS,
    labels=range(1, N_GROUPS + 1),
).astype(int)

def summarize(data):
    return pd.Series({
        "observations": len(data),
        "mean_predicted_risk": data[MODEL].mean(),
        "crash_rate": data["actual"].mean(),
        "mean_return_72h": data["forward_return_72h"].mean(),
        "median_return_72h": data["forward_return_72h"].median(),
    })

summary = (
    df.groupby("risk_group")
    .apply(summarize, include_groups=False)
)

fold_summary = (
    df.groupby(["fold", "risk_group"])
    .apply(summarize, include_groups=False)
)

summary.to_csv(OUTPUT / "risk_groups_summary.csv")
fold_summary.to_csv(OUTPUT / "risk_groups_folds.csv")
df.to_csv(OUTPUT / "risk_groups_predictions.csv", index=False)

print("\nRisk groups (1 = lowest, 5 = highest):")
print(summary.round(4).to_string())

print("\nMean 72h return by risk group:")
print(
    (summary["mean_return_72h"] * 100)
    .round(2)
    .to_string()
)

print("\nCrash rate by risk group:")
print(
    (summary["crash_rate"] * 100)
    .round(2)
    .to_string()
)

print(f"\nResults saved to {OUTPUT}")
