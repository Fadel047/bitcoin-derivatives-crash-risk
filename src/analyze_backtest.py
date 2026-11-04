
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTIONS = Path("results/models/walk_forward_predictions.csv")
PRICES = Path("data/processed/BTCUSDT_research_8h.csv")
OUTPUT = Path("results/backtest")
OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL = "Price + positioning"
THRESHOLD = 0.15
REDUCED_EXPOSURE = 0.50
COST = 0.001
PERIODS_PER_YEAR = 365 * 3

pred = pd.read_csv(PREDICTIONS, parse_dates=["timestamp"])
prices = pd.read_csv(PRICES, parse_dates=["timestamp"])

pred = pred.sort_values("timestamp").reset_index(drop=True)
prices = prices.sort_values("timestamp").reset_index(drop=True)

if pred["timestamp"].duplicated().any():
    raise ValueError("Duplicate prediction timestamps.")

if prices["timestamp"].duplicated().any():
    raise ValueError("Duplicate price timestamps.")

if not prices["timestamp"].diff().iloc[1:].eq(
    pd.Timedelta(hours=8)
).all():
    raise ValueError("Non-consecutive price timestamps.")

if pred[MODEL].isna().any():
    raise ValueError("Missing predictions.")

prices["btc_return"] = (
    prices["close"].shift(-1) / prices["close"] - 1
)

df = prices[
    prices["timestamp"].between(
        pred["timestamp"].min(),
        pred["timestamp"].max(),
    )
].copy()

df = df.merge(
    pred[["timestamp", "fold", MODEL]],
    on="timestamp",
    how="left",
    validate="one_to_one",
)

df["signal"] = np.where(
    df[MODEL].notna(),
    np.where(df[MODEL] >= THRESHOLD, REDUCED_EXPOSURE, 1.0),
    np.nan,
)

# Keep the last signal when predictions are missing
df["exposure"] = df["signal"].ffill().shift(1)
df["fold"] = df["fold"].ffill()

df = df.dropna(
    subset=["exposure", "btc_return", "fold"]
).reset_index(drop=True)

if not df["timestamp"].diff().iloc[1:].eq(
    pd.Timedelta(hours=8)
).all():
    raise ValueError("Non-consecutive backtest timestamps.")

df["fold"] = df["fold"].astype(int)
CONSTANT_EXPOSURE = df["exposure"].mean()


def simulate(returns, targets, cost_rate):
    wealth = 1.0
    weight = 1.0
    records = []

    for r, target in zip(returns, targets):
        turnover = abs(target - weight)
        cost = cost_rate * turnover

        period_return = (1 - cost) * (1 + target * r) - 1
        wealth *= 1 + period_return

        weight = target * (1 + r) / (1 + target * r)

        records.append((period_return, wealth, turnover, cost))

    return pd.DataFrame(
        records,
        columns=["return", "equity", "turnover", "cost"],
    )


returns = df["btc_return"].to_numpy()

strategies = {
    "Buy & Hold": (np.ones(len(df)), 0.0),
    "Fixed 75% BTC": (np.full(len(df), 0.75), COST),
    "Realized mean exposure": (
        np.full(len(df), CONSTANT_EXPOSURE), COST
    ),
    "Defensive gross": (df["exposure"].to_numpy(), 0.0),
    "Defensive net": (df["exposure"].to_numpy(), COST),
}

summary = []
fold_results = []

for name, (targets, cost_rate) in strategies.items():
    sim = simulate(returns, targets, cost_rate)

    df[name + "_return"] = sim["return"].to_numpy()

    total_return = sim["equity"].iloc[-1] - 1

    summary.append({
        "model": name,
        "total_return": total_return,
        "turnover": sim["turnover"].sum(),
        "transaction_costs": sim["cost"].sum(),
    })

    for fold, group in df.groupby("fold", sort=True):
        r = group[name + "_return"]

        equity = (1 + r).cumprod()
        drawdown = equity / np.maximum(
            1.0, equity.cummax()
        ) - 1

        fold_results.append({
            "fold": fold,
            "model": name,
            "observations": len(group),
            "total_return": equity.iloc[-1] - 1,
            "sharpe": (
                r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR)
                if r.std() > 0 else np.nan
            ),
            "max_drawdown": drawdown.min(),
        })

summary = pd.DataFrame(summary).set_index("model")
fold_results = pd.DataFrame(fold_results)

summary.to_csv(OUTPUT / "cost_analysis.csv")
fold_results.to_csv(OUTPUT / "fold_performance.csv", index=False)

print("\nGross vs net performance:")
print(summary.round(4).to_string())

print("\nPerformance by fold:")
print(
    fold_results.pivot(
        index="fold",
        columns="model",
        values="total_return",
    ).round(4).to_string()
)

print(f"\nResults saved to {OUTPUT}")
