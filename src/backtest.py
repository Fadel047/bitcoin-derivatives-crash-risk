
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
TRANSACTION_COST = 0.001
PERIODS_PER_YEAR = 365 * 3
STEP = pd.Timedelta(hours=8)

pred = pd.read_csv(PREDICTIONS, parse_dates=["timestamp"])
prices = pd.read_csv(PRICES, parse_dates=["timestamp"])

pred = pred.sort_values("timestamp").reset_index(drop=True)
prices = prices.sort_values("timestamp").reset_index(drop=True)

for name, data in [("Predictions", pred), ("Prices", prices)]:
    if data["timestamp"].duplicated().any():
        raise ValueError(f"{name}: duplicate timestamps.")

if not prices["timestamp"].diff().iloc[1:].eq(STEP).all():
    raise ValueError("Non-consecutive price timestamps.")

if pred[MODEL].isna().any():
    raise ValueError("Missing predictions.")

# Each return covers timestamp t to t + 8h
prices["btc_return"] = (
    prices["close"].shift(-1) / prices["close"] - 1
)

start = pred["timestamp"].min()
end = pred["timestamp"].max()

df = prices[
    (prices["timestamp"] >= start)
    & (prices["timestamp"] <= end)
].copy()

df = df.merge(
    pred[["timestamp", MODEL]],
    on="timestamp",
    how="left",
    validate="one_to_one",
)

# Missing predictions do not create new signals
df["signal"] = np.where(
    df[MODEL].notna(),
    np.where(
        df[MODEL] >= THRESHOLD,
        REDUCED_EXPOSURE,
        1.0,
    ),
    np.nan,
)

# Conservative one-bar execution delay
df["exposure"] = df["signal"].ffill().shift(1)

df = df.dropna(
    subset=["exposure", "btc_return"]
).reset_index(drop=True)

if df.empty:
    raise ValueError("No valid backtest observations.")

if not df["timestamp"].diff().iloc[1:].eq(STEP).all():
    raise ValueError("Non-consecutive backtest timestamps.")

CONSTANT_EXPOSURE = df["exposure"].mean()


def simulate(returns, targets, cost_rate):
    wealth = 1.0
    weight = 1.0
    records = []

    for r, target in zip(returns, targets):
        turnover = abs(target - weight)
        cost = cost_rate * turnover

        wealth_before = wealth
        wealth *= 1 - cost

        portfolio_return = target * r
        wealth *= 1 + portfolio_return

        weight = (
            target * (1 + r) / (1 + portfolio_return)
        )

        records.append({
            "return": wealth / wealth_before - 1,
            "equity": wealth,
            "turnover": turnover,
            "cost": cost,
        })

    return pd.DataFrame(records)


btc_returns = df["btc_return"].to_numpy()

strategies = {
    "Buy & Hold": np.ones(len(df)),
    "Fixed 75% BTC": np.full(len(df), 0.75),
    "Realized mean exposure": np.full(
        len(df), CONSTANT_EXPOSURE
    ),
    "Defensive BTC": df["exposure"].to_numpy(),
}

simulations = {}
summary = []

for name, targets in strategies.items():
    simulation = simulate(
        btc_returns,
        targets,
        TRANSACTION_COST,
    )

    simulations[name] = simulation

    returns = simulation["return"]
    equity = simulation["equity"]

    volatility = returns.std() * np.sqrt(PERIODS_PER_YEAR)

    sharpe = (
        returns.mean() / returns.std()
        * np.sqrt(PERIODS_PER_YEAR)
        if returns.std() > 0 else np.nan
    )

    drawdown = (
        equity / np.maximum(1.0, equity.cummax()) - 1
    )

    summary.append({
        "model": name,
        "total_return": equity.iloc[-1] - 1,
        "annual_return": (
            equity.iloc[-1] ** (PERIODS_PER_YEAR / len(df)) - 1
        ),
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "average_exposure": targets.mean(),
        "total_turnover": simulation["turnover"].sum(),
        "transaction_costs": simulation["cost"].sum(),
    })

    df[f"{name}_return"] = simulation["return"].to_numpy()
    df[f"{name}_equity"] = simulation["equity"].to_numpy()

summary = pd.DataFrame(summary).set_index("model")

df.to_csv(OUTPUT / "backtest_timeseries.csv", index=False)
summary.to_csv(OUTPUT / "backtest_summary.csv")

print(f"Backtest observations: {len(df)}")
print(
    f"Period: {df['timestamp'].min()} "
    f"to {df['timestamp'].max()}"
)
print(f"Risk threshold: {THRESHOLD:.0%}")
print(f"Constant exposure benchmark: {CONSTANT_EXPOSURE:.2%}")

print("\nBacktest performance:")
print(summary.round(4).to_string())

print(f"\nResults saved to {OUTPUT}")