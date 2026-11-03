
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/processed/BTCUSDT_research_8h.csv")
OUTPUT = Path("data/processed/BTCUSDT_features_8h.csv")

df = pd.read_csv(INPUT, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Price-based features
df["return_8h"] = df["close"].pct_change(fill_method=None)
df["return_24h"] = df["close"].pct_change(3, fill_method=None)
df["return_72h"] = df["close"].pct_change(9, fill_method=None)

df["realized_vol_72h"] = (
    df["return_8h"].rolling(9).std() * np.sqrt(9)
)

# Target: BTC decline of at least 5% over the next 72 hours
future_return = df["close"].shift(-9) / df["close"] - 1

df["crash_72h"] = (
    future_return.le(-0.05)
    .where(future_return.notna())
    .astype("Int64")
)

features = [
    "oi_change_24h",
    "oi_change_3d",
    "oi_zscore",
    "top_trader_ratio",
    "taker_ratio",
    "count_long_short_ratio",
    "return_8h",
    "return_24h",
    "return_72h",
    "realized_vol_72h",
]

# Replace infinite values and remove invalid observations
df[features] = df[features].replace(
    [np.inf, -np.inf], np.nan
)

invalid_rows = df[features].isna().any(axis=1).sum()

df = df.dropna(subset=features + ["crash_72h"]).copy()
df["crash_72h"] = df["crash_72h"].astype(int)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print(f"Invalid feature rows: {invalid_rows}")
print(f"Dataset saved: {len(df)} observations")
print(
    f"Crashes: {df['crash_72h'].sum()} "
    f"({df['crash_72h'].mean():.2%})"
)
