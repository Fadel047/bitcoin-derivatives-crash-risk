
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/processed/BTCUSDT_research_8h.csv")
HORIZONS = [1, 3, 9]  # 8h, 24h, 72h
THRESHOLDS = [0.02, 0.03, 0.05]

df = pd.read_csv(DATA_PATH)

price_col = next(
    col for col in ["close", "spot_close", "Close"]
    if col in df.columns
)

price = pd.to_numeric(df[price_col], errors="coerce")

results = []

for horizon in HORIZONS:
    future_return = price.shift(-horizon) / price - 1
    valid = future_return.dropna()

    for threshold in THRESHOLDS:
        crashes = valid <= -threshold

        results.append({
            "horizon_h": horizon * 8,
            "threshold": f"-{threshold:.0%}",
            "observations": len(valid),
            "crashes": int(crashes.sum()),
            "frequency_pct": round(crashes.mean() * 100, 2),
        })

summary = pd.DataFrame(results)

print("\nCrash target analysis\n")
print(summary.to_string(index=False))
