
from pathlib import Path

import numpy as np
import pandas as pd

POSITIONING = Path("data/processed/BTCUSDT_positioning_8h.csv")
PRICES = Path("data/raw/prices/BTCUSDT_spot_8h.csv")
OUTPUT = Path("data/processed/BTCUSDT_research_8h.csv")


def main():
    positioning = pd.read_csv(POSITIONING)
    prices = pd.read_csv(PRICES)

    for df in (positioning, prices):
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True
        )

    df = positioning.merge(
        prices[["timestamp", "close"]],
        on="timestamp",
        how="inner",
        validate="one_to_one"
    ).sort_values("timestamp").reset_index(drop=True)

    if len(df) != len(positioning):
        raise ValueError("Positioning and price timestamps do not match")

    if not df["timestamp"].diff().iloc[1:].eq(
        pd.Timedelta(hours=8)
    ).all():
        raise ValueError("Non-consecutive 8h timestamps")

    df["forward_return_8h"] = np.log(
        df["close"].shift(-1) / df["close"]
    )

    df["crash_5pct"] = (
        df["forward_return_8h"] < -0.05
    ).astype("Int64")

    df.loc[df["forward_return_8h"].isna(), "crash_5pct"] = pd.NA

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    events = int(df["crash_5pct"].sum())
    valid = df["crash_5pct"].count()

    print(f"Observations: {len(df):,}")
    print(f"Crash events: {events}")
    print(f"Crash frequency: {events / valid:.2%}")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
