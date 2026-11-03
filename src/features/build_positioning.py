
from pathlib import Path
import numpy as np
import pandas as pd

START, END = "2023-01-01", "2025-12-31"

INPUT = Path(
    "data/raw/metrics/BTCUSDT_metrics_2021-01-01_2025-12-31.csv"
)
OUTPUT = Path("data/processed/BTCUSDT_positioning_8h.csv")


def main():
    df = pd.read_csv(INPUT, parse_dates=["create_time"])
    df = df.set_index("create_time").sort_index()

    df = df.loc[START:END].drop(columns="symbol")

    data = df.resample("8h", label="right", closed="left").last()

    counts = df["sum_open_interest"].resample(
        "8h", label="right", closed="left"
    ).count()

    data = data.where(counts >= 90)

    oi = data["sum_open_interest"]

    data["oi_change_24h"] = oi.pct_change(3, fill_method=None)
    data["oi_change_3d"] = oi.pct_change(9, fill_method=None)

    rolling_mean = oi.rolling(90, min_periods=60).mean()
    rolling_std = oi.rolling(90, min_periods=60).std()

    data["oi_zscore"] = (oi - rolling_mean) / rolling_std

    data["top_trader_ratio"] = np.log(
        data["sum_toptrader_long_short_ratio"]
    )

    data["taker_ratio"] = np.log(
        data["sum_taker_long_short_vol_ratio"]
    )

    data = data.loc[:, [
        "sum_open_interest",
        "sum_open_interest_value",
        "oi_change_24h",
        "oi_change_3d",
        "oi_zscore",
        "top_trader_ratio",
        "taker_ratio",
        "count_long_short_ratio",
    ]]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index_label="timestamp")

    print(f"Saved {len(data):,} observations to {OUTPUT}")


if __name__ == "__main__":
    main()
