
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

SYMBOL = "BTCUSDT"
START, END = "2023-01", "2025-12"

BASE_URL = f"https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/8h"
OUTPUT = Path("data/raw/prices/BTCUSDT_spot_8h.csv")

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore"
]


def download(month):
    url = f"{BASE_URL}/{SYMBOL}-8h-{month}.zip"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with ZipFile(BytesIO(response.content)) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".csv"))
        with archive.open(name) as file:
            return pd.read_csv(file, header=None, names=COLUMNS)


def main():
    months = pd.period_range(START, END, freq="M")

    print(f"Downloading {SYMBOL} spot prices from {START} to {END}...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        frames = list(executor.map(download, months.astype(str)))

    df = pd.concat(frames, ignore_index=True)

    open_time = pd.to_numeric(df["open_time"])
    timestamps_ms = open_time.where(open_time < 1e14, open_time / 1000)

    df["timestamp"] = pd.to_datetime(
        timestamps_ms, unit="ms", utc=True
    ) + pd.Timedelta(hours=8)

    df = (
        df[["timestamp", "open", "high", "low", "close", "volume"]]
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    expected = pd.date_range(
        "2023-01-01 08:00",
        "2026-01-01 00:00",
        freq="8h",
        tz="UTC"
    )

    if len(df) != len(expected) or not df["timestamp"].equals(
        pd.Series(expected, name="timestamp")
    ):
        raise ValueError("Missing or invalid price timestamps")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"Completed: {len(df):,} observations saved to {OUTPUT}")


if __name__ == "__main__":
    main()
