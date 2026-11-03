
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

SYMBOL = "BTCUSDT"
START, END = "2021-01-01", "2025-12-31"

BASE_URL = f"https://data.binance.vision/data/futures/um/daily/metrics/{SYMBOL}"
OUTPUT_DIR = Path("data/raw/metrics")
DAILY_DIR = OUTPUT_DIR / "daily"
DAILY_DIR.mkdir(parents=True, exist_ok=True)


def download(date):
    date = date.strftime("%Y-%m-%d")
    path = DAILY_DIR / f"{SYMBOL}_{date}.csv"

    if path.exists():
        return path

    url = f"{BASE_URL}/{SYMBOL}-metrics-{date}.zip"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with ZipFile(BytesIO(response.content)) as archive:
        csv_file = next(
            name for name in archive.namelist()
            if name.endswith(".csv")
        )
        with archive.open(csv_file) as file:
            df = pd.read_csv(file)

    df.to_csv(path, index=False)
    return path


def main():
    dates = pd.date_range(START, END)

    print(f"Downloading {SYMBOL} metrics from {START} to {END}...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(download, dates))

    data = pd.concat(
        (pd.read_csv(path) for path in paths),
        ignore_index=True
    )

    data["create_time"] = pd.to_datetime(data["create_time"], utc=True)

    data = (
        data.sort_values("create_time")
        .drop_duplicates(["create_time", "symbol"])
        .reset_index(drop=True)
    )

    output = OUTPUT_DIR / f"{SYMBOL}_metrics_{START}_{END}.csv"
    data.to_csv(output, index=False)

    print(f"Completed: {len(data):,} observations saved to {output}")


if __name__ == "__main__":
    main()
