
import pandas as pd

START, END = "2021-01-01", "2025-12-31"
PATH = f"data/raw/metrics/BTCUSDT_metrics_{START}_{END}.csv"

df = pd.read_csv(PATH, parse_dates=["create_time"])
df["create_time"] = pd.to_datetime(df["create_time"], utc=True)

expected = pd.date_range(
    START,
    pd.Timestamp(END) + pd.Timedelta(days=1),
    freq="5min",
    inclusive="left",
    tz="UTC"
)

actual = pd.DatetimeIndex(df["create_time"])
missing = expected.difference(actual)

print(f"Observations: {len(df):,}")
print(f"Duplicates: {actual.duplicated().sum()}")
print(f"Missing values: {df.isna().sum().sum()}")
print(f"Missing timestamps: {len(missing)}")

if len(missing):
    gaps = pd.Series(missing).dt.strftime("%Y-%m-%d").value_counts()
    print("\nMissing observations by day:")
    print(gaps.sort_index().to_string())



print("\nMissing values by column:")
print(df.isna().sum().sort_values(ascending=False))

print("\nMissing values by year:")
print(df.groupby(df["create_time"].dt.year).apply(
    lambda x: x.isna().sum()
).to_string())

print("\nRows with missing values:")
print(df.loc[df.isna().any(axis=1)].head(10).to_string(index=False))
