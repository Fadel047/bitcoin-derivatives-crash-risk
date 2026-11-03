
from pathlib import Path

import pandas as pd

DATA = Path("data/processed/BTCUSDT_features_8h.csv")
OUTPUT = Path("results/eda")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

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

target = "crash_72h"
baseline = df[target].mean()

# 1. Feature statistics by target
stats = df.groupby(target)[features].mean().T
stats.columns = ["no_crash", "crash"]
stats["difference"] = stats["crash"] - stats["no_crash"]
stats.to_csv(OUTPUT / "feature_comparison.csv")

# 2. Feature correlations
corr = df[features].corr()
corr.to_csv(OUTPUT / "feature_correlations.csv")

# 3. Crash probability by feature quintile
results = []

for feature in features:
    sample = df[[feature, target]].dropna().copy()

    sample["quintile"] = pd.qcut(
        sample[feature].rank(method="first"),
        q=5,
        labels=False,
    ) + 1

    grouped = sample.groupby("quintile")[target].agg(
        ["mean", "count", "sum"]
    )

    for quintile, row in grouped.iterrows():
        results.append({
            "feature": feature,
            "quintile": quintile,
            "observations": int(row["count"]),
            "crashes": int(row["sum"]),
            "crash_rate": row["mean"],
            "relative_risk": row["mean"] / baseline,
        })

quantiles = pd.DataFrame(results)
quantiles.to_csv(OUTPUT / "crash_by_quintile.csv", index=False)

print(f"\nBaseline crash probability: {baseline:.2%}")

print("\nFeature comparison:")
print(stats.round(4).to_string())

print("\nCrash probability by quintile:")
pivot = quantiles.pivot(
    index="feature",
    columns="quintile",
    values="crash_rate",
)
print((pivot * 100).round(2).to_string())

print(f"\nResults saved to {OUTPUT}")
