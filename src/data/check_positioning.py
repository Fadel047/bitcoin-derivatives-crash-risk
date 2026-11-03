import pandas as pd

df = pd.read_csv("data/processed/BTCUSDT_positioning_8h.csv")

print("Missing values:")
print(df.isna().sum())

print("\nFirst observations:")
print(df.head(3).to_string(index=False))