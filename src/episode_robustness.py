from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score

RESULTS = Path("results")
OUTPUT = RESULTS / "episodes"
MODEL = "Price + positioning"
HORIZON = pd.Timedelta(hours=72)
STEP = pd.Timedelta(hours=8)
THRESHOLD = 0.15


def read_series(path):
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    if df.index.has_duplicates or df.index.hasnans:
        raise ValueError(f"Invalid timestamps in {path}")
    return df


def ap_gap(rows):
    if rows["actual"].nunique() < 2:
        return float("nan")
    scores = [average_precision_score(rows["actual"], rows[col])
              for col in (MODEL, "Price only")]
    return scores[0] - scores[1]


def main():
    prices = read_series("data/raw/prices/BTCUSDT_spot_8h.csv")["close"]
    pred = read_series(RESULTS / "models/walk_forward_predictions.csv")
    episodes = pd.read_csv(OUTPUT / "logistic_episodes.csv", parse_dates=["start", "end"])
    if not prices.index.to_series().diff().dropna().eq(STEP).all():
        raise ValueError("Incomplete price history.")
    future = (prices.shift(-9) / prices - 1).reindex(pred.index)
    if future.isna().any() or not future.le(-0.05).eq(pred["actual"]).all():
        raise ValueError("Stored labels do not match 72h spot returns.")

    rows = []
    for episode in episodes.itertuples():
        start = episode.start
        path = prices.loc[start:start + HORIZON]
        breach = path[path <= 0.95 * prices.loc[start]].index[0]
        window = pd.date_range(breach - HORIZON, breach - STEP, freq=STEP)
        probs = pred[MODEL].reindex(window)
        alerts = probs[probs >= THRESHOLD]
        lead = (breach - alerts.index[0]) / pd.Timedelta(hours=1) if len(alerts) else None
        kept = pred[(pred.index < start) | (pred.index >= episode.end)]
        rows.append(dict(start=start, end=episode.end, breach=breach,
                         alert_history_complete=probs.notna().all(), lead_hours=lead,
                         removed_bars=len(pred) - len(kept), remaining_ap_gap=ap_gap(kept)))

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "breaches_and_sensitivity.csv", index=False)
    by_fold = {fold: ap_gap(group) for fold, group in pred.groupby("fold")}
    pd.Series(by_fold, name="ap_gap").rename_axis("fold").to_csv(OUTPUT / "ap_gap_by_fold.csv")

    fig, ax = plt.subplots(figsize=(11, 4))
    series = prices.loc[pred.index.min():pred.index.max()] / 1000
    ax.plot(series, color="#244466", linewidth=1.3, label="BTC close")
    alarms = pred.index[pred[MODEL] >= THRESHOLD]
    ax.scatter(alarms, series.loc[alarms], s=10, color="#bb432e", label="Alert (p ≥ 15%)")
    for i, e in enumerate(episodes.itertuples()):
        ax.axvspan(e.start, e.end, color="#e9ba6a", alpha=0.25, label="Episode" if i == 0 else None)
    ax.set(title="Bitcoin price, risk alerts and downside episodes", ylabel="BTC / USDT (thousands)")
    ax.legend(frameon=False, ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = Path("docs/alerts.png")
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(table.to_string(index=False))
    print("AP gain by fold:", by_fold)


if __name__ == "__main__":
    main()
