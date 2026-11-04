from pathlib import Path

import pandas as pd

INPUT = Path("results/models/walk_forward_predictions.csv")
OUTPUT = Path("results/episodes")
MODEL = "Price + positioning"
STEP = pd.Timedelta(hours=8)
HORIZON = pd.Timedelta(hours=72)
THRESHOLD = 0.15


def analyze(df):
    df = df.sort_index()
    if df.empty or df.index.has_duplicates or df.index.hasnans:
        raise ValueError("Expected unique, valid forecast timestamps.")
    if not df["actual"].isin([0, 1]).all() or not df[MODEL].between(0, 1).all():
        raise ValueError("Invalid labels or forecast probabilities.")

    episodes = []
    false_alerts = late_alerts = early_alerts = censored_alerts = 0
    segments = df.index.to_series().diff().ne(STEP).cumsum()
    for _, segment in df.groupby(segments):
        positive = segment.index[segment["actual"].eq(1)]
        alarms = segment.index[segment[MODEL].ge(THRESHOLD)]
        early = pd.Series(False, index=alarms)
        active = early.copy()

        # Consecutive positive forecast windows belong together while they overlap.
        groups = positive.to_series().diff().ge(HORIZON).cumsum()
        for _, group in positive.to_series().groupby(groups):
            start = group.iloc[0]
            end = group.iloc[-1] + HORIZON
            prior = alarms[(alarms >= start - HORIZON) & (alarms < start)]
            early.loc[prior] = True
            active.loc[(alarms >= start) & (alarms < end)] = True
            complete = start - HORIZON >= segment.index[0]
            episodes.append({
                "start": start, "end": end, "positive_windows": len(group),
                "full_lookback": complete,
                "first_early_alert": prior[0] if len(prior) else pd.NaT,
                "lead_hours": (start - prior[0]) / pd.Timedelta(hours=1)
                if len(prior) else None,
            })

        # Early takes precedence if an alert also falls inside a prior episode.
        unknown = ((alarms < segment.index[0] + HORIZON)
                   | (alarms + HORIZON > segment.index[-1]))
        early_alerts += int(early.sum())
        late_alerts += int((active & ~early).sum())
        false_alerts += int((~early & ~active & ~unknown).sum())
        censored_alerts += int((~early & ~active & unknown).sum())

    events = pd.DataFrame(episodes, columns=[
        "start", "end", "positive_windows", "full_lookback",
        "first_early_alert", "lead_hours",
    ])
    eligible = events.loc[events["full_lookback"].eq(True)]
    detected = eligible["first_early_alert"].notna().sum()
    print(f"Episodes: {len(events)} | full 72h lookback: {len(eligible)}"
          f" | anticipated: {detected}/{len(eligible)}")
    if detected:
        print(f"Median first-alert lead: {eligible['lead_hours'].median():.0f}h")
    print(f"Alert bars: early={early_alerts}, during episode={late_alerts},"
          f" false={false_alerts}, boundary unknown={censored_alerts}")
    return events


def main():
    df = pd.read_csv(INPUT, usecols=["timestamp", "actual", MODEL])
    df.index = pd.to_datetime(df.pop("timestamp"), utc=True, errors="raise")
    events = analyze(df)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUTPUT / "logistic_episodes.csv", index=False)


if __name__ == "__main__":
    main()
