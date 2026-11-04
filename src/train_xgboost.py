from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from xgboost import XGBClassifier

DATA = Path("data/processed/BTCUSDT_features_8h.csv")
OUTPUT = Path("results/models")
OUTPUT.mkdir(parents=True, exist_ok=True)

TARGET = "crash_72h"
HORIZON = pd.Timedelta(hours=72)
TEST_WINDOW = pd.Timedelta(days=90)
INITIAL_TRAIN_RATIO = 0.7

price_features = [
    "return_8h",
    "return_24h",
    "return_72h",
    "realized_vol_72h",
]

positioning_features = [
    "oi_change_24h",
    "oi_change_3d",
    "oi_zscore",
    "top_trader_ratio",
    "taker_ratio",
    "count_long_short_ratio",
]

models = {
    "Price only": price_features,
    "Price + positioning": price_features + positioning_features,
}


def metrics(actual, probabilities):
    result = {
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "brier": brier_score_loss(actual, probabilities),
        "log_loss": log_loss(actual, probabilities, labels=[0, 1]),
    }
    if pd.Series(actual).nunique() == 2:
        result["roc_auc"] = roc_auc_score(actual, probabilities)
        result["pr_auc"] = average_precision_score(actual, probabilities)
    return result


def main(regularized=False):
    prefix = "xgboost_regularized" if regularized else "xgboost"
    params = (dict(n_estimators=120, max_depth=1, min_child_weight=5, reg_lambda=5)
              if regularized else dict(n_estimators=200, max_depth=2))
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    required = ["timestamp", TARGET] + price_features + positioning_features
    df = df.dropna(subset=required).reset_index(drop=True)

    if df.empty or df["timestamp"].duplicated().any():
        raise ValueError("Dataset is empty or contains duplicate timestamps.")

    start = df.loc[int(len(df) * INITIAL_TRAIN_RATIO), "timestamp"]
    end = df["timestamp"].max()
    predictions = []
    fold_results = []
    fold = 0

    while start <= end:
        train = df[df["timestamp"] + HORIZON < start]
        test = df[
            (df["timestamp"] >= start)
            & (df["timestamp"] < start + TEST_WINDOW)
        ]
        if train.empty or test.empty:
            break
        if train[TARGET].nunique() < 2:
            raise ValueError(f"Fold {fold}: training has only one class.")

        fold_predictions = pd.DataFrame({
            "timestamp": test["timestamp"].to_numpy(),
            "actual": test[TARGET].to_numpy(),
            "fold": fold,
        })

        for name, features in models.items():
            model = XGBClassifier(
                **params,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=42,
                n_jobs=1,
            )
            model.fit(train[features], train[TARGET])
            probabilities = model.predict_proba(test[features])[:, 1]
            fold_predictions[name] = probabilities
            fold_results.append({
                "fold": fold,
                "model": name,
                "test_start": start,
                "n_train": len(train),
                "n_test": len(test),
                "crash_rate": test[TARGET].mean(),
                **metrics(test[TARGET], probabilities),
            })

        predictions.append(fold_predictions)
        fold += 1
        start += TEST_WINDOW

    if not predictions:
        raise ValueError("No valid walk-forward folds.")

    predictions = pd.concat(predictions, ignore_index=True)
    fold_results = pd.DataFrame(fold_results)
    summary = pd.DataFrame([
        {"model": name, **metrics(predictions["actual"], predictions[name])}
        for name in models
    ]).set_index("model")

    predictions.to_csv(OUTPUT / f"{prefix}_predictions.csv", index=False)
    fold_results.to_csv(OUTPUT / f"{prefix}_folds.csv", index=False)
    summary.to_csv(OUTPUT / f"{prefix}_summary.csv")

    print(f"Walk-forward folds: {fold}")
    print(f"Out-of-sample observations: {len(predictions)}")
    print(f"Crash rate: {predictions['actual'].mean():.2%}")
    print("\nXGBoost walk-forward performance:")
    print(summary.round(4).to_string())
    print(f"\nResults saved to {OUTPUT}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--regularized", action="store_true")
    main(parser.parse_args().regularized)
