# Bitcoin Derivatives Positioning and Crash Risk Prediction

Does BTC perpetual-futures positioning help predict a decline in Bitcoin and improve portfolio allocation after trading costs?

The study uses Binance BTCUSDT spot closes and USD-M perpetual positioning on 8-hour bars from 2023 through 2025. A positive label means the close **72 hours later** is at least 5% below the current close. Price features are compared with price plus open interest, long/short ratios and taker activity. Logistic regression and XGBoost use expanding training windows, a 72-hour purge and chronological out-of-sample tests.

[Findings and alert chart](docs/findings.md)

## Reproduce

Run these commands from the repository root in a Python environment. The download steps require access to Binance's public archive. If the processed feature and research datasets and spot CSV are already available, start the analysis at the walk-forward step.

```powershell
python -m pip install -r requirements.txt
python src/data/download_metrics.py
python src/data/download_prices.py
python src/features/build_positioning.py
python src/features/build_dataset.py
python src/build_features.py
python src/walk_forward.py
python src/train_xgboost.py
python src/train_xgboost.py --regularized
python src/backtest.py
python src/analyze_backtest.py
python src/analyze_crash_episodes.py
python src/episode_robustness.py
```

## Reference

Tran et al. (2026), [Systemic Risk from Financial Leverage in Digital Asset Markets: Evidence From Perpetual Futures](https://doi.org/10.51505/IJEBMR.2026.10315). This project studies BTC alone, extends the prediction horizon to 72 hours and tests an allocation rule after transaction costs.
