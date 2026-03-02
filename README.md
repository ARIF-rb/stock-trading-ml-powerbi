# Stock Trading ML & Power BI Dashboard

An end-to-end stock analysis pipeline that fetches historical price data, computes technical indicators, trains ensemble machine learning models to generate Buy/Hold/Sell signals, backtests the strategy, and exports results for Power BI visualization.

## Use Case

Quantitative traders and analysts who want to automate signal generation on equities/ETFs using technical analysis + ML, then visualize portfolio performance metrics in Power BI without writing a separate data pipeline.

## Features

- Fetches 5 years of OHLCV data for **META**, **USO**, and **URTH** via Yahoo Finance
- Computes 10+ technical indicators: RSI, MACD, Bollinger Bands, Ichimoku Cloud, Stochastic, SMA/EMA, ATR
- Trains 5 ML classifiers + ensemble VotingClassifier with SMOTE class balancing
- Backtests strategy with Sharpe Ratio and Max Drawdown reporting
- Exports a single flat CSV ready for Power BI import

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.8+ |
| Data Fetching | yfinance |
| Feature Engineering | pandas, NumPy, ta |
| Machine Learning | scikit-learn (Random Forest, Decision Tree, KNN, AdaBoost), XGBoost, imbalanced-learn (SMOTE) |
| Backtesting | Custom portfolio simulator (built-in) |
| Visualization | Power BI Desktop (.pbix), Jupyter Notebook |

## Prerequisites

- Python 3.8 or higher
- ~4 GB RAM (XGBoost + SMOTE on 5 years of data)
- Internet connection (yfinance pulls live data at runtime)
- [Power BI Desktop](https://powerbi.microsoft.com/desktop/) (free) to open the dashboard

## Installation

```bash
pip install pandas numpy yfinance ta scikit-learn xgboost imbalanced-learn
```

## Running

```bash
python stock_ml_pipeline.py
```

The script runs end-to-end: fetch → feature engineering → train → backtest → export. No arguments required.

## Pipeline Steps

| Step | Description |
|---|---|
| 1. Data Collection | Downloads 5-year daily OHLCV from Yahoo Finance |
| 2. Technical Indicators | RSI, MACD, Bollinger Bands, Ichimoku, Stochastic, SMA/EMA, ATR |
| 3. Target Labeling | Buy if Δ < −1%, Sell if Δ > +1%, Hold otherwise |
| 4. SMOTE Balancing | Oversamples minority classes before training |
| 5. Model Training | Trains 5 individual models + ensemble VotingClassifier |
| 6. Backtesting | Simulates portfolio; computes Sharpe Ratio & Max Drawdown |
| 7. Export | Writes `stock_analysis_powerbi_output.csv` |

## Output & Results

| File | Description |
|---|---|
| `stock_analysis_powerbi_output.csv` | All OHLCV features, indicators, ML signals, and backtest metrics |
| Console output | Per-model accuracy, classification report, Sharpe Ratio, Max Drawdown |

Import the CSV into Power BI and use the included `PowerBI sample visual.png` as a reference for dashboard layout.

## Project Structure

```
├── stock_ml_pipeline.py                    # Main pipeline (run this)
├── DataFetching.ipynb                      # Exploratory notebook
├── Financial Trading Bot - Document.pdf   # Project report
├── Financial Trading Bot - Presentation 1.pdf
├── Financial Trading Bot - Presentation 2.pptx
├── Trading_Bot_Implementation_Plan.docx
└── PowerBI sample visual.png              # Dashboard screenshot reference
```

## Notes & Limitations

- Signal thresholds (±1%) are hardcoded in `stock_ml_pipeline.py` — adjust for different risk tolerances
- Models are retrained from scratch on every run; no model persistence (no `.pkl` files saved)
- Past performance of backtested signals does not guarantee future results
- yfinance data availability depends on Yahoo Finance API uptime
