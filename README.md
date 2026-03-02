# Stock Trading ML & Power BI Dashboard

An end-to-end stock analysis pipeline that fetches historical data, computes technical indicators, trains ensemble machine learning models to generate Buy/Hold/Sell signals, backtests the strategy, and exports a comprehensive dataset for Power BI visualization.

## Overview

The pipeline analyzes three instruments — **META** (tech stock), **USO** (oil ETF), and **URTH** (world ETF) — over 5 years of historical data. ML models predict daily trading signals, a backtesting engine simulates portfolio performance, and the results are exported to CSV for Power BI dashboards.

## Tech Stack

- **Python** — pandas, NumPy, yfinance, ta
- **Machine Learning** — scikit-learn (Random Forest, Decision Tree, KNN, AdaBoost), XGBoost, SMOTE
- **Visualization** — Power BI (.pbix), Jupyter Notebook
- **Data Source** — Yahoo Finance via yfinance

## Project Structure

```
├── stock_ml_pipeline.py          # Main end-to-end pipeline
├── DataFetching.ipynb            # Exploratory data fetching notebook
├── Financial Trading Bot - Document.pdf
├── Financial Trading Bot - Presentation 1.pdf
├── Financial Trading Bot - Presentation 2.pptx
├── Trading_Bot_Implementation_Plan.docx
└── PowerBI sample visual.png
```

## Pipeline Steps

1. **Data Collection** — Fetches 5 years of OHLCV data from Yahoo Finance
2. **Technical Indicators** — RSI, MACD, Bollinger Bands, Ichimoku Cloud, Stochastic, SMA/EMA, ATR
3. **Target Variable** — Buy if daily change < −1%, Sell if > +1%, else Hold
4. **ML Training** — Trains 5 models + ensemble VotingClassifier with SMOTE for class balancing
5. **Backtesting** — Simulates portfolio with Sharpe Ratio and Max Drawdown metrics
6. **Power BI Export** — Outputs `stock_analysis_powerbi_output.csv` with all features and metrics

## Running the Pipeline

```bash
pip install pandas numpy yfinance ta scikit-learn xgboost imbalanced-learn
python stock_ml_pipeline.py
```

Output CSV can be imported directly into Power BI for dashboard creation.
