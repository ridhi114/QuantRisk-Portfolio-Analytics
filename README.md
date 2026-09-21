# QuantRisk — Portfolio & Market Risk Analytics

An end-to-end financial analytics project that turns historical adjusted prices into portfolio, benchmark, and downside-risk insights. It combines a reusable Python analytics package with an interactive Streamlit dashboard.

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-Open_App-FF4B4B?logo=streamlit&logoColor=white)](https://quantrisk-ridhi-jain.streamlit.app/)

**Live demo:** [Launch QuantRisk](https://quantrisk-ridhi-jain.streamlit.app/)

## Why this project is different

QuantRisk goes beyond plotting stock returns. It validates risk estimates through rolling Value-at-Risk backtesting, compares assets with a market benchmark, measures drawdown recovery, and builds minimum-volatility and maximum-Sharpe portfolios.

## Features

- Live historical market data through `yfinance`
- CAGR, annualized return/volatility, Sharpe and Sortino ratios
- Maximum drawdown and recovery duration
- Alpha, beta, correlation and diversification analysis
- Historical and Gaussian VaR/CVaR
- Rolling VaR backtesting with exception rate and Kupiec coverage test
- Long-only efficient frontier optimization
- Stress-window analysis for COVID-19 and the 2022 selloff
- CSV export of summary metrics
- Offline unit tests for the core financial calculations

## Architecture

```text
yfinance prices → return engine → performance/risk metrics
                              ├→ benchmark analysis
                              ├→ VaR backtest
                              ├→ portfolio optimizer
                              └→ Streamlit visualizations
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The default universe is AAPL, JPM, XOM, AMD, KO and TSLA, benchmarked against the S&P 500 (`^GSPC`). Change the tickers, dates, confidence level and portfolio assumptions from the sidebar.

## Methodology

Prices are converted to simple daily returns. Annualized figures assume 252 trading days. Historical VaR uses the empirical loss quantile; Gaussian VaR assumes normally distributed returns. CVaR is the average loss beyond VaR. VaR backtesting uses a one-day forecast estimated only from the preceding rolling window, avoiding look-ahead bias. The Kupiec likelihood-ratio test checks whether observed breaches are consistent with the expected exception rate.

Portfolio optimization uses SLSQP under long-only, fully-invested constraints. The maximum-Sharpe portfolio maximizes excess return per unit of volatility; the minimum-volatility portfolio minimizes annualized portfolio variance.

## Suggested recruiter demo

1. Compare cumulative growth and drawdowns across the default stocks.
2. Show how AMD's historical growth differs from TSLA's downside-risk profile.
3. Open the VaR backtest and explain why model validation matters.
4. Compare equal-weight, minimum-volatility and maximum-Sharpe allocations.
5. Change the lookback period or confidence level to demonstrate robustness.

## Responsible interpretation

Historical results are not forecasts. VaR is a threshold rather than a maximum possible loss, Gaussian VaR can understate fat-tail risk, and optimized weights are sensitive to estimation error. The dashboard is educational and is not investment advice.

## Project structure

```text
quantrisk/
├── app.py
├── src/quantrisk/
│   ├── data.py
│   ├── metrics.py
│   ├── optimization.py
│   └── stress.py
├── tests/
├── requirements.txt
└── pyproject.toml
```

