from __future__ import annotations

from datetime import date, timedelta
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from quantrisk.data import download_prices, simple_returns
from quantrisk.metrics import drawdown_series, performance_summary, var_backtest
from quantrisk.optimization import efficient_frontier, optimize_portfolios
from quantrisk.stress import stress_summary

st.set_page_config(page_title="QuantRisk", page_icon="📈", layout="wide")
st.title("QuantRisk")
st.caption("Portfolio performance, downside risk, VaR validation and optimization — in one reproducible workflow.")

with st.sidebar:
    st.header("Analysis settings")
    ticker_text = st.text_input("Assets", "AAPL, JPM, XOM, AMD, KO, TSLA")
    benchmark = st.text_input("Benchmark", "^GSPC").strip().upper()
    default_start = date.today() - timedelta(days=3652)
    start = st.date_input("Start date", default_start)
    end = st.date_input("End date", date.today())
    confidence = st.select_slider("VaR confidence", [0.90, 0.95, 0.99], 0.95, format_func=lambda x: f"{x:.0%}")
    risk_free = st.number_input("Risk-free rate", 0.0, 0.20, 0.02, 0.005, format="%.3f")
    window = st.selectbox("Rolling window", [60, 126, 252], index=2)

tickers = list(dict.fromkeys(x.strip().upper() for x in ticker_text.split(",") if x.strip()))
if benchmark in tickers:
    tickers.remove(benchmark)

@st.cache_data(ttl=3600, show_spinner=False)
def load(symbols, start_date, end_date):
    return download_prices(list(symbols), str(start_date), str(end_date))

try:
    with st.spinner("Downloading market history…"):
        prices = load(tuple(tickers + [benchmark]), start, end)
except Exception as exc:
    st.error(str(exc))
    st.stop()

returns = simple_returns(prices)
asset_returns = returns[tickers]
benchmark_returns = returns[benchmark].rename(benchmark)
summary = performance_summary(asset_returns, benchmark_returns, risk_free, confidence)

tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Risk & VaR", "Portfolio Lab", "Stress Tests"])

with tab1:
    growth = (1 + returns).cumprod()
    fig = px.line(growth, title="Growth of $1", labels={"value": "Portfolio value", "variable": "Asset"})
    st.plotly_chart(fig, use_container_width=True)
    display = summary.copy()
    pct_cols = ["CAGR", "Annual Return", "Annual Volatility", "Max Drawdown", f"Historical VaR ({confidence:.0%})", f"Historical CVaR ({confidence:.0%})", "Alpha"]
    st.dataframe(display.style.format({c: "{:.2%}" for c in pct_cols}).format({"Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Beta": "{:.2f}"}), use_container_width=True)
    st.download_button("Download metrics CSV", summary.to_csv().encode(), "quantrisk_metrics.csv", "text/csv")

with tab2:
    selected = st.selectbox("Asset for risk diagnostics", tickers)
    dd = pd.DataFrame({selected: drawdown_series(asset_returns[selected])})
    st.plotly_chart(px.area(dd, title=f"{selected} drawdown", labels={"value": "Drawdown"}), use_container_width=True)
    backtest, stats = var_backtest(asset_returns[selected], confidence, window)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=backtest.index, y=backtest["Return"], name="Daily return", line={"width": 1}))
    fig.add_trace(go.Scatter(x=backtest.index, y=-backtest["VaR"], name="VaR threshold", line={"color": "orange"}))
    breaches = backtest[backtest["Exception"]]
    fig.add_trace(go.Scatter(x=breaches.index, y=breaches["Return"], name="Exception", mode="markers", marker={"color": "red", "size": 7}))
    fig.update_layout(title=f"Rolling historical VaR backtest ({confidence:.0%}, {window} days)", yaxis_title="Daily return")
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Exceptions", f"{stats['exceptions']} / {stats['observations']}")
    c2.metric("Observed rate", f"{stats['exception_rate']:.2%}", delta=f"Expected {stats['expected_rate']:.2%}", delta_color="off")
    c3.metric("Kupiec p-value", f"{stats['kupiec_p_value']:.3f}")
    st.caption("A Kupiec p-value below 0.05 indicates that the observed exception frequency is inconsistent with the model's stated confidence level.")

with tab3:
    optimized = optimize_portfolios(asset_returns, risk_free)
    frontier = efficient_frontier(asset_returns)
    fig = px.line(frontier, x="Annual Volatility", y="Annual Return", title="Long-only efficient frontier")
    fig.add_scatter(x=optimized["Annual Volatility"], y=optimized["Annual Return"], mode="markers+text",
                    text=optimized.index, textposition="top center", marker={"size": 11})
    st.plotly_chart(fig, use_container_width=True)
    weight_cols = tickers
    st.subheader("Optimized allocations")
    st.dataframe(optimized.style.format({"Annual Return": "{:.2%}", "Annual Volatility": "{:.2%}", "Sharpe": "{:.2f}", **{c: "{:.1%}" for c in weight_cols}}), use_container_width=True)
    melted = optimized[weight_cols].reset_index().melt(id_vars="Strategy", var_name="Asset", value_name="Weight")
    st.plotly_chart(px.bar(melted, x="Strategy", y="Weight", color="Asset", title="Portfolio weights", barmode="stack"), use_container_width=True)

with tab4:
    stress = stress_summary(asset_returns)
    if stress.empty:
        st.info("The selected date range does not overlap a built-in stress period.")
    else:
        period = st.selectbox("Stress period", stress["Period"].unique())
        view = stress[stress["Period"] == period]
        st.plotly_chart(px.bar(view, x="Asset", y="Total Return", color="Asset", title=f"Total return during {period}"), use_container_width=True)
        st.dataframe(view.set_index("Asset").drop(columns="Period").style.format("{:.2%}"), use_container_width=True)

st.divider()
st.caption("Educational analytics only — historical performance does not guarantee future results and this is not investment advice.")
