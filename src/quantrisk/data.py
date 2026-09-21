"""Market-data access and return transformations."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted closing prices and reject incomplete requests clearly."""
    symbols = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if not symbols:
        raise ValueError("Enter at least one ticker.")
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("No market data was returned. Check the tickers and date range.")
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if len(symbols) == 1:
        prices.columns = symbols
    prices = prices.sort_index().dropna(axis=1, how="all").ffill().dropna(how="all")
    missing = sorted(set(symbols) - set(prices.columns))
    if missing:
        raise ValueError(f"No valid price history for: {', '.join(missing)}")
    return prices[symbols]


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert prices to simple returns without silently filling missing observations."""
    return prices.pct_change(fill_method=None).dropna(how="all")
