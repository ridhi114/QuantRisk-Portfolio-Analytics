"""Long-only portfolio optimization and frontier calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def portfolio_statistics(weights: np.ndarray, mean_returns: np.ndarray, covariance: np.ndarray, risk_free_rate: float) -> tuple[float, float, float]:
    annual_return = float(weights @ mean_returns * TRADING_DAYS)
    annual_vol = float(np.sqrt(weights @ (covariance * TRADING_DAYS) @ weights))
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol else np.nan
    return annual_return, annual_vol, sharpe


def optimize_portfolios(returns: pd.DataFrame, risk_free_rate: float = 0.02) -> pd.DataFrame:
    """Return equal-weight, minimum-volatility, and maximum-Sharpe portfolios."""
    clean = returns.dropna()
    if clean.shape[1] < 2 or len(clean) < 30:
        raise ValueError("Optimization needs at least two assets and 30 shared observations.")
    mean, cov, n = clean.mean().values, clean.cov().values, clean.shape[1]
    initial = np.repeat(1 / n, n)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = tuple((0.0, 1.0) for _ in range(n))

    min_vol = minimize(lambda w: portfolio_statistics(w, mean, cov, risk_free_rate)[1], initial,
                       method="SLSQP", bounds=bounds, constraints=constraints)
    max_sharpe = minimize(lambda w: -portfolio_statistics(w, mean, cov, risk_free_rate)[2], initial,
                          method="SLSQP", bounds=bounds, constraints=constraints)
    if not min_vol.success or not max_sharpe.success:
        raise RuntimeError("Portfolio optimization did not converge.")

    portfolios = {"Equal Weight": initial, "Minimum Volatility": min_vol.x, "Maximum Sharpe": max_sharpe.x}
    rows = []
    for strategy, weights in portfolios.items():
        ret, vol, sharpe = portfolio_statistics(weights, mean, cov, risk_free_rate)
        row = {"Strategy": strategy, "Annual Return": ret, "Annual Volatility": vol, "Sharpe": sharpe}
        row.update(dict(zip(clean.columns, weights)))
        rows.append(row)
    return pd.DataFrame(rows).set_index("Strategy")


def efficient_frontier(returns: pd.DataFrame, points: int = 30) -> pd.DataFrame:
    clean = returns.dropna()
    mean, cov, n = clean.mean().values * TRADING_DAYS, clean.cov().values * TRADING_DAYS, clean.shape[1]
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints_sum = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    targets = np.linspace(mean.min(), mean.max(), points)
    rows = []
    for target in targets:
        constraints = (constraints_sum, {"type": "eq", "fun": lambda w, t=target: w @ mean - t})
        result = minimize(lambda w: np.sqrt(w @ cov @ w), np.repeat(1 / n, n), method="SLSQP",
                          bounds=bounds, constraints=constraints)
        if result.success:
            rows.append({"Annual Return": target, "Annual Volatility": result.fun})
    return pd.DataFrame(rows)
