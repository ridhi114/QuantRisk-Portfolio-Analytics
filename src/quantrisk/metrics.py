"""Performance, downside-risk, benchmark, and VaR validation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

TRADING_DAYS = 252


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns.dropna()).cumprod()
    return wealth / wealth.cummax() - 1


def historical_var_cvar(returns: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    clean = returns.dropna()
    cutoff = clean.quantile(1 - confidence)
    tail = clean[clean <= cutoff]
    return float(-cutoff), float(-tail.mean())


def gaussian_var_cvar(returns: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    clean = returns.dropna()
    mu, sigma = clean.mean(), clean.std(ddof=1)
    z = norm.ppf(1 - confidence)
    var = -(mu + sigma * z)
    cvar = -(mu - sigma * norm.pdf(z) / (1 - confidence))
    return float(var), float(cvar)


def benchmark_stats(asset: pd.Series, benchmark: pd.Series, risk_free_rate: float = 0.02) -> tuple[float, float]:
    joined = pd.concat([asset, benchmark], axis=1).dropna()
    if len(joined) < 2 or joined.iloc[:, 1].var() == 0:
        return np.nan, np.nan
    beta = joined.cov().iloc[0, 1] / joined.iloc[:, 1].var()
    ann_asset = joined.iloc[:, 0].mean() * TRADING_DAYS
    ann_bench = joined.iloc[:, 1].mean() * TRADING_DAYS
    alpha = ann_asset - (risk_free_rate + beta * (ann_bench - risk_free_rate))
    return float(alpha), float(beta)


def performance_summary(
    returns: pd.DataFrame,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.02,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Calculate annualized performance and risk statistics for each asset."""
    rows: dict[str, dict[str, float]] = {}
    for name in returns.columns:
        r = returns[name].dropna()
        if len(r) < 2:
            continue
        years = len(r) / TRADING_DAYS
        cumulative = (1 + r).prod()
        cagr = cumulative ** (1 / years) - 1
        ann_return = r.mean() * TRADING_DAYS
        ann_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
        downside = r[r < 0].std(ddof=1) * np.sqrt(TRADING_DAYS)
        hvar, hcvar = historical_var_cvar(r, confidence)
        dd = drawdown_series(r)
        alpha, beta = (benchmark_stats(r, benchmark, risk_free_rate)
                       if benchmark is not None and name != benchmark.name else (np.nan, np.nan))
        rows[name] = {
            "CAGR": cagr,
            "Annual Return": ann_return,
            "Annual Volatility": ann_vol,
            "Sharpe": (ann_return - risk_free_rate) / ann_vol if ann_vol else np.nan,
            "Sortino": (ann_return - risk_free_rate) / downside if downside else np.nan,
            "Max Drawdown": dd.min(),
            f"Historical VaR ({confidence:.0%})": hvar,
            f"Historical CVaR ({confidence:.0%})": hcvar,
            "Alpha": alpha,
            "Beta": beta,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _kupiec_test(exceptions: int, observations: int, expected_rate: float) -> tuple[float, float]:
    if observations == 0:
        return np.nan, np.nan
    observed = exceptions / observations
    if observed in (0, 1):
        observed = np.clip(observed, 1e-12, 1 - 1e-12)
    ll_null = (observations - exceptions) * np.log(1 - expected_rate) + exceptions * np.log(expected_rate)
    ll_alt = (observations - exceptions) * np.log(1 - observed) + exceptions * np.log(observed)
    statistic = -2 * (ll_null - ll_alt)
    return float(statistic), float(chi2.sf(statistic, 1))


def var_backtest(returns: pd.Series, confidence: float = 0.95, window: int = 252) -> tuple[pd.DataFrame, dict[str, float]]:
    """Backtest rolling historical VaR; today's threshold uses only prior returns."""
    clean = returns.dropna().sort_index()
    forecast = -clean.shift(1).rolling(window).quantile(1 - confidence)
    result = pd.DataFrame({"Return": clean, "VaR": forecast}).dropna()
    result["Exception"] = result["Return"] < -result["VaR"]
    n, x = len(result), int(result["Exception"].sum())
    lr, p_value = _kupiec_test(x, n, 1 - confidence)
    stats = {
        "observations": n,
        "exceptions": x,
        "exception_rate": x / n if n else np.nan,
        "expected_rate": 1 - confidence,
        "kupiec_lr": lr,
        "kupiec_p_value": p_value,
    }
    return result, stats
