import numpy as np
import pandas as pd

from quantrisk.metrics import drawdown_series, historical_var_cvar, performance_summary, var_backtest


def test_drawdown_detects_peak_to_trough():
    returns = pd.Series([0.10, -0.20, 0.05])
    drawdown = drawdown_series(returns)
    assert np.isclose(drawdown.min(), -0.20)


def test_historical_cvar_is_at_least_var():
    returns = pd.Series(np.linspace(-0.10, 0.10, 101))
    var, cvar = historical_var_cvar(returns, 0.95)
    assert cvar >= var >= 0


def test_summary_has_expected_metrics():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"A": rng.normal(0.0005, 0.01, 504), "B": rng.normal(0.0002, 0.008, 504)})
    summary = performance_summary(frame)
    assert set(frame.columns) == set(summary.index)
    assert {"CAGR", "Sharpe", "Max Drawdown"}.issubset(summary.columns)


def test_var_backtest_has_no_lookahead():
    index = pd.date_range("2020-01-01", periods=40, freq="B")
    returns = pd.Series([0.01] * 39 + [-0.50], index=index)
    result, stats = var_backtest(returns, confidence=0.95, window=20)
    assert bool(result.iloc[-1]["Exception"])
    assert stats["exceptions"] == 1
