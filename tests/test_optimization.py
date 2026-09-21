import numpy as np
import pandas as pd

from quantrisk.optimization import optimize_portfolios


def test_portfolio_weights_are_long_only_and_sum_to_one():
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(rng.normal(0.0005, 0.01, (500, 3)), columns=["A", "B", "C"])
    result = optimize_portfolios(returns)
    weights = result[["A", "B", "C"]]
    assert np.allclose(weights.sum(axis=1), 1)
    assert (weights >= -1e-8).all().all()
