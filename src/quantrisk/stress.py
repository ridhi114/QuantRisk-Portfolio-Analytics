"""Named market stress-window analysis."""

from __future__ import annotations

import pandas as pd

STRESS_PERIODS = {
    "COVID-19 crash": ("2020-02-19", "2020-03-23"),
    "2022 selloff": ("2022-01-03", "2022-10-12"),
}


def stress_summary(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end) in STRESS_PERIODS.items():
        sample = returns.loc[start:end]
        if sample.empty:
            continue
        for asset in sample.columns:
            r = sample[asset].dropna()
            if not r.empty:
                rows.append({"Period": period, "Asset": asset, "Total Return": (1 + r).prod() - 1,
                             "Worst Day": r.min(), "Daily Volatility": r.std(ddof=1)})
    return pd.DataFrame(rows)
