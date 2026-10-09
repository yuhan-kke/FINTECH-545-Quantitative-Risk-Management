"""Price -> return conversion."""
from __future__ import annotations

import numpy as np
import pandas as pd


def return_calculate(prices: pd.DataFrame, method: str = "arithmetic",
                     date_column: str | None = None) -> pd.DataFrame:
    """Convert a price table to returns.

    method: 'arithmetic' (P_t/P_{t-1} - 1), 'log' (ln P_t - ln P_{t-1}),
            or 'diff' (P_t - P_{t-1}).
    date_column: column kept as-is (e.g. 'Date' or 'Day'); the first row is
                 dropped since it has no prior price.
    """
    df = prices.copy()
    keep = None
    if date_column is not None:
        keep = df[date_column]
        df = df.drop(columns=date_column)
    df = df.astype(float)
    m = method.lower()
    if m in ("arithmetic", "discrete", "simple"):
        r = df / df.shift(1) - 1
    elif m == "log":
        r = np.log(df).diff()
    elif m == "diff":
        r = df.diff()
    else:
        raise ValueError(f"unknown method {method!r}")
    if keep is not None:
        r.insert(0, date_column, keep)
    return r.iloc[1:].reset_index(drop=True)
