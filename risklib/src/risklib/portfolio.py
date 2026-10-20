"""Small portfolio helpers: holdings, shares, dollar P&L."""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_prices(prices: pd.DataFrame):
    """(date_column or None, list of asset columns) - the date column is the
    first non-numeric column."""
    non_num = prices.select_dtypes(exclude="number").columns
    date_col = non_num[0] if len(non_num) else None
    return date_col, [c for c in prices.columns if c != date_col]


def equal_weight_shares(prices: pd.DataFrame, total_value: float) -> pd.DataFrame:
    """Shares so each asset gets total_value / n at the LATEST price."""
    _, assets = split_prices(prices)
    last = prices[assets].iloc[-1]
    dollars = pd.Series(total_value / len(assets), index=assets)
    return pd.DataFrame({"price": last, "dollars": dollars, "shares": dollars / last})


def holdings_value(prices: pd.DataFrame, shares) -> pd.Series:
    """$ value of each holding at the LATEST price (shares: scalar, list or dict)."""
    _, assets = split_prices(prices)
    last = prices[assets].iloc[-1]
    if isinstance(shares, dict):
        shares = pd.Series(shares)
    return last * (shares if np.isscalar(shares) else pd.Series(shares, index=assets))
