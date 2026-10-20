"""Single-factor (CAPM) regression, hedge ratio and hedged VaR/ES."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fitting import fit_regression_normal
from .risk import var_es_table, var_normal, es_normal


def capm(stock, market) -> dict:
    """OLS of stock returns on market returns.
    Returns alpha, beta, residual sd (ddof=2), R^2, and the variance split
    total = beta^2 var(m) + var(e)."""
    y, x = np.asarray(stock, float), np.asarray(market, float)
    beta, alpha = np.polyfit(x, y, 1)
    e = y - alpha - beta * x
    var_m, var_y = x.var(ddof=1), y.var(ddof=1)
    sys_var = beta ** 2 * var_m
    return {"alpha": alpha, "beta": beta, "resid_sd": e.std(ddof=2),
            "r2": 1 - e.var(ddof=1) / var_y, "total_var": var_y,
            "systematic_var": sys_var, "idiosyncratic_var": e.var(ddof=1),
            "systematic_share": sys_var / var_y, "residuals": e}


def hedge_ratio(stock, market) -> float:
    """Minimum-variance hedge ratio = beta = cov(s, m) / var(m).
    Short beta x (stock $) of the market to hedge."""
    s, m = np.asarray(stock, float), np.asarray(market, float)
    return float(np.cov(s, m, ddof=1)[0, 1] / m.var(ddof=1))


def hedged_var_es(stock, market, position: float, beta: float | None = None,
                  alphas=(0.05,)):
    """$ VaR / ES of: unhedged stock, the market hedge alone, and the hedged
    position (long `position` $ stock, short beta*position $ market).
    Historical and normal (sample mean / sd) versions."""
    s, m = np.asarray(stock, float), np.asarray(market, float)
    b = hedge_ratio(s, m) if beta is None else beta
    pnl = {"Unhedged": position * s,
           "Hedge leg (short mkt)": -b * position * m,
           "Hedged": position * (s - b * m)}
    hist = var_es_table(pnl, alphas).add_prefix("Hist ")
    rows = {}
    for k, x in pnl.items():
        rows[k] = {}
        for a in alphas:
            rows[k][f"Normal VaR {a:.0%}"] = var_normal(x.std(ddof=1), x.mean(), a)
            rows[k][f"Normal ES {a:.0%}"] = es_normal(x.std(ddof=1), x.mean(), a)
    out = pd.concat([hist, pd.DataFrame(rows).T], axis=1)
    out.attrs["beta"] = b
    return out
