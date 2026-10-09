"""AR / MA / ARMA fitting and AICc model selection."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .fitting import aicc as _aicc


def fit_arma(x, p: int = 0, q: int = 0, trend: str = "c"):
    """Exact-MLE ARMA(p, q) via statsmodels ARIMA. All orders use the same
    likelihood and the same n, so AICc values are directly comparable."""
    from statsmodels.tsa.arima.model import ARIMA
    x = pd.Series(np.asarray(x, dtype=float)).dropna().reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(x, order=(p, 0, q), trend=trend).fit()


def model_aicc(res) -> float:
    """AICc for a statsmodels result. k = every estimated parameter (for
    ARIMA results this already includes sigma2)."""
    return _aicc(res.llf, len(res.params), int(res.nobs))


def select_arma(x, orders=None, trend: str = "c") -> pd.DataFrame:
    """Fit a set of (p, q) orders and rank by AICc.

    Default orders: AR(1..3) and MA(1..3).
    Returns a DataFrame sorted best-first with the fitted results in `result`.
    """
    if orders is None:
        orders = [(p, 0) for p in (1, 2, 3)] + [(0, q) for q in (1, 2, 3)]
    rows = []
    for p, q in orders:
        r = fit_arma(x, p, q, trend)
        name = f"AR({p})" if q == 0 else f"MA({q})" if p == 0 else f"ARMA({p},{q})"
        rows.append({"model": name, "p": p, "q": q, "k": len(r.params),
                     "loglik": r.llf, "AIC": r.aic, "AICc": model_aicc(r),
                     "result": r})
    return pd.DataFrame(rows).sort_values("AICc").reset_index(drop=True)


def acf(x, nlags: int = 20) -> np.ndarray:
    from statsmodels.tsa.stattools import acf as _acf
    return _acf(np.asarray(x, dtype=float), nlags=nlags)


def pacf(x, nlags: int = 20) -> np.ndarray:
    from statsmodels.tsa.stattools import pacf as _pacf
    return _pacf(np.asarray(x, dtype=float), nlags=nlags, method="ywm")


def variance_breakpoint(x, min_size: int = 10) -> dict:
    """Single change in variance (zero-mean series), located by maximising the
    two-regime normal likelihood  -n1/2 ln s1^2 - n2/2 ln s2^2  over the split.
    Returns the index of the first obs of the new regime, the number of obs
    after the break, both regime std devs, and the likelihood-ratio statistic
    against a single variance."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    c = np.cumsum(x ** 2)
    best, k_best = -np.inf, None
    for k in range(min_size, n - min_size + 1):
        s1, s2 = c[k - 1] / k, (c[-1] - c[k - 1]) / (n - k)
        ll = -0.5 * (k * np.log(s1) + (n - k) * np.log(s2))
        if ll > best:
            best, k_best = ll, k
    ll0 = -0.5 * n * np.log(c[-1] / n)
    return {"break_index": k_best, "n_after": n - k_best,
            "sd_before": float(np.sqrt(np.mean(x[:k_best] ** 2))),
            "sd_after": float(np.sqrt(np.mean(x[k_best:] ** 2))),
            "lr_stat": float(2 * (best - ll0))}
