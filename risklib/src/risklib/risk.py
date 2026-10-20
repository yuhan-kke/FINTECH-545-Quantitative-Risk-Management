"""Value at Risk and Expected Shortfall.

Conventions (used everywhere in this module):
  * alpha is the tail probability (0.05 -> 95% VaR).
  * Inputs are P&L or returns where NEGATIVE = loss.
  * VaR and ES are returned as POSITIVE numbers for a loss.
  * Historical / simulated VaR uses numpy's default linear-interpolated
    quantile; ES is the mean of all outcomes at or below that quantile.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Empirical (historical simulation or Monte Carlo output)
# ---------------------------------------------------------------------------
def var_hist(x, alpha: float = 0.05) -> float:
    x = np.asarray(x, dtype=float)
    return float(-np.quantile(x[~np.isnan(x)], alpha))


def es_hist(x, alpha: float = 0.05) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    q = np.quantile(x, alpha)
    return float(-x[x <= q].mean())


def var_es_hist(x, alpha: float = 0.05):
    return var_hist(x, alpha), es_hist(x, alpha)


# ---------------------------------------------------------------------------
# Parametric
# ---------------------------------------------------------------------------
def var_normal(sigma: float, mu: float = 0.0, alpha: float = 0.05) -> float:
    return float(-(mu + sigma * stats.norm.ppf(alpha)))


def es_normal(sigma: float, mu: float = 0.0, alpha: float = 0.05) -> float:
    z = stats.norm.ppf(alpha)
    return float(-mu + sigma * stats.norm.pdf(z) / alpha)


def var_t(nu: float, mu: float = 0.0, sigma: float = 1.0, alpha: float = 0.05) -> float:
    return float(-stats.t.ppf(alpha, nu, mu, sigma))


def es_t(nu: float, mu: float = 0.0, sigma: float = 1.0, alpha: float = 0.05) -> float:
    """Closed form ES of a location-scale t:
    ES = -mu + sigma * (nu + q^2)/(nu - 1) * f_nu(q) / alpha, q = t_nu^-1(alpha)."""
    q = stats.t.ppf(alpha, nu)
    return float(-mu + sigma * (nu + q * q) / (nu - 1) * stats.t.pdf(q, nu) / alpha)


def var_dist(dist, alpha: float = 0.05, relative: bool = False) -> float:
    """VaR of any frozen scipy distribution of P&L / returns.
    relative=False -> distance from 0 (default, course convention);
    relative=True  -> distance from the distribution's mean."""
    v = float(-dist.ppf(alpha))
    return v + float(dist.mean()) if relative else v


def es_dist(dist, alpha: float = 0.05, relative: bool = False) -> float:
    """ES of any frozen scipy distribution by numerical integration of the
    quantile function: ES = -(1/alpha) * int_0^alpha F^-1(u) du."""
    from scipy import integrate
    val, _ = integrate.quad(dist.ppf, 1e-12, alpha, limit=200)
    v = float(-val / alpha)
    return v + float(dist.mean()) if relative else v


def delta_normal_var(weights, cov, alpha: float = 0.05, mu=None) -> float:
    """Portfolio VaR when P&L = w'r with r ~ N(mu, cov). weights in dollars."""
    w = np.asarray(weights, dtype=float)
    sd = float(np.sqrt(w @ np.asarray(cov, dtype=float) @ w))
    m = 0.0 if mu is None else float(w @ np.asarray(mu, dtype=float))
    return var_normal(sd, m, alpha)


def delta_normal_es(weights, cov, alpha: float = 0.05, mu=None) -> float:
    w = np.asarray(weights, dtype=float)
    sd = float(np.sqrt(w @ np.asarray(cov, dtype=float) @ w))
    m = 0.0 if mu is None else float(w @ np.asarray(mu, dtype=float))
    return es_normal(sd, m, alpha)


def portfolio_variance(weights, cov) -> float:
    """w' Sigma w."""
    w = np.asarray(weights, dtype=float)
    return float(w @ np.asarray(cov, dtype=float) @ w)


def var_es_table(pnls: dict, alphas=(0.05,)):
    """VaR and ES (historical) for several P&L vectors at several alphas.
    pnls: {name: array}. Returns a DataFrame, one row per name."""
    import pandas as pd
    rows = {}
    for name, x in pnls.items():
        row = {}
        for a in alphas:
            row[f"VaR {a:.0%}"] = var_hist(x, a)
            row[f"ES {a:.0%}"] = es_hist(x, a)
        rows[name] = row
    return pd.DataFrame(rows).T


def normal_var_es_table(pnls: dict, alphas=(0.05,)):
    """Normal VaR / ES from each P&L vector's sample mean and std (ddof=1)."""
    import pandas as pd
    rows = {}
    for name, x in pnls.items():
        x = np.asarray(x, dtype=float)
        mu, sd = x.mean(), x.std(ddof=1)
        row = {}
        for a in alphas:
            row[f"Normal VaR {a:.0%}"] = var_normal(sd, mu, a)
            row[f"Normal ES {a:.0%}"] = es_normal(sd, mu, a)
        rows[name] = row
    return pd.DataFrame(rows).T


def subadditivity(rho_a: float, rho_b: float, rho_ab: float) -> dict:
    """Is rho(A+B) <= rho(A) + rho(B)?"""
    return {"rho(A+B)": rho_ab, "rho(A)+rho(B)": rho_a + rho_b,
            "subadditive": bool(rho_ab <= rho_a + rho_b + 1e-9)}


def ew_var_standard_error(sigma: float, n_eff: float, alpha: float = 0.05,
                          position: float = 1.0) -> float:
    """SE of a normal VaR built from an EW volatility: z * sigma / sqrt(2 n_eff)."""
    return float(-stats.norm.ppf(alpha) * sigma / np.sqrt(2 * n_eff) * position)


def fitted_var_es_table(fits: dict, alphas=(0.05,), position: float = 1.0,
                        relative: bool = False):
    """VaR / ES for several fitted distributions. fits: {name: FitResult or
    frozen scipy dist}. position scales returns into dollars."""
    import pandas as pd
    rows = {}
    for name, f in fits.items():
        # FitResult carries .dist; a frozen scipy dist ALSO has .dist (its
        # generic class), so only unwrap objects that are not frozen dists.
        d = f if hasattr(f, "ppf") and hasattr(f, "mean") and not hasattr(f, "loglik") else f.dist
        row = {}
        for a in alphas:
            row[f"VaR {a:.0%}"] = var_dist(d, a, relative) * position
            row[f"ES {a:.0%}"] = es_dist(d, a, relative) * position
        rows[name] = row
    return pd.DataFrame(rows).T


def group_var_es(pnl, groups: dict, alphas=(0.05,)):
    """VaR / ES for groups of assets (e.g. trading books) and the total.

    pnl    : DataFrame of $ P&L, one column per asset (simulated or historical)
    groups : {"A": ["S1", "S2"], "B": [...]}   or a Series asset -> group
    Adds rows 'Total' and 'Sum of groups', plus a diversification row
    (Sum of groups - Total; positive = diversification benefit).
    """
    import pandas as pd
    if not isinstance(groups, dict):
        groups = {g: list(idx) for g, idx in pd.Series(groups).groupby(pd.Series(groups)).groups.items()}
    books = {g: pnl[list(cols)].sum(axis=1).values for g, cols in groups.items()}
    total = pnl.sum(axis=1).values
    tab = var_es_table({**books, "Total": total}, alphas)
    tab.loc["Sum of groups"] = tab.loc[list(groups)].sum()
    tab.loc["Diversification"] = tab.loc["Sum of groups"] - tab.loc["Total"]
    return tab


def es_contributions(pnl, alpha: float = 0.05, groups: dict | None = None):
    """Component ES: each asset's (or group's) average loss on the scenarios
    in the total's alpha tail. Contributions add up exactly to the total ES
    (Euler allocation for ES). Also returns each part's share of total ES."""
    import pandas as pd
    if groups is not None:
        pnl = pd.DataFrame({g: pnl[list(c)].sum(axis=1) for g, c in groups.items()})
    total = pnl.sum(axis=1).values
    v = np.quantile(total, alpha)
    tail = total <= v
    contrib = -pnl[tail].mean()
    out = pd.DataFrame({"ES contribution": contrib,
                        "share": contrib / contrib.sum()})
    out.loc["Total"] = [contrib.sum(), 1.0]
    return out


def weighted_hist_var_es(x, lam: float = 0.94, alpha: float = 0.05):
    """Weighted historical simulation (Week 04 Sec 6.2), computed exactly
    rather than by resampling: each past day gets EW weight (newest largest,
    x oldest first), sort outcomes, VaR = -(first outcome where cumulative
    weight >= alpha), ES = -(weighted mean of outcomes in that tail)."""
    from .covariance import ew_weights
    x = np.asarray(x, dtype=float)
    w = ew_weights(len(x), lam)
    o = np.argsort(x)
    xs, cw = x[o], np.cumsum(w[o])
    k = int(np.searchsorted(cw, alpha))
    var = -xs[k]
    es = -np.sum(w[o][:k + 1] * xs[:k + 1]) / cw[k]
    return float(var), float(es)


def sqrt_t_var(var_1day: float, T: int) -> float:
    """Square-root-of-time rule (Week 04 Sec 7): VaR_T = sqrt(T) VaR_1.
    Needs normal, iid, constant variance, zero mean."""
    return float(np.sqrt(T) * var_1day)
