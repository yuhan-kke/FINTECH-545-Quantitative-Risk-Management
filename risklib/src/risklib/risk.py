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


def var_dist(dist, alpha: float = 0.05) -> float:
    """VaR of any frozen scipy distribution of P&L / returns."""
    return float(-dist.ppf(alpha))


def es_dist(dist, alpha: float = 0.05) -> float:
    """ES of any frozen scipy distribution by numerical integration of the
    quantile function: ES = -(1/alpha) * int_0^alpha F^-1(u) du."""
    from scipy import integrate
    val, _ = integrate.quad(dist.ppf, 1e-12, alpha, limit=200)
    return float(-val / alpha)


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
