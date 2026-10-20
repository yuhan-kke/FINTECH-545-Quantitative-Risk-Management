"""Week 06: European option valuation - generalized Black-Scholes-Merton,
put-call parity, binomial tree, Monte Carlo, implied volatility.

T is in years (e.g. days / 255 as in the lecture). b is the cost of carry:
b = r (no dividends), r - q (continuous dividend q), 0 (futures),
r - r_foreign (FX).
"""
from __future__ import annotations

from math import comb

import numpy as np
from scipy import optimize, stats


def gbsm(call: bool, S, X, T, r, b, sigma) -> float:
    """Generalized Black-Scholes-Merton price."""
    d1 = (np.log(S / X) + (b + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return float(S * np.exp((b - r) * T) * stats.norm.cdf(d1)
                     - X * np.exp(-r * T) * stats.norm.cdf(d2))
    return float(X * np.exp(-r * T) * stats.norm.cdf(-d2)
                 - S * np.exp((b - r) * T) * stats.norm.cdf(-d1))


def put_call_parity_gap(C, P, S, X, T, r) -> float:
    """(C + X e^{-rT}) - (P + S). Zero under no arbitrage (no dividends)."""
    return float(C + X * np.exp(-r * T) - P - S)


def binomial_european(call: bool, S, X, T, r, b, sigma, N: int) -> float:
    """N-step CRR tree, European exercise (closed-form sum over terminal
    nodes): u = e^{sigma sqrt(dt)}, d = 1/u, p = (e^{b dt} - d)/(u - d)."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(b * dt) - d) / (u - d)
    i = np.arange(N + 1)
    ST = S * u ** i * d ** (N - i)
    pay = np.maximum(ST - X, 0) if call else np.maximum(X - ST, 0)
    probs = np.array([comb(N, k) for k in i], dtype=float) * p ** i * (1 - p) ** (N - i)
    return float(np.exp(-r * T) * probs @ pay)


def binomial_params(S, T, r, b, sigma, N: int) -> dict:
    """u, d, p and terminal prices/probabilities, for showing the tree."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt)); d = 1 / u
    p = (np.exp(b * dt) - d) / (u - d)
    i = np.arange(N + 1)[::-1]
    return {"u": u, "d": d, "p": p, "S_T": S * u ** i * d ** (N - i),
            "prob": np.array([comb(N, k) for k in i]) * p ** i * (1 - p) ** (N - i)}


def mc_european(call: bool, S, X, T, r, b, sigma, n_sims: int = 100_000, seed=None):
    """Monte Carlo price under the lognormal: ln S_T ~ N(ln S + (b - s^2/2)T, s^2 T).
    Returns (price, standard error)."""
    rng = np.random.default_rng(seed)
    ST = S * np.exp((b - sigma ** 2 / 2) * T + sigma * np.sqrt(T) * rng.standard_normal(n_sims))
    pay = np.maximum(ST - X, 0) if call else np.maximum(X - ST, 0)
    disc = np.exp(-r * T) * pay
    return float(disc.mean()), float(disc.std(ddof=1) / np.sqrt(n_sims))


def implied_vol(call: bool, price, S, X, T, r, b, lo=1e-6, hi=5.0) -> float:
    """Solve gbsm(sigma) = price with a root finder (price is monotone in sigma)."""
    return float(optimize.brentq(lambda s: gbsm(call, S, X, T, r, b, s) - price, lo, hi))
