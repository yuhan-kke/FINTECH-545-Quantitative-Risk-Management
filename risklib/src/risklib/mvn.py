"""Multivariate normal: conditional distributions and simulation."""
from __future__ import annotations

import numpy as np

from .psd import chol_psd


def conditional_normal(mu, Sigma, given_idx, x_given):
    """Distribution of X_rest | X_given = x_given for X ~ N(mu, Sigma).

    mean = mu_r + S_rg S_gg^-1 (x_g - mu_g)
    cov  = S_rr - S_rg S_gg^-1 S_gr

    x_given may be a single vector (len(given_idx)) or a 2-D array of many
    observations (n x len(given_idx)), in which case the mean is n x k.
    Returns (cond_mean, cond_cov, beta) where beta = S_rg S_gg^-1.
    """
    mu = np.asarray(mu, dtype=float)
    S = np.asarray(Sigma, dtype=float)
    g = np.atleast_1d(given_idx)
    r = np.setdiff1d(np.arange(len(mu)), g)
    Sgg, Srg = S[np.ix_(g, g)], S[np.ix_(r, g)]
    beta = Srg @ np.linalg.inv(Sgg)
    xg = np.asarray(x_given, dtype=float)
    if xg.ndim == 1 and len(g) == 1 and xg.size > 1:
        xg = xg[:, None]                               # many obs, one given var
    mean = mu[r] + (xg - mu[g]) @ beta.T
    cov = S[np.ix_(r, r)] - beta @ S[np.ix_(g, r)]
    return mean, cov, beta


def band_coverage(y, cond_mean, cond_sd, z: float = 1.96) -> float:
    """Fraction of y inside cond_mean ± z * cond_sd."""
    y = np.asarray(y, dtype=float)
    m = np.asarray(cond_mean, dtype=float).reshape(y.shape)
    return float(np.mean(np.abs(y - m) <= z * cond_sd))


def simulate_normal(n_sims: int, Sigma, mu=None, seed=None) -> np.ndarray:
    """Draw n_sims rows from N(mu, Sigma) via PSD-safe Cholesky."""
    S = np.asarray(Sigma, dtype=float)
    L = chol_psd(S)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_sims, S.shape[0]))
    out = Z @ L.T
    return out if mu is None else out + np.asarray(mu, dtype=float)


def simulate_pca(n_sims: int, Sigma, pct_explained: float = 1.0, mu=None,
                 seed=None) -> np.ndarray:
    """Draw from N(mu, Sigma) using the top eigen-components that explain at
    least pct_explained of total variance."""
    S = np.asarray(Sigma, dtype=float)
    lam, V = np.linalg.eigh(S)
    lam, V = lam[::-1], V[:, ::-1]
    pos = lam > 1e-12
    lam, V = lam[pos], V[:, pos]
    k = int(np.searchsorted(np.cumsum(lam) / lam.sum(), pct_explained - 1e-12) + 1)
    B = V[:, :k] * np.sqrt(lam[:k])
    rng = np.random.default_rng(seed)
    out = rng.standard_normal((n_sims, k)) @ B.T
    return out if mu is None else out + np.asarray(mu, dtype=float)
