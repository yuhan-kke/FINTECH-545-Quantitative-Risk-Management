"""Covariance / correlation estimation: Pearson, Spearman, missing data,
exponentially weighted, and conversions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _df(x) -> pd.DataFrame:
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame(np.asarray(x, dtype=float))


def cov2cor(S):
    """Covariance -> (correlation, std vector)."""
    S = np.asarray(S, dtype=float)
    sd = np.sqrt(np.diag(S))
    R = S / np.outer(sd, sd)
    np.fill_diagonal(R, 1.0)
    return R, sd


def cor2cov(R, sd):
    """Correlation + std vector -> covariance (D R D)."""
    sd = np.asarray(sd, dtype=float)
    return np.asarray(R, dtype=float) * np.outer(sd, sd)


def pearson(x) -> pd.DataFrame:
    return _df(x).corr(method="pearson")


def spearman(x) -> pd.DataFrame:
    return _df(x).corr(method="spearman")


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------
def joint_counts(x) -> pd.DataFrame:
    """Number of rows on which each pair of columns is jointly observed."""
    m = _df(x).notna().astype(int)
    return m.T @ m


def missing_cov(x, skip_missing: bool = True, corr: bool = False,
                ddof: int = 1) -> pd.DataFrame:
    """Covariance (or correlation) with missing values.

    skip_missing=True  -> complete cases only (rows with any NaN dropped).
                          Always PSD.
    skip_missing=False -> pairwise deletion: each entry uses the rows where
                          both series are observed. May NOT be PSD.
    """
    df = _df(x)
    if skip_missing:
        df = df.dropna()
        return df.corr() if corr else df.cov(ddof=ddof)
    cols = df.columns
    n = len(cols)
    out = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            pair = df[[cols[i], cols[j]]].dropna()
            if corr:
                v = 1.0 if i == j else np.corrcoef(pair.values.T)[0, 1]
            else:
                a, b = pair.iloc[:, 0].values, pair.iloc[:, -1].values
                v = np.sum((a - a.mean()) * (b - b.mean())) / (len(a) - ddof)
            out[i, j] = out[j, i] = v
    return pd.DataFrame(out, index=cols, columns=cols)


# ---------------------------------------------------------------------------
# Exponentially weighted
# ---------------------------------------------------------------------------
def ew_weights(n: int, lam: float) -> np.ndarray:
    """Normalised weights w_t ∝ (1-lam) lam^(t-1), oldest first in the
    returned array (so weights[-1] is the most recent and largest)."""
    w = (1 - lam) * lam ** np.arange(n)[::-1]
    return w / w.sum()


def ew_diagnostics(n: int, lam: float, recent: int | None = None) -> dict:
    """Effective sample size 1/sum(w^2), half life ln(0.5)/ln(lam), and
    (optionally) the share of total weight on the most recent `recent` obs."""
    w = ew_weights(n, lam)
    out = {"lambda": lam, "n_eff": float(1 / np.sum(w ** 2)),
           "half_life": float(np.log(0.5) / np.log(lam))}
    if recent is not None:
        out[f"weight_last_{recent}"] = float(w[-recent:].sum())
    return out


def ew_std(x, lam: float, demean: bool = False) -> float:
    """Exponentially weighted standard deviation of a single series
    (oldest first). demean=False assumes the mean was already removed."""
    x = np.asarray(x, dtype=float)
    w = ew_weights(len(x), lam)
    xc = x - (w @ x if demean else 0.0)
    return float(np.sqrt(w @ xc ** 2))


def ew_cov(x, lam: float = 0.97, corr: bool = False) -> pd.DataFrame:
    """Exponentially weighted covariance (RiskMetrics style). Rows must be
    in time order, oldest first. Weighted mean is removed first."""
    df = _df(x).dropna()
    X = df.values
    w = ew_weights(len(X), lam)
    Xc = X - w @ X
    S = (w[:, None] * Xc).T @ Xc
    if corr:
        S, _ = cov2cor(S)
    return pd.DataFrame(S, index=df.columns, columns=df.columns)


def ew_cov_mixed(x, lam_var: float = 0.97, lam_corr: float = 0.94) -> pd.DataFrame:
    """Covariance using EW variances with one lambda and EW correlation
    with another."""
    _, sd = cov2cor(ew_cov(x, lam_var).values)
    R = ew_cov(x, lam_corr, corr=True)
    return pd.DataFrame(cor2cov(R.values, sd), index=R.index, columns=R.columns)
