"""Sample moments and normal tail diagnostics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Moments:
    n: int
    mean: float
    variance: float
    std: float
    skew: float
    excess_kurtosis: float

    def to_series(self) -> pd.Series:
        return pd.Series(self.__dict__)


def first4_moments(x, unbiased: bool = True) -> Moments:
    """Mean, variance, skewness and excess kurtosis of a sample.

    unbiased=True uses ddof=1 for the variance and the bias-corrected
    skew/kurtosis estimators (scipy bias=False). NaNs are dropped.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    ddof = 1 if unbiased else 0
    var = np.var(x, ddof=ddof)
    return Moments(
        n=len(x),
        mean=float(np.mean(x)),
        variance=float(var),
        std=float(np.sqrt(var)),
        skew=float(stats.skew(x, bias=not unbiased)),
        excess_kurtosis=float(stats.kurtosis(x, bias=not unbiased)),
    )


def tail_exceedances(x, probs=(0.001, 0.005, 0.01, 0.02, 0.05, 0.95, 0.99, 0.995, 0.999),
                     dist=None) -> pd.DataFrame:
    """Compare observed vs expected tail counts against a fitted distribution.

    For p < 0.5 counts observations below the p-quantile; for p >= 0.5 counts
    observations above it. `dist` is any frozen scipy distribution; defaults to
    a normal with the sample mean and (ddof=1) std.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if dist is None:
        dist = stats.norm(np.mean(x), np.std(x, ddof=1))
    rows = []
    for p in probs:
        q = dist.ppf(p)
        if p < 0.5:
            obs, exp, side = int(np.sum(x < q)), p * n, "below"
        else:
            obs, exp, side = int(np.sum(x > q)), (1 - p) * n, "above"
        rows.append({"p": p, "quantile": q, "side": side,
                     "observed": obs, "expected": exp, "ratio": obs / exp})
    return pd.DataFrame(rows)


def mixture_excess_kurtosis(sds, weights) -> float:
    """Excess kurtosis of a mixture of zero-mean normals:
    3 * sum(p s^4) / (sum(p s^2))^2 - 3."""
    s = np.asarray(sds, dtype=float)
    p = np.asarray(weights, dtype=float)
    p = p / p.sum()
    return float(3 * np.sum(p * s ** 4) / np.sum(p * s ** 2) ** 2 - 3)


def moments_table(df: pd.DataFrame) -> pd.DataFrame:
    """first4_moments for every column, one column per series."""
    return pd.DataFrame({c: first4_moments(df[c]).to_series() for c in df.columns})


def drop_most_extreme(x):
    """Return (index, value, moments without that observation) for the
    observation furthest from the mean."""
    x = np.asarray(x, dtype=float)
    i = int(np.argmax(np.abs(x - x.mean())))
    return i, float(x[i]), first4_moments(np.delete(x, i))
