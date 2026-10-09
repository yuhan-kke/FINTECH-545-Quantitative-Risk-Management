"""Maximum-likelihood fitting: univariate distributions and linear regression
with Normal or Student-t errors. Every fit returns a FitResult carrying the
log-likelihood, AIC and AICc so models can be compared directly."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats


# ---------------------------------------------------------------------------
# Information criteria
# ---------------------------------------------------------------------------
def aic(loglik: float, k: int) -> float:
    return 2 * k - 2 * loglik


def aicc(loglik: float, k: int, n: int) -> float:
    """Small-sample corrected AIC. k counts every estimated parameter
    (including scale / sigma and degrees of freedom)."""
    if n - k - 1 <= 0:
        return np.inf
    return aic(loglik, k) + 2 * k * (k + 1) / (n - k - 1)


def bic(loglik: float, k: int, n: int) -> float:
    return k * np.log(n) - 2 * loglik


@dataclass
class FitResult:
    model: str
    params: dict
    loglik: float
    n: int
    k: int
    dist: object = field(default=None, repr=False)   # frozen scipy dist of the
                                                     # (error) distribution
    residuals: np.ndarray | None = field(default=None, repr=False)

    @property
    def aic(self) -> float:
        return aic(self.loglik, self.k)

    @property
    def aicc(self) -> float:
        return aicc(self.loglik, self.k, self.n)

    @property
    def bic(self) -> float:
        return bic(self.loglik, self.k, self.n)

    def ppf(self, p):
        """Quantile of the fitted (error) distribution."""
        return self.dist.ppf(p)

    def summary(self) -> str:
        ps = ", ".join(f"{k}={v:.6g}" for k, v in self.params.items())
        return (f"{self.model}: {ps} | logL={self.loglik:.4f} "
                f"AIC={self.aic:.4f} AICc={self.aicc:.4f} BIC={self.bic:.4f}")


def _clean(x):
    x = np.asarray(x, dtype=float)
    return x[~np.isnan(x)]


# ---------------------------------------------------------------------------
# Univariate distributions
# ---------------------------------------------------------------------------
def fit_normal(x) -> FitResult:
    """MLE normal: mu = mean, sigma = std with ddof=0."""
    x = _clean(x)
    mu, sigma = x.mean(), x.std(ddof=0)
    ll = stats.norm.logpdf(x, mu, sigma).sum()
    return FitResult("Normal", {"mu": mu, "sigma": sigma}, ll, len(x), 2,
                     stats.norm(mu, sigma))


def _t_negll(theta, z):
    """theta = (log scale, log(nu - 2)) on residuals z; nu > 2 keeps
    the variance finite."""
    s, nu = np.exp(theta[0]), 2.0 + np.exp(theta[1])
    return -np.sum(stats.t.logpdf(z / s, nu) - np.log(s))


def fit_t(x, nu0: float = 5.0) -> FitResult:
    """MLE location-scale Student t (mu, sigma, nu), nu constrained > 2."""
    x = _clean(x)

    def nll(th):
        return _t_negll(th[1:], x - th[0])

    s0 = x.std(ddof=1) * np.sqrt((nu0 - 2) / nu0)
    th0 = np.array([np.median(x), np.log(s0), np.log(nu0 - 2)])
    res = optimize.minimize(nll, th0, method="Nelder-Mead",
                            options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10})
    mu, s, nu = res.x[0], np.exp(res.x[1]), 2 + np.exp(res.x[2])
    return FitResult("Student-t", {"mu": mu, "sigma": s, "nu": nu}, -res.fun,
                     len(x), 3, stats.t(nu, mu, s))


# ---------------------------------------------------------------------------
# Regression  y = alpha + X beta + e
# ---------------------------------------------------------------------------
def _design(X, add_const=True):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if add_const:
        X = np.column_stack([np.ones(len(X)), X])
    return X


def ols(y, X, add_const: bool = True) -> dict:
    """Ordinary least squares with classical standard errors.

    Returns dict: beta (incl. intercept first if add_const), se, sigma
    (residual std, ddof = n - p), residuals, r2.
    """
    y = np.asarray(y, dtype=float)
    Xd = _design(X, add_const)
    n, p = Xd.shape
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    s2 = resid @ resid / (n - p)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(Xd.T @ Xd)))
    r2 = 1 - resid @ resid / np.sum((y - y.mean()) ** 2)
    return {"beta": beta, "se": se, "sigma": np.sqrt(s2), "residuals": resid, "r2": r2}


def _beta_names(p, add_const):
    names = ["alpha"] if add_const else []
    names += [f"beta{i}" if p - add_const > 1 else "beta"
              for i in range(1, p - add_const + 1)]
    return names


def fit_regression_normal(y, X, add_const: bool = True) -> FitResult:
    """MLE regression with Normal errors (beta = OLS, sigma uses ddof=0)."""
    y = np.asarray(y, dtype=float)
    Xd = _design(X, add_const)
    n, p = Xd.shape
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    sigma = np.sqrt(resid @ resid / n)
    ll = stats.norm.logpdf(resid, 0, sigma).sum()
    params = dict(zip(_beta_names(p, add_const), beta))
    params["sigma"] = sigma
    return FitResult("Regression-Normal", params, ll, n, p + 1,
                     stats.norm(0, sigma), resid)


def fit_regression_t(y, X, add_const: bool = True, nu0: float = 5.0) -> FitResult:
    """MLE regression with Student-t errors: params alpha, beta(s), sigma, nu.
    Starts from OLS."""
    y = np.asarray(y, dtype=float)
    Xd = _design(X, add_const)
    n, p = Xd.shape
    o = ols(y, Xd, add_const=False)

    def nll(th):
        return _t_negll(th[p:], y - Xd @ th[:p])

    th0 = np.concatenate([o["beta"],
                          [np.log(o["sigma"] * np.sqrt((nu0 - 2) / nu0)), np.log(nu0 - 2)]])
    res = optimize.minimize(nll, th0, method="Nelder-Mead",
                            options={"maxiter": 50000, "xatol": 1e-10, "fatol": 1e-10})
    res = optimize.minimize(nll, res.x, method="BFGS")   # polish
    beta = res.x[:p]
    s, nu = np.exp(res.x[p]), 2 + np.exp(res.x[p + 1])
    params = dict(zip(_beta_names(p, add_const), beta))
    params.update(sigma=s, nu=nu)
    return FitResult("Regression-t", params, -res.fun, n, p + 2,
                     stats.t(nu, 0, s), y - Xd @ beta)


def compare_fits(*fits: FitResult):
    """Table of fits sorted by AICc (best first)."""
    import pandas as pd
    df = pd.DataFrame([{"model": f.model, "k": f.k, "loglik": f.loglik,
                        "AIC": f.aic, "AICc": f.aicc, "BIC": f.bic} for f in fits])
    return df.sort_values("AICc").reset_index(drop=True)
