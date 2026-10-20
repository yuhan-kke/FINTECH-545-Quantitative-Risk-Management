"""Gaussian and Student-t copulas.

Workflow used in the course:
  1. fit each margin (fitting.fit_normal / fit_t),
  2. U = F_i(x_i)            -> uniforms through the fitted margins
  3. R = sin(pi/2 * tau)     -> correlation from Kendall's tau
  4. fit nu of the t copula by MLE with R held fixed
  5. simulate copula uniforms, push through each margin's ppf.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .fitting import aic as _aic, aicc as _aicc, bic as _bic
from .psd import chol_psd, near_psd, is_psd


# ---------------------------------------------------------------------------
# Uniform transforms and tail counts
# ---------------------------------------------------------------------------
def rank_uniforms(x) -> np.ndarray:
    """Pseudo-observations rank/(n+1) in (0, 1), column by column."""
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    return stats.rankdata(X, axis=0) / (n + 1)


def margin_uniforms(x, fits) -> np.ndarray:
    """U[:, i] = F_i(x[:, i]) through each fitted margin (FitResult.dist)."""
    X = np.asarray(x, dtype=float)
    return np.column_stack([f.dist.cdf(X[:, i]) for i, f in enumerate(fits)])


def joint_tail_counts(U, q: float = 0.025, names=None) -> pd.DataFrame:
    """For every pair: days both in their lower q tail, both in upper q tail.
    U are uniforms (ranks or copula draws). Expected under independence
    is n * q^2 for each side."""
    U = np.asarray(U, dtype=float)
    d = U.shape[1]
    names = names or [f"X{i+1}" for i in range(d)]
    rows = []
    for i in range(d):
        for j in range(i + 1, d):
            lo = int(np.sum((U[:, i] <= q) & (U[:, j] <= q)))
            hi = int(np.sum((U[:, i] >= 1 - q) & (U[:, j] >= 1 - q)))
            rows.append({"pair": f"{names[i]}-{names[j]}", "lower": lo, "upper": hi})
    out = pd.DataFrame(rows)
    out.attrs["expected_independent"] = len(U) * q * q
    return out


# ---------------------------------------------------------------------------
# Correlation from Kendall's tau
# ---------------------------------------------------------------------------
def kendall_matrix(x) -> np.ndarray:
    return pd.DataFrame(np.asarray(x, dtype=float)).corr(method="kendall").values


def kendall_to_pearson(tau) -> np.ndarray:
    """R = sin(pi/2 * tau), valid for every elliptical copula. Repaired to
    PSD (Higham-free spectral fix) if needed."""
    R = np.sin(np.pi / 2 * np.asarray(tau, dtype=float))
    np.fill_diagonal(R, 1.0)
    return R if is_psd(R) else near_psd(R)


# ---------------------------------------------------------------------------
# Log densities (per observation)
# ---------------------------------------------------------------------------
def gaussian_copula_logpdf(U, R) -> np.ndarray:
    Z = stats.norm.ppf(np.clip(U, 1e-12, 1 - 1e-12))
    d = Z.shape[1]
    return (stats.multivariate_normal(np.zeros(d), R).logpdf(Z)
            - stats.norm.logpdf(Z).sum(axis=1))


def t_copula_logpdf(U, R, nu: float) -> np.ndarray:
    T = stats.t.ppf(np.clip(U, 1e-12, 1 - 1e-12), nu)
    d = T.shape[1]
    return (stats.multivariate_t(np.zeros(d), R, df=nu).logpdf(T)
            - stats.t.logpdf(T, nu).sum(axis=1))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
@dataclass
class CopulaFit:
    kind: str                      # 'gaussian' or 't'
    R: np.ndarray
    nu: float | None
    loglik: float
    n: int
    k: int                         # d(d-1)/2 correlations (+1 for nu)
    loglik_obs: np.ndarray = field(repr=False, default=None)

    @property
    def aic(self):
        return _aic(self.loglik, self.k)

    @property
    def aicc(self):
        return _aicc(self.loglik, self.k, self.n)

    @property
    def bic(self):
        return _bic(self.loglik, self.k, self.n)

    def simulate(self, n_sims: int, seed=None) -> np.ndarray:
        return simulate_copula(self, n_sims, seed)


def copula_corr(U, method: str = "spearman") -> np.ndarray:
    """Correlation matrix to plug into a Gaussian copula.
    'spearman' : Spearman rank correlation of U (FINTECH 545 course convention)
    'kendall'  : sin(pi/2 * Kendall tau) (exact for elliptical copulas)
    'pearson'  : Pearson correlation of the normal scores Phi^-1(U)."""
    U = np.asarray(U, dtype=float)
    if method == "spearman":
        R = pd.DataFrame(U).corr(method="spearman").values
    elif method == "kendall":
        return kendall_to_pearson(kendall_matrix(U))
    elif method == "pearson":
        R = np.corrcoef(stats.norm.ppf(np.clip(U, 1e-12, 1 - 1e-12)).T)
    else:
        raise ValueError("method must be 'spearman', 'kendall' or 'pearson'")
    return R if is_psd(R) else near_psd(R)


def fit_gaussian_copula(U, R=None, method: str = "kendall") -> CopulaFit:
    """Gaussian copula on uniforms U. R given -> used as is; otherwise built
    with copula_corr(U, method)."""
    U = np.asarray(U, dtype=float)
    R = copula_corr(U, method) if R is None else np.asarray(R)
    d = U.shape[1]
    ll = gaussian_copula_logpdf(U, R)
    return CopulaFit("gaussian", R, None, float(ll.sum()), len(U),
                     d * (d - 1) // 2, ll)


def fit_t_copula(U, R=None, nu_bounds=(1.0, 200.0)) -> CopulaFit:
    """R fixed from Kendall's tau; nu by 1-D maximum likelihood."""
    U = np.asarray(U, dtype=float)
    R = kendall_to_pearson(kendall_matrix(U)) if R is None else np.asarray(R)
    d = U.shape[1]
    res = optimize.minimize_scalar(lambda v: -t_copula_logpdf(U, R, v).sum(),
                                   bounds=nu_bounds, method="bounded",
                                   options={"xatol": 1e-6})
    nu = float(res.x)
    ll = t_copula_logpdf(U, R, nu)
    return CopulaFit("t", R, nu, float(ll.sum()), len(U),
                     d * (d - 1) // 2 + 1, ll)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate_copula(fit: CopulaFit, n_sims: int, seed=None) -> np.ndarray:
    """Uniform draws from a fitted copula."""
    rng = np.random.default_rng(seed)
    d = fit.R.shape[0]
    Z = rng.standard_normal((n_sims, d)) @ chol_psd(fit.R).T
    if fit.kind == "gaussian":
        return stats.norm.cdf(Z)
    W = rng.chisquare(fit.nu, size=(n_sims, 1)) / fit.nu
    return stats.t.cdf(Z / np.sqrt(W), fit.nu)


def uniforms_to_margins(U, fits) -> np.ndarray:
    """X[:, i] = F_i^-1(U[:, i])."""
    return np.column_stack([f.dist.ppf(U[:, i]) for i, f in enumerate(fits)])


def t_tail_dependence(rho: float, nu: float) -> float:
    """Lower (= upper) tail dependence of a bivariate t copula:
    lambda = 2 * t_{nu+1}( -sqrt((nu+1)(1-rho)/(1+rho)) ). Gaussian -> 0."""
    return float(2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1))


def fit_margins(df: pd.DataFrame):
    """Fit normal and t to each column and keep the one with lower AICc.
    Returns (list of chosen FitResults, comparison table)."""
    from .fitting import fit_normal, fit_t
    chosen, rows = [], []
    for c in df.columns:
        fn, ft = fit_normal(df[c]), fit_t(df[c])
        best = ft if ft.aicc < fn.aicc else fn
        chosen.append(best)
        rows.append({"series": c, "AICc normal": fn.aicc, "AICc t": ft.aicc,
                     "chosen": best.model, "t nu": ft.params["nu"],
                     "t sigma": ft.params["sigma"]})
    return chosen, pd.DataFrame(rows).set_index("series")


def compare_copulas(*fits: "CopulaFit") -> pd.DataFrame:
    return pd.DataFrame([{"copula": f.kind, "loglik": f.loglik, "nu": f.nu, "k": f.k,
                          "AICc": f.aicc, "BIC": f.bic} for f in fits]).set_index("copula")


def simulate_returns(fit: "CopulaFit", margins, n_sims: int, seed=None):
    """Copula uniforms pushed through the fitted margins.
    Returns (uniforms, returns)."""
    U = simulate_copula(fit, n_sims, seed)
    return U, uniforms_to_margins(U, margins)


def outer_tail_mask(U, q: float = 0.05) -> np.ndarray:
    """True on rows where any column is in its outer q on either side."""
    U = np.asarray(U, dtype=float)
    return ((U < q) | (U > 1 - q)).any(axis=1)


def copula_simulate_pnl(returns: pd.DataFrame, dollars, margin: str = "best",
                        method: str = "spearman", n_sims: int = 100_000, seed=None,
                        copula: str = "gaussian"):
    """Simulate $ P&L for each asset with a fitted copula.

    returns : DataFrame of (usually de-meaned) returns, one column per asset
    dollars : $ held in each asset (same order as the columns)
    margin  : 'best' (normal vs t by AICc per column), 't' or 'normal'
    method  : copula correlation, see copula_corr ('spearman' = course)
    copula  : 'gaussian' or 't'
    Returns (pnl DataFrame n_sims x assets, info dict with R, margins, fit).
    """
    from .fitting import fit_normal, fit_t
    cols = list(returns.columns)
    if margin == "best":
        margins, _ = fit_margins(returns)
    elif margin == "t":
        margins = [fit_t(returns[c]) for c in cols]
    elif margin == "normal":
        margins = [fit_normal(returns[c]) for c in cols]
    else:
        raise ValueError("margin must be 'best', 't' or 'normal'")
    U = margin_uniforms(returns, margins)
    R = copula_corr(U, method)
    fit = fit_gaussian_copula(U, R) if copula == "gaussian" else fit_t_copula(U, R)
    _, sim = simulate_returns(fit, margins, n_sims, seed)
    pnl = pd.DataFrame(sim * np.asarray(dollars, dtype=float), columns=cols)
    return pnl, {"R": pd.DataFrame(R, cols, cols), "margins": dict(zip(cols, margins)),
                 "fit": fit}


def copula_var_es(returns: pd.DataFrame, dollars, alphas=(0.05,), margin: str = "best",
                  method: str = "spearman", n_sims: int = 100_000, seed=None):
    """One-shot Gaussian-copula VaR/ES in dollars for each column and the
    portfolio. Returns (table, corr_matrix, margins list)."""
    from .risk import var_es_table
    pnl, info = copula_simulate_pnl(returns, dollars, margin, method, n_sims, seed)
    d = {c: pnl[c].values for c in pnl}
    d["Portfolio"] = pnl.sum(axis=1).values
    return var_es_table(d, alphas), info["R"], list(info["margins"].values())


def hist_var_es(returns: pd.DataFrame, dollars, alphas=(0.05,)):
    """Historical-simulation VaR/ES in dollars for each column and the portfolio."""
    from .risk import var_es_table
    w = np.asarray(dollars, dtype=float)
    X = returns.values
    pnl = {c: w[i] * X[:, i] for i, c in enumerate(returns.columns)}
    pnl["Portfolio"] = X @ w
    return var_es_table(pnl, alphas)


def hist_pnl(returns: pd.DataFrame, dollars) -> pd.DataFrame:
    """Historical $ P&L per asset: each past return times today's $ holding."""
    return returns * np.asarray(dollars, dtype=float)


def choose_copula(U, R=None, nu_max: float = 100.0):
    """Gaussian vs t copula, the Week 05 Sec 6.5 way.

    Margins are already fitted and frozen; both copulas get the SAME R from
    Kendall's tau (sin(pi*tau/2)); nu is profiled on 2 < nu <= nu_max with the
    search done over theta = 1/nu; parameter counts k_Gauss = 0, k_t = 1.
    Returns (table, gaussian_fit, t_fit). Take the t when dBIC > 0.
    """
    U = np.asarray(U, dtype=float)
    R = kendall_to_pearson(kendall_matrix(U)) if R is None else np.asarray(R)
    if not is_psd(R):
        R = near_psd(R)
    m = len(U)
    llg = gaussian_copula_logpdf(U, R)
    thetas = np.linspace(1 / nu_max, 0.49, 200)              # 200-point grid in 1/nu
    prof = np.array([t_copula_logpdf(U, R, 1 / th).sum() for th in thetas])
    i = int(np.argmax(prof))
    lo, hi = thetas[max(i - 1, 0)], thetas[min(i + 1, len(thetas) - 1)]
    res = optimize.minimize_scalar(lambda th: -t_copula_logpdf(U, R, 1 / th).sum(),
                                   bounds=(lo, hi), method="bounded")
    nu = float(1 / res.x)
    llt = t_copula_logpdf(U, R, nu)
    g = CopulaFit("gaussian", R, None, float(llg.sum()), m, 0, llg)
    t = CopulaFit("t", R, nu, float(llt.sum()), m, 1, llt)
    tab = compare_copulas(g, t)
    tab["dBIC (t vs Gauss)"] = [np.nan, 2 * (t.loglik - g.loglik) - np.log(m)]
    return tab, g, t
