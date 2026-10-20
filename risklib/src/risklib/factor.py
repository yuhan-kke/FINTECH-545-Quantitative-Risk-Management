"""Single-factor (market model) fitting and simulation:
    r_i = alpha_i + beta_i * r_M + e_i,   r_M ~ N(mu_M, s_M^2),  e ~ N(0, S_e)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fitting import ols
from .psd import chol_psd


@dataclass
class FactorModel:
    names: list
    alpha: np.ndarray
    beta: np.ndarray
    resid: np.ndarray            # n x k OLS residuals
    market_mean: float
    market_sd: float
    ddof: int = 1

    @property
    def resid_cov(self) -> np.ndarray:
        return np.cov(self.resid.T, ddof=self.ddof)

    @property
    def resid_sd(self) -> np.ndarray:
        return np.sqrt(np.diag(self.resid_cov))

    @property
    def resid_corr(self) -> np.ndarray:
        s = self.resid_sd
        return self.resid_cov / np.outer(s, s)

    def implied_cov(self, diagonal_resid: bool = False) -> np.ndarray:
        """beta beta' s_M^2 + S_e  (S_e diagonal if diagonal_resid)."""
        Se = self.resid_cov
        if diagonal_resid:
            Se = np.diag(np.diag(Se))
        return np.outer(self.beta, self.beta) * self.market_sd ** 2 + Se

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({"alpha": self.alpha, "beta": self.beta,
                             "resid_sd": self.resid_sd}, index=self.names)

    def simulate(self, n_sims: int, diagonal_resid: bool = False,
                 zero_mean: bool = True, seed=None) -> np.ndarray:
        """Simulated stock returns (n_sims x k).
        zero_mean=True drops alpha and the market mean (zero expected
        returns); otherwise they are included."""
        rng = np.random.default_rng(seed)
        Se = self.resid_cov
        if diagonal_resid:
            Se = np.diag(np.diag(Se))
        mM = 0.0 if zero_mean else self.market_mean
        rM = mM + self.market_sd * rng.standard_normal(n_sims)
        E = rng.standard_normal((n_sims, len(self.beta))) @ chol_psd(Se).T
        a = 0.0 if zero_mean else self.alpha
        return a + np.outer(rM, self.beta) + E


def fit_factor_model(returns: pd.DataFrame, market: str, ddof: int = 1) -> FactorModel:
    """OLS of every non-market column on the market column."""
    names = [c for c in returns.columns if c != market]
    m = returns[market].values
    alphas, betas, resids = [], [], []
    for c in names:
        o = ols(returns[c].values, m)
        alphas.append(o["beta"][0])
        betas.append(o["beta"][1])
        resids.append(o["residuals"])
    return FactorModel(names, np.array(alphas), np.array(betas),
                       np.column_stack(resids), float(m.mean()),
                       float(m.std(ddof=ddof)), ddof)


def factor_var_table(fm: FactorModel, portfolios: dict, n_sims: int, seed=None,
                     alpha: float = 0.05) -> pd.DataFrame:
    """Simulated VaR of each portfolio with the full and the diagonal residual
    covariance (zero expected returns)."""
    from .risk import var_hist
    out = {}
    for label, diag in [("full residual cov", False), ("independent residuals", True)]:
        sims = fm.simulate(n_sims, diagonal_resid=diag, zero_mean=True, seed=seed)
        out[label] = {p: var_hist(sims @ np.asarray(w, float), alpha)
                      for p, w in portfolios.items()}
    return pd.DataFrame(out).T
