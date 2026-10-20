import numpy as np
import pandas as pd
import pytest

import risklib as rl


@pytest.fixture
def X():
    rng = np.random.default_rng(1)
    C = np.array([[1, .6, .3], [.6, 1, .4], [.3, .4, 1]]) * 1e-4
    return pd.DataFrame(rng.multivariate_normal([0.001, 0, -0.001], C, 400),
                        columns=["a", "b", "c"])


def test_ew_cov_simple_mean_matches_manual(X):
    lam = 0.94
    w = rl.ew_weights(len(X), lam)
    Xc = X.values - X.values.mean(axis=0)
    manual = (w[:, None] * Xc).T @ Xc
    assert np.allclose(rl.ew_cov(X, lam).values, manual)
    assert np.allclose(rl.ew_var(X, lam).values, np.diag(manual))
    assert not np.allclose(rl.ew_cov(X, lam, mean="weighted").values, manual)


def test_ew_cov_mixed_combines(X):
    S = rl.ew_cov_mixed(X, lam_var=0.97, lam_corr=0.94).values
    assert np.allclose(np.diag(S), rl.ew_var(X, 0.97).values)
    R, _ = rl.cov2cor(S)
    assert np.allclose(R, rl.ew_cov(X, 0.94, corr=True).values)


def test_psd_status_and_pca():
    assert rl.psd_status(np.eye(3)) == "Positive Definite"
    assert rl.psd_status(np.ones((3, 3))) == "Positive Semi-Definite"
    bad = np.array([[1, .9, -.9], [.9, 1, .9], [-.9, .9, 1]])
    assert rl.psd_status(bad) == "Non-Definite"
    assert rl.psd_status(rl.higham_psd(bad)) != "Non-Definite"
    t = rl.pca_explained(np.diag([3.0, 1.0]))
    assert np.allclose(t["explained"], [0.75, 0.25])
    assert t["cumulative"].iloc[-1] == pytest.approx(1.0)


def test_missing_corr_pairwise():
    df = pd.DataFrame({"x": [1, 2, 3, np.nan, 5.0], "y": [2, 1, 4, 3, np.nan]})
    r = rl.missing_corr(df, skip_missing=False)
    assert r.loc["x", "y"] == pytest.approx(np.corrcoef([1, 2, 3], [2, 1, 4])[0, 1])


def test_relative_var():
    d = rl.fit_normal(np.random.default_rng(0).normal(0.01, 0.02, 5000)).dist
    assert rl.var_dist(d, 0.05, relative=True) - rl.var_dist(d, 0.05) == pytest.approx(d.mean())
    tab = rl.fitted_var_es_table({"N": d}, alphas=(0.05, 0.01), position=100)
    assert tab.loc["N", "VaR 5%"] == pytest.approx(100 * rl.var_dist(d, 0.05))


def test_copula_corr_methods(X):
    U = rl.rank_uniforms(X)
    for m in ("spearman", "kendall", "pearson"):
        R = rl.copula_corr(U, m)
        assert np.allclose(np.diag(R), 1) and rl.is_psd(R)
    assert np.allclose(rl.copula_corr(U, "spearman"), X.corr(method="spearman").values)


def test_copula_var_es_and_hist(X):
    tab, R, margins = rl.copula_var_es(X - X.mean(), [100, 200, 300], alphas=(0.05,),
                                       margin="t", n_sims=20_000, seed=1)
    assert list(tab.index) == ["a", "b", "c", "Portfolio"]
    assert (tab["VaR 5%"] > 0).all()
    h = rl.hist_var_es(X - X.mean(), [100, 200, 300])
    assert tab.loc["Portfolio", "VaR 5%"] == pytest.approx(h.loc["Portfolio", "VaR 5%"], rel=0.2)


def test_equal_weight_shares():
    p = pd.DataFrame({"Date": ["d1", "d2"], "A": [10.0, 20.0], "B": [5.0, 50.0]})
    s = rl.equal_weight_shares(p, 100_000)
    assert s.loc["A", "shares"] == pytest.approx(2500)
    assert s.loc["B", "shares"] == pytest.approx(1000)
    assert rl.holdings_value(p, 100).sum() == pytest.approx(7000)
