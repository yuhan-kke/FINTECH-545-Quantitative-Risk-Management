from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import risklib as rl

DATA = Path(__file__).parent / "data"
rng = np.random.default_rng(0)


# ---------------- moments ----------------
def test_moments_match_scipy():
    x = rng.standard_t(5, 2000)
    m = rl.first4_moments(x)
    assert m.variance == pytest.approx(np.var(x, ddof=1))
    assert m.skew == pytest.approx(stats.skew(x, bias=False))
    assert m.excess_kurtosis == pytest.approx(stats.kurtosis(x, bias=False))


def test_tail_exceedances_shape():
    t = rl.tail_exceedances(rng.normal(size=5000))
    assert {"observed", "expected"} <= set(t.columns)
    assert t.loc[t.p == 0.05, "ratio"].iloc[0] == pytest.approx(1, abs=0.25)


# ---------------- fitting ----------------
def test_fit_t_recovers_params():
    x = stats.t.rvs(4, loc=0.5, scale=2, size=20000, random_state=1)
    f = rl.fit_t(x)
    assert f.params["nu"] == pytest.approx(4, rel=0.15)
    assert f.params["sigma"] == pytest.approx(2, rel=0.05)
    assert f.params["mu"] == pytest.approx(0.5, abs=0.05)
    assert f.k == 3


def test_regression_t_beats_normal_on_fat_tails():
    x = rng.normal(size=1000)
    y = 1.5 + 3 * x + stats.t.rvs(3, scale=1, size=1000, random_state=2)
    fn, ft = rl.fit_regression_normal(y, x), rl.fit_regression_t(y, x)
    assert ft.params["beta"] == pytest.approx(3, abs=0.1)
    assert ft.aicc < fn.aicc
    assert rl.compare_fits(fn, ft).model.iloc[0] == "Regression-t"
    o = rl.ols(y, x)
    assert fn.params["beta"] == pytest.approx(o["beta"][1])


def test_aicc_formula():
    assert rl.aicc(-100, 3, 200) == pytest.approx(206 + 24 / 196)


# ---------------- covariance / psd ----------------
@pytest.fixture
def p1():
    return pd.read_csv(DATA / "missing_returns.csv").set_index("Day")


def test_joint_counts(p1):
    jc = rl.joint_counts(p1)
    assert jc.loc["C", "D"] == 83 and jc.loc["IDX", "IDX"] == 250
    assert len(p1.dropna()) == 81


def test_pairwise_matches_pandas(p1):
    R = rl.missing_cov(p1, skip_missing=False, corr=True)
    np.testing.assert_allclose(R.values, p1.corr().values, atol=1e-12)
    C = rl.missing_cov(p1, skip_missing=False)
    np.testing.assert_allclose(C.values, p1.cov().values, atol=1e-14)


def test_pairwise_not_psd_complete_is(p1):
    assert rl.is_psd(rl.missing_cov(p1, True, corr=True))
    Rpw = rl.missing_cov(p1, False, corr=True)
    assert rl.min_eigenvalue(Rpw) == pytest.approx(-0.011072, abs=1e-5)
    assert not rl.try_cholesky(Rpw)[0]
    with pytest.raises(ValueError):
        rl.chol_psd(Rpw)


def test_repairs(p1):
    Rpw = rl.missing_cov(p1, False, corr=True).values
    sd = p1.std().values
    w = np.array([-.4, -.3, -.2, -.1, 1])
    for fix, dist, var in [(rl.near_psd, 0.016182, 6.8073e-7),
                           (rl.higham_psd, 0.015520, 6.7876e-7)]:
        R = fix(Rpw)
        assert rl.is_psd(R)
        np.testing.assert_allclose(np.diag(R), 1)
        assert rl.frobenius(R, Rpw) == pytest.approx(dist, abs=1e-5)
        assert w @ rl.cor2cov(R, sd) @ w == pytest.approx(var, rel=1e-3)
        L = rl.chol_psd(R)
        np.testing.assert_allclose(L @ L.T, R, atol=1e-8)


def test_higham_is_nearest():
    A = np.array([[1, .9, .7], [.9, 1, -.4], [.7, -.4, 1.]])
    H, RJ = rl.higham_psd(A), rl.near_psd(A)
    assert rl.frobenius(H, A) <= rl.frobenius(RJ, A) + 1e-12


def test_repair_covariance_keeps_variances():
    A = np.array([[1, .9, .7], [.9, 1, -.4], [.7, -.4, 1.]])
    S = rl.cor2cov(A, [1, 2, 3])
    np.testing.assert_allclose(np.diag(rl.higham_psd(S)), [1, 4, 9])
    np.testing.assert_allclose(np.diag(rl.near_psd(S)), [1, 4, 9])


def test_chol_psd_singular():
    v = np.array([1., 2, 3])
    A = np.outer(v, v)
    L = rl.chol_psd(A)
    np.testing.assert_allclose(L @ L.T, A, atol=1e-10)


def test_ew_cov_lambda_one_limit():
    X = rng.normal(size=(300, 3))
    S = rl.ew_cov(X, lam=0.999999).values
    np.testing.assert_allclose(S, np.cov(X.T, ddof=0), atol=1e-4)


# ---------------- mvn ----------------
def test_conditional_normal():
    S = np.array([[1.0, 0.8], [0.8, 2.0]])
    m, c, b = rl.conditional_normal([0, 1], S, [0], [1.0])
    assert b[0, 0] == pytest.approx(0.8)
    assert m[0] == pytest.approx(1.8)
    assert c[0, 0] == pytest.approx(2 - 0.64)
    m2, _, _ = rl.conditional_normal([0, 1], S, 0, np.array([0.0, 1.0, 2.0]))
    assert m2.shape == (3, 1)


def test_simulation_recovers_cov():
    S = np.array([[1, .5, .2], [.5, 2, .3], [.2, .3, 1.5]])
    for sim in (rl.simulate_normal(200000, S, seed=3),
                rl.simulate_pca(200000, S, 1.0, seed=3)):
        np.testing.assert_allclose(np.cov(sim.T), S, atol=0.03)


# ---------------- time series / returns ----------------
def test_select_arma_picks_ar2():
    n, e = 1500, rng.normal(size=1500)
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = 0.5 * x[t - 1] + 0.3 * x[t - 2] + e[t]
    tab = rl.select_arma(x)
    assert tab.model.iloc[0] in ("AR(2)", "AR(3)")


def test_return_calculate():
    p = pd.DataFrame({"Day": [0, 1, 2], "P": [100., 110, 99]})
    r = rl.return_calculate(p, "arithmetic", "Day")
    np.testing.assert_allclose(r.P, [0.1, -0.1])
    r = rl.return_calculate(p, "log", "Day")
    np.testing.assert_allclose(r.P, np.log([1.1, 0.9]))
