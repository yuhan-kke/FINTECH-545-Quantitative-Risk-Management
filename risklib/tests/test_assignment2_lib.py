import numpy as np
import pandas as pd
import pytest
from scipy import stats

import risklib as rl


def test_normal_var_es_closed_form():
    assert rl.var_normal(1.0) == pytest.approx(1.644854, abs=1e-6)
    assert rl.es_normal(1.0) == pytest.approx(2.062713, abs=1e-6)


def test_t_es_matches_integration():
    d = stats.t(4, 0.01, 0.02)
    assert rl.es_t(4, 0.01, 0.02) == pytest.approx(rl.es_dist(d), rel=1e-6)
    assert rl.var_t(4, 0.01, 0.02) == pytest.approx(rl.var_dist(d))


def test_hist_converges_to_normal():
    x = np.random.default_rng(0).normal(0, 1, 1_000_000)
    assert rl.var_hist(x) == pytest.approx(1.6449, abs=0.01)
    assert rl.es_hist(x) == pytest.approx(2.0627, abs=0.01)


def test_ew_diagnostics():
    d = rl.ew_diagnostics(10_000, 0.94, recent=10)
    assert d["n_eff"] == pytest.approx((1 + .94) / (1 - .94), rel=1e-6)
    assert d["half_life"] == pytest.approx(11.2023, abs=1e-3)
    assert d["weight_last_10"] == pytest.approx(1 - .94 ** 10, rel=1e-9)


def test_kendall_to_pearson_and_copula_fit():
    R = np.array([[1, .6, .3], [.6, 1, .2], [.3, .2, 1]])
    X = stats.multivariate_t(np.zeros(3), R, df=4).rvs(5000, random_state=1)
    U = rl.rank_uniforms(X)
    tf, gf = rl.fit_t_copula(U), rl.fit_gaussian_copula(U)
    np.testing.assert_allclose(tf.R, R, atol=0.04)
    assert 3 < tf.nu < 6
    assert tf.aicc < gf.aicc
    sim = tf.simulate(50_000, seed=2)
    assert sim.min() > 0 and sim.max() < 1
    np.testing.assert_allclose(rl.kendall_to_pearson(rl.kendall_matrix(sim[:20000])),
                               R, atol=0.03)


def test_tail_dependence():
    assert rl.t_tail_dependence(0.5, 1e6) == pytest.approx(0, abs=1e-6)
    assert rl.t_tail_dependence(0.5, 4) == pytest.approx(0.25, abs=0.01)


def test_factor_identity():
    rng = np.random.default_rng(3)
    m = rng.normal(0, .01, 500)
    df = pd.DataFrame({"M": m, "A": 1.2 * m + rng.normal(0, .01, 500),
                       "B": .8 * m + rng.normal(0, .01, 500)})
    fm = rl.fit_factor_model(df, "M")
    # OLS residuals are orthogonal to the market => implied cov == sample cov
    np.testing.assert_allclose(fm.implied_cov(), df[["A", "B"]].cov().values, atol=1e-15)
    sims = fm.simulate(200_000, seed=1)
    np.testing.assert_allclose(np.cov(sims.T), fm.implied_cov(), rtol=0.02)
