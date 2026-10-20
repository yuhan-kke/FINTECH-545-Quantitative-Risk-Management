"""Checks against worked examples printed in the lecture notes."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import risklib as rl


def test_week06_worked_examples():
    T = 125 / 255
    c = rl.gbsm(True, 100, 100, T, 0.05, 0.05, 0.20)
    assert c == pytest.approx(6.8089, abs=2e-3)                       # Sec 4.1
    assert rl.implied_vol(True, 6.0, 100, 100, T, 0.05, 0.05) == pytest.approx(0.17, abs=0.005)
    assert rl.gbsm(False, 50, 100, 1, 0.08, 0.08, 0.30) == pytest.approx(42.46, abs=0.01)
    assert rl.gbsm(False, 30, 100, 1, 0.08, 0.08, 0.30) == pytest.approx(62.31, abs=0.01)
    # two-step tree: 6.32, BSM 6.80, N=1000 -> 6.8039 vs 6.8050
    assert rl.binomial_european(True, 100, 100, 1, 0.05, 0.05, 0.1, 2) == pytest.approx(6.32, abs=0.01)
    assert rl.gbsm(True, 100, 100, 1, 0.05, 0.05, 0.1) == pytest.approx(6.8050, abs=1e-4)
    assert rl.binomial_european(True, 100, 100, 1, 0.05, 0.05, 0.1, 1000) == pytest.approx(6.8039, abs=1e-3)
    p = rl.binomial_params(100, 1, 0.05, 0.05, 0.1, 2)
    assert p["p"] == pytest.approx(0.6612, abs=1e-4)
    P = rl.gbsm(False, 100, 100, T, 0.05, 0.05, 0.2)
    assert rl.put_call_parity_gap(c, P, 100, 100, T, 0.05) == pytest.approx(0, abs=1e-10)
    m, se = rl.mc_european(True, 100, 100, T, 0.05, 0.05, 0.2, 200_000, seed=1)
    assert abs(m - c) < 4 * se


def test_week04_delta_normal_example():
    grad = np.array([7 / 270 * (10 * 0.5 + 20 * 1), 10 / 270 * 5])
    S = np.array([[0.01, 0.0075], [0.0075, 0.0225]])
    assert np.sqrt(grad @ S @ grad) == pytest.approx(0.0823, abs=1e-4)
    assert rl.delta_normal_var(270 * grad, S) == pytest.approx(36.55, abs=0.05)


def test_week05_kendall_example_and_es_ratio():
    x = [-2.1, -1.4, -0.9, -0.3, 0.4, 0.8, 1.5, 2.6]
    y = [0.2, -0.5, -1.8, -1.1, 1.0, 0.6, 2.2, 1.7]
    tau = stats.kendalltau(x, y)[0]
    assert tau == pytest.approx(0.5)
    assert np.sin(np.pi * tau / 2) == pytest.approx(0.7071, abs=1e-4)
    assert rl.es_normal(1) / rl.var_normal(1) == pytest.approx(2.063 / 1.645, abs=1e-3)


def test_week03_neff_halflife():
    d = rl.ew_diagnostics(10_000, 0.94)
    assert d["n_eff"] == pytest.approx((1 + .94) / (1 - .94), rel=1e-6)
    assert d["half_life"] == pytest.approx(11.2, abs=0.05)


def test_nig_fit_recovers_skew():
    d = stats.norminvgauss(2.0, -1.0, 0.0, 0.01)
    x = d.rvs(5000, random_state=3)
    f = rl.fit_nig(x)
    assert f.params["beta"] < 0 and f.k == 4
    assert f.aicc < rl.fit_t(x).aicc + 5


def test_choose_copula_course_convention():
    rng = np.random.default_rng(0)
    Z = rng.multivariate_normal([0, 0, 0], [[1, .6, .3], [.6, 1, .4], [.3, .4, 1]], 2000)
    U = stats.norm.cdf(Z)
    tab, g, t = rl.choose_copula(U)
    assert g.k == 0 and t.k == 1 and 2 < t.nu <= 100
    assert tab.loc["t", "dBIC (t vs Gauss)"] < 0              # Gaussian data -> Gaussian wins
    W = rng.chisquare(3, (2000, 1)) / 3
    tab2, _, t2 = rl.choose_copula(stats.t.cdf(Z / np.sqrt(W), 3))
    assert tab2.loc["t", "dBIC (t vs Gauss)"] > 10 and t2.nu < 6


def test_weighted_hist():
    x = np.random.default_rng(2).normal(0, 1, 2000)
    v, e = rl.weighted_hist_var_es(x, 0.9999999, 0.05)
    assert v == pytest.approx(rl.var_hist(x, 0.05), abs=0.02)
    assert e > v
