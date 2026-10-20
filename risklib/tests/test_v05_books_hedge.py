import numpy as np
import pandas as pd
import pytest

import risklib as rl


@pytest.fixture
def pnl():
    rng = np.random.default_rng(3)
    return pd.DataFrame(rng.standard_t(4, (50_000, 4)) * [1, 2, 3, 4], columns=list("abcd"))


def test_group_var_es_diversification(pnl):
    t = rl.group_var_es(pnl, {"A": ["a", "b"], "B": ["c", "d"]}, (0.05,))
    assert t.loc["Total", "VaR 5%"] == pytest.approx(rl.var_hist(pnl.sum(axis=1), 0.05))
    assert t.loc["Diversification", "ES 5%"] > 0          # ES is subadditive
    s = pd.Series({"a": "A", "b": "A", "c": "B", "d": "B"})
    assert np.allclose(rl.group_var_es(pnl, s).values, t.values)


def test_es_contributions_add_up(pnl):
    c = rl.es_contributions(pnl, 0.05)
    assert c.loc["Total", "ES contribution"] == pytest.approx(rl.es_hist(pnl.sum(axis=1), 0.05))
    assert c["share"].drop("Total").sum() == pytest.approx(1)
    g = rl.es_contributions(pnl, 0.05, {"A": ["a", "b"], "B": ["c", "d"]})
    assert g.loc["Total", "ES contribution"] == pytest.approx(c.loc["Total", "ES contribution"])


def test_capm_and_hedge():
    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.01, 5000)
    s = 0.0001 + 1.3 * m + rng.normal(0, 0.01, 5000)
    c = rl.capm(s, m)
    assert c["beta"] == pytest.approx(1.3, abs=0.05)
    assert c["beta"] == pytest.approx(rl.hedge_ratio(s, m))
    assert c["systematic_share"] == pytest.approx(c["r2"], abs=1e-3)
    t = rl.hedged_var_es(s, m, 1e6)
    assert t.loc["Hedged", "Hist VaR 5%"] < t.loc["Unhedged", "Hist VaR 5%"]
    assert t.attrs["beta"] == pytest.approx(c["beta"])


def test_copula_simulate_pnl_matches_table():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.multivariate_normal([0, 0], [[1, .5], [.5, 1]], 500) * 0.01, columns=["x", "y"])
    pnl, info = rl.copula_simulate_pnl(X, [100, 200], margin="normal", n_sims=20_000, seed=7)
    tab, R, _ = rl.copula_var_es(X, [100, 200], margin="normal", n_sims=20_000, seed=7)
    assert tab.loc["Portfolio", "VaR 5%"] == pytest.approx(rl.var_hist(pnl.sum(axis=1), 0.05))
    assert set(info["margins"]) == {"x", "y"}
