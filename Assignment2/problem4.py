# Problem 4 - Gaussian or t copula
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "risklib" / "src"))   # use risklib without pip install
import numpy as np
import pandas as pd
import risklib as rl

POS = 1_000_000
d = pd.read_csv(HERE / "data" / "problem4.csv")
names = list(d.columns)

# ---- (a) moments, and how much the biggest day matters
print("(a)\n", rl.moments_table(d))
for c in names:
    i, x, m = rl.drop_most_extreme(d[c])
    print(f"{c}: drop row {i} ({x:.4f}) -> skew {m.skew:.3f}, excess kurt {m.excess_kurtosis:.3f}")

# ---- (b) ranks and joint 2.5% tail counts
U_rank = rl.rank_uniforms(d)
rl.plots.plot_pairs(U_rank, names, HERE / "figures" / "p4_rank_pairs.png")
print("\n(b)\n", rl.joint_tail_counts(U_rank, 0.025, names),
      "\nexpected if independent:", 1000 * 0.025 ** 2)

# ---- (c) margins
margins, table = rl.fit_margins(d)
print("\n(c)\n", table)

# ---- (d) copulas
U = rl.margin_uniforms(d, margins)
R = rl.kendall_to_pearson(rl.kendall_matrix(U))
gauss = rl.fit_gaussian_copula(U, R)
tcop = rl.fit_t_copula(U, R)
print("\n(d) R =\n", R.round(4), "\n", rl.compare_copulas(gauss, tcop))

# ---- (e) simulate 100,000 days and compute portfolio VaR / ES
U_g, X_g = rl.simulate_returns(gauss, margins, 100_000, seed=545)
U_t, X_t = rl.simulate_returns(tcop, margins, 100_000, seed=546)
pnl = {"Gaussian": POS * X_g.sum(axis=1),
       "t": POS * X_t.sum(axis=1),
       "Historical": POS * d.sum(axis=1).values}
print("\n(e)\n", rl.var_es_table(pnl, [0.05, 0.01]).round(0))

# ---- (f) joint tail days per 1,000 days
for name, Us in [("Gaussian", U_g), ("t", U_t)]:
    counts = rl.joint_tail_counts(Us, 0.025, names)
    counts[["lower", "upper"]] *= 1000 / len(Us)
    print(f"\n(f) {name}\n", counts.round(2))

# ---- (i) where does the likelihood gain come from?
gain = tcop.loglik_obs - gauss.loglik_obs
tail = rl.outer_tail_mask(U, 0.05)
print("\n(i) total gain:", gain.sum().round(3))
print("days with NO series in outer 5% :", (~tail).sum(),      # count   -> 793
      " contribution:", gain[~tail].sum().round(3))           # sum     -> +50.478
print("days with >=1 series in outer 5%:", tail.sum(),         # count   -> 207
      " contribution:", gain[tail].sum().round(3))            # sum     -> +51.081
print("biggest single days:", np.argsort(gain)[::-1][:3], np.sort(gain)[::-1][:3].round(3))
keep = np.arange(len(U)) != 951                    # refit without the crash day
g2, t2 = rl.fit_gaussian_copula(U[keep], R), rl.fit_t_copula(U[keep], R)
print("without row 951: nu =", round(t2.nu, 2), " AICc gap =", round(g2.aicc - t2.aicc, 3))
rl.plots.plot_bars(gain, HERE / "figures" / "p4_ll_contrib.png", "l_t - l_Gauss per day", highlight=tail)

# ---- (j) tail dependence, most correlated pair (X1-X2)
print("\n(j) lambda_t(X1,X2) =", round(rl.t_tail_dependence(R[0, 1], tcop.nu), 4), " Gaussian = 0")
