# Problem 2 - A volatility estimate after the regime changed
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "risklib" / "src"))   # use risklib without pip install
import pandas as pd
import risklib as rl

POS = 1_000_000
prices = pd.read_csv(HERE / "data" / "problem2.csv")
r = rl.return_calculate(prices, "arithmetic", "Day")["Price"].values
r = r - r.mean()                                    # remove the sample mean
n = len(r)

# ---- Predict
brk = rl.variance_breakpoint(r)                     # where the regime changes
k = brk["n_after"]
print("regime break:", brk)
rl.plots.plot_series(r * 100, HERE / "figures" / "p2_returns.png", "Demeaned returns", "%", n - k)

print("\n(b)")
for lam in (0.94, 0.97):
    print(rl.ew_diagnostics(n, lam, recent=k))
print("\n(c)", rl.first4_moments(r))

# ---- (d) five VaRs
sd_eq, sd_97, sd_94 = r.std(ddof=1), rl.ew_std(r, 0.97), rl.ew_std(r, 0.94)
t = rl.fit_t(r)
print("\n(d) 5% VaR on $1M")
print("1 Normal equal weight:", rl.var_normal(sd_eq) * POS)
print("2 Normal EW 0.97     :", rl.var_normal(sd_97) * POS)
print("3 Normal EW 0.94     :", rl.var_normal(sd_94) * POS)
print("4 Student t          :", rl.var_t(t.params["nu"], t.params["mu"], t.params["sigma"]) * POS)
print("5 Historical         :", rl.var_hist(r) * POS)
print(t.summary())

# ---- (f) regimes
calm, stress = r[:brk["break_index"]], r[brk["break_index"]:]
print("\n(f) calm sd:", calm.std(ddof=1), " stress sd:", stress.std(ddof=1))
print("mixture excess kurtosis:",
      rl.mixture_excess_kurtosis([calm.std(ddof=1), stress.std(ddof=1)], [len(calm), len(stress)]))

# ---- (g) noise in the EW VaR
for lam, s in [(0.97, sd_97), (0.94, sd_94)]:
    n_eff = rl.ew_diagnostics(n, lam)["n_eff"]
    print(f"(g) lambda {lam}: SE of VaR = {rl.ew_var_standard_error(s, n_eff, position=POS):,.0f}")
print("gap between the EW VaRs:", (rl.var_normal(sd_94) - rl.var_normal(sd_97)) * POS)
print("VaR in the stress regime:", rl.var_normal(stress.std(ddof=1)) * POS,
      " calm regime:", rl.var_normal(calm.std(ddof=1)) * POS)
