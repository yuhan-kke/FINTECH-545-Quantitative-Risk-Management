# Problem 5 - Model based simulation and residual correlation
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "risklib" / "src"))   # use risklib without pip install
import numpy as np
import pandas as pd
import risklib as rl

prices = pd.read_csv(HERE / "data" / "problem5.csv")
ret = rl.return_calculate(prices, "arithmetic", "Day").drop(columns="Day")
portfolios = {"P1": np.array([1e6, 1e6]),       # long A, long B
              "P2": np.array([1e6, -1e6])}      # long A, short B

# ---- (b) regressions on the market
fm = rl.fit_factor_model(ret, "MKT")
print("(b)\n", fm.summary())
print("residual correlation:", round(fm.resid_corr[0, 1], 4))

# ---- (c) simulated VaR with full vs independent residuals
print("\n(c)\n", rl.factor_var_table(fm, portfolios, 100_000, seed=545).round(0))

# ---- (d) delta normal from the sample covariance
S = ret[["A", "B"]].cov().values
for p, w in portfolios.items():
    print(f"(d) delta normal VaR {p}: {rl.delta_normal_var(w, S):,.0f}")

# ---- (f) the factor model reproduces the sample covariance exactly
print("\n(f) max |implied cov - sample cov| =", np.abs(fm.implied_cov() - S).max())
for label, diag in [("full", False), ("independent", True)]:
    C = fm.implied_cov(diagonal_resid=diag)
    print(f"closed-form VaR, {label} residuals:",
          {p: round(rl.delta_normal_var(w, C)) for p, w in portfolios.items()})
