# Problem 3 - When diversification raises VaR
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "risklib" / "src"))   # use risklib without pip install
import pandas as pd
import risklib as rl

d = pd.read_csv(HERE / "data" / "problem3.csv")
pnl = {"$1M A": 1e6 * d["A"],
       "$1M B": 1e6 * d["B"],
       "$2M A": 2e6 * d["A"],
       "$1M A + $1M B": 1e6 * d["A"] + 1e6 * d["B"]}

# ---- Predict
rl.plots.plot_hists({k: pnl[k] / 1e3 for k in ["$2M A", "$1M A + $1M B"]},
                    HERE / "figures" / "p3_pnl.png", xlabel="P&L ($000)")
print("(a)\n", rl.moments_table(d[["A", "B"]]))
lossA, lossB = d["A"] < -0.2, d["B"] < -0.2
print("\n(b) loss > 20%: A", lossA.sum(), " B", lossB.sum(),
      " at least one", (lossA | lossB).sum(), " both", (lossA & lossB).sum())

# ---- (c) (d) (e)
print("\n(c) historical, 5%\n", rl.var_es_table(pnl, [0.05]).round(0))
print("\n(d) normal, 5%\n", rl.normal_var_es_table(pnl, [0.05]).round(0))
print("\n(e) historical, 1%\n", rl.var_es_table(pnl, [0.01]).round(0))

# ---- (f) subadditivity at 5%
for f in (rl.var_hist, rl.es_hist):
    print(f"\n(f) {f.__name__}:",
          rl.subadditivity(f(pnl["$1M A"]), f(pnl["$1M B"]), f(pnl["$1M A + $1M B"])))
print("\ncorrelation of the big-loss indicators:", round(lossA.astype(float).corr(lossB.astype(float)), 4))
