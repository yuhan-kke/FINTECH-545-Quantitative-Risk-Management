"""Problem 1 (missing data, PSD failure and repair) using risklib.
Usage: python examples/problem1_missing_psd.py path/to/problem1.csv"""
import sys
import numpy as np
import pandas as pd
import risklib as rl

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "problem1.csv").set_index("Day")
cols = df.columns

print("Joint observation counts:\n", rl.joint_counts(df), "\nAll five trade:", len(df.dropna()))

R_cc = rl.missing_cov(df, skip_missing=True, corr=True)
R_pw = rl.missing_cov(df, skip_missing=False, corr=True)
for name, R in [("complete-case", R_cc), ("pairwise", R_pw)]:
    print(f"\n{name}:\n{R.round(4)}\neigenvalues: {np.linalg.eigvalsh(R)}")
    print("Cholesky:", "ok" if rl.try_cholesky(R)[0] else "FAILED (not PD)")

sd = df.std().values
w = np.array([-.4, -.3, -.2, -.1, 1.0])          # long IDX, short basket
tvar = lambda R: w @ rl.cor2cov(R, sd) @ w
print(f"\n(e) tracking variance, pairwise: {tvar(R_pw.values):.4e}")

for name, fix in [("Rebonato-Jaeckel", rl.near_psd), ("Higham", rl.higham_psd)]:
    R = fix(R_pw.values)
    print(f"(f) {name:16s} min eig={rl.min_eigenvalue(R):.2e}  "
          f"Frobenius={rl.frobenius(R, R_pw):.6f}  tracking var={tvar(R):.4e}")
