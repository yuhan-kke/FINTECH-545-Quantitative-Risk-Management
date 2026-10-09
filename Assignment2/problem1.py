# Problem 1 - Correlations from mismatched histories
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "risklib" / "src"))   # use risklib without pip install
import numpy as np
import pandas as pd
import risklib as rl

df = pd.read_csv(HERE / "data" / "problem1.csv").set_index("Day")
w = np.array([-0.4, -0.3, -0.2, -0.1, 1.0])   # long 1 IDX, short the basket

# ---- Predict
print("Jointly observed days:\n", rl.joint_counts(df))
print("Days all five trade:", len(df.dropna()))

# ---- (d) complete case vs pairwise
R_cc = rl.missing_cov(df, skip_missing=True, corr=True)
R_pw = rl.missing_cov(df, skip_missing=False, corr=True)
for name, R in [("Complete case", R_cc), ("Pairwise", R_pw)]:
    print(f"\n{name}:\n", R.round(4))
    print("eigenvalues:", np.linalg.eigvalsh(R).round(6))
    print("Cholesky succeeds:", rl.try_cholesky(R)[0])

# ---- (e) covariance from pairwise corr + full-history std
sd = df.std().values
print("\nfull-history std:", sd.round(6))
print("tracking variance (pairwise):", rl.portfolio_variance(w, rl.cor2cov(R_pw, sd)))
lamC, VC = np.linalg.eigh(rl.cor2cov(R_pw, sd))
print("cos(tracking weights, negative eigenvector):", abs(VC[:, 0] @ w) / np.linalg.norm(w))

# ---- (f) repairs
R_rj = rl.near_psd(R_pw)
R_hi = rl.higham_psd(R_pw)
print("\nRebonato-Jaeckel:\n", R_rj.round(4))
print("Higham:\n", R_hi.round(4))
for name, R in [("Rebonato-Jaeckel", R_rj), ("Higham", R_hi)]:
    print(f"\n{name}: min eig = {rl.min_eigenvalue(R):.3e}, "
          f"Frobenius = {rl.frobenius(R, R_pw):.6f}, "
          f"tracking var = {rl.portfolio_variance(w, rl.cor2cov(R, sd)):.4e}")

# ---- Reconcile
print("\n(g) realised var(IDX - basket) on complete rows:",
      (df.dropna()["IDX"] - df.dropna()[["A", "B", "C", "D"]] @ -w[:4]).var())
print("\n(h) Higham change by entry:\n", (pd.DataFrame(R_hi, df.columns, df.columns) - R_pw).round(4))
print("\n(i) ||Higham - pairwise||  =", round(rl.frobenius(R_hi, R_pw), 4))
print("    ||complete - pairwise|| =", round(rl.frobenius(R_cc, R_pw), 4))
print("    tracking var with complete-case corr:", rl.portfolio_variance(w, rl.cor2cov(R_cc, sd)))
lam, V = np.linalg.eigh(R_pw)
print("    eigenvector of the negative eigenvalue (A,B,C,D,IDX):", V[:, 0].round(4))
