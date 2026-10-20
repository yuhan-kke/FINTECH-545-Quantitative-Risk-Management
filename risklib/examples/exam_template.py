# Exam-style template: one block per question type seen on the past midterms.
# Every code snippet in GUIDE.md is copied from this file and runs as is.
# Swap the CSV names for the real ones; delete the blocks you don't need.
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))          # risklib without pip install
import numpy as np
import pandas as pd
import risklib as rl

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", "{:.6f}".format)
DATA = HERE / "data"

# =====================================================================
# A. One series: moments -> fit normal & t -> VaR/ES (fitted + historical)
# =====================================================================
x = pd.read_csv(DATA / "single_series.csv")["X"]

print("A.1 moments\n", rl.first4_moments(x).to_series())

fn, ft = rl.fit_normal(x), rl.fit_t(x)
print("A.2 fits\n", fn.summary(), "\n", ft.summary())
print(rl.compare_fits(fn, ft))                         # lower AICc = better

tab = rl.fitted_var_es_table({"Normal": fn, "t": ft}, alphas=(0.05, 0.01))
h = rl.var_es_table({"Historical": x.values}, alphas=(0.05, 0.01))
print("A.3 VaR/ES (distance from 0, positive = loss)\n", pd.concat([tab, h]))
print("   VaR relative to the mean:\n",
      rl.fitted_var_es_table({"Normal": fn, "t": ft}, (0.05,), relative=True))

# =====================================================================
# B. EW correlation + EW variance, combined
# =====================================================================
r = pd.read_csv(DATA / "returns_5.csv")

R_ew = rl.ew_cov(r, lam=0.97, corr=True)               # EW correlation
v_ew = rl.ew_var(r, lam=0.94)                          # EW variances
S_ew = rl.ew_cov_mixed(r, lam_var=0.94, lam_corr=0.97) # D R D
print("\nB.1 EW corr (0.97)\n", R_ew, "\nB.2 EW var (0.94)\n", v_ew, "\nB.3 combined cov\n", S_ew)
print(rl.ew_diagnostics(len(r), 0.97), rl.ew_diagnostics(len(r), 0.94))

# =====================================================================
# C. A given (non-PSD) matrix: eigenvalues -> Higham & Rebonato-Jaeckel
# =====================================================================
A = pd.DataFrame([[1.0, 0.9, 0.7], [0.9, 1.0, 0.95], [0.7, 0.95, 1.0]])
A.iloc[0, 2] = A.iloc[2, 0] = 0.2                      # break PSD on purpose
# A = pd.read_csv(DATA / "matrix.csv")                 # real exam: a matrix CSV

print("\nC.1 eigenvalues", np.linalg.eigvalsh(A), rl.psd_status(A))
H, RJ = rl.higham_psd(A), rl.near_psd(A)
for name, M in [("Higham", H), ("Rebonato-Jaeckel", RJ)]:
    print(f"C.2 {name}: eig {np.linalg.eigvalsh(M).round(6)}  "
          f"||A-M||_F = {rl.frobenius(A, M):.6f}  {rl.psd_status(M)}")

# =====================================================================
# D. Raw data with blanks: pairwise cov -> status -> Higham -> PCA
# =====================================================================
m = pd.read_csv(DATA / "missing_5.csv")
print("\nD.0 jointly observed rows\n", rl.joint_counts(m))
S_pw = rl.missing_cov(m, skip_missing=False)           # pairwise covariance
print("D.1 pairwise cov\n", S_pw)
print("D.2 min eig", rl.min_eigenvalue(S_pw), "->", rl.psd_status(S_pw))
S_fix = rl.higham_psd(S_pw) if rl.psd_status(S_pw) == "Non-Definite" else S_pw.values
print("D.3 fixed: min eig", rl.min_eigenvalue(S_fix), rl.psd_status(S_fix))
print("D.4 PCA\n", rl.pca_explained(S_fix))

# =====================================================================
# E. Portfolio from prices: shares -> returns -> demean -> fits ->
#    Gaussian copula VaR/ES per asset & portfolio + historical
# =====================================================================
prices = pd.read_csv(DATA / "prices_5.csv")
hold = rl.equal_weight_shares(prices, 100_000)         # $100k split equally
# hold = holdings from fixed shares: rl.holdings_value(prices, 100)
print("\nE.1 holdings\n", hold)

ret = rl.return_calculate(prices, "arithmetic", "Date").drop(columns="Date")
print("E.2 returns head/tail\n", ret.head(), "\n", ret.tail())
ret = ret - ret.mean()                                 # demean if asked

margins_t = {c: rl.fit_t(ret[c]) for c in ret}
margins_n = {c: rl.fit_normal(ret[c]) for c in ret}
print("E.3 t fits\n", pd.DataFrame({c: f.params for c, f in margins_t.items()}).T)

dollars = hold["dollars"].values
cop, R_cop, _ = rl.copula_var_es(ret, dollars, alphas=(0.01,), margin="t",
                                 method="spearman", n_sims=100_000, seed=545)
cop_n, _, _ = rl.copula_var_es(ret, dollars, alphas=(0.01,), margin="normal",
                               method="spearman", n_sims=100_000, seed=545)
hist = rl.hist_var_es(ret, dollars, alphas=(0.01,))
print("E.4 copula correlation (Spearman)\n", R_cop)
print("E.5 $ VaR/ES at 1%\n",
      pd.concat({"t copula": cop, "normal copula": cop_n, "historical": hist}, axis=1))
