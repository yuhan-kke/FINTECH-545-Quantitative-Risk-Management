# Using risklib for FINTECH 545 exam questions

This guide is built from the two past midterms (Fall 2025 and 2026-02-21) and the Week 01–05 lecture notes. Week 06 (options) is not on the exam. risklib's `options` module is extra and not needed. It covers which function answers which question, the conventions that decide whether your number matches the key, and the mistakes that came up most often while practising.

> **Exam rules.** Both past midterms say: notes and the internet for *coding syntax only*, no LLMs, no coding assistants, no working with other students. Use this library and guide to **prepare**. Check with the instructor whether you may bring your own library code into the exam.

Everything below is copied from [`examples/exam_template.py`](examples/exam_template.py), which runs end to end on the sample CSVs in `examples/data/`. Copy the template, change the file names, and delete the blocks you don't need.

---

## 0. Setup: one header for every script

```python
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "risklib" / "src"))   # adjust to where risklib/ sits
import numpy as np, pandas as pd
import risklib as rl
pd.set_option("display.float_format", "{:.6f}".format)      # stop 1e-05 notation
pd.set_option("display.max_columns", 20)
```

Alternatively, run `pip install -e risklib` once, then use a plain `import risklib as rl`.

If you get `ModuleNotFoundError: risklib`, you are running from inside a zip or the path is wrong. Extract the zip and run `print(sys.path[0])` to see where the script is looking.

---

## 1. Conventions cheat sheet (read before every exam)

| Topic | risklib convention | Common mistake |
|---|---|---|
| **α** | Tail probability: `alpha=0.05` for 5% VaR, `0.01` for 1% | Passing `0.95` |
| **Sign** | VaR and ES are **positive numbers = losses** | Reporting the negative quantile |
| **"Distance from 0"** | Default: `VaR = -quantile(α)` | Use `relative=True` only if the question says *relative to the mean* |
| **Fitted vs sample** | `var_normal(sigma, mu, alpha)` takes **parameters** | Passing the DataFrame: `var_normal(df, 0.95)` ✗ |
| **Dollars** | $ VaR = holding value × return VaR (or compute VaR on the $ P&L directly) | Using shares instead of $ |
| **Row order** | Oldest row first; the last row gets the largest EW weight | Data sorted newest first, so the weights are reversed |
| **EW mean** | `ew_cov(..., mean="simple")` (default) removes the ordinary sample mean, which matches the course `ewCovar` | — |
| **λ** | λ is the *decay*: 0.97 means slower decay and a longer memory | Swapping the two λs. `ew_cov_mixed` forces you to name `lam_var=` and `lam_corr=` |
| **Covariance ddof** | `ddof=1` (pandas default) | — |
| **Fits** | MLE. t is location-scale `(mu, sigma, nu)`; sd = σ√(ν/(ν−2)) | Reading σ as the standard deviation |
| **Historical VaR** | numpy linear-interpolated quantile; ES = mean of outcomes ≤ that quantile | — |
| **Copula correlation** | `copula_corr(U, "spearman")` is the course convention; `"kendall"` gives sin(πτ/2) | Mixing methods between parts |
| **Demeaning** | Do it **only when asked**: `ret = ret - ret.mean()` | Demeaning twice, or forgetting the step |

---

## 2. Recipes by question type

### A. One series: moments, choosing a distribution, VaR/ES
*(Fall 2025 Q2–Q3, 2026 Q1)*

```python
x = pd.read_csv("problem1.csv")["X"]          # a Series, not the whole DataFrame
print(rl.first4_moments(x).to_series())        # mean, var, sd, skew, EXCESS kurtosis

fn, ft = rl.fit_normal(x), rl.fit_t(x)
print(fn.summary(), ft.summary())
print(rl.compare_fits(fn, ft))                 # lower AICc wins

tab = rl.fitted_var_es_table({"Normal": fn, "t": ft}, alphas=(0.05, 0.01))
h   = rl.var_es_table({"Historical": x.values}, alphas=(0.05, 0.01))
print(pd.concat([tab, h]))
```

How to justify the choice of distribution:

- Excess kurtosis well above 0 suggests the t. Also look at the AICc difference and at the fitted ν: a small ν means fat tails, while ν > 30 or so means the t is close to normal.
- At **5%** the t VaR can be *lower* than the normal VaR, because a fitted t has a sharper peak and thinner shoulders. Fat tails dominate only further out, roughly below 2.5–3.5%. At **1%**, the t VaR and ES are usually higher.
- ES grows relative to VaR much more under the t than under the normal.

### B. EW correlation and EW variance, combined
*(Fall 2025 Q4: corr λ=0.94, var λ=0.97. 2026 Q2: corr λ=0.97, var λ=0.94. The λs are swapped between years, so read the question.)*

```python
r = pd.read_csv("problem2.csv")                # returns, oldest first
R_ew = rl.ew_cov(r, lam=0.97, corr=True)       # EW correlation matrix
v_ew = rl.ew_var(r, lam=0.94)                  # EW variances (a vector)
S    = rl.ew_cov_mixed(r, lam_var=0.94, lam_corr=0.97)   # Sigma = D R D
print(rl.ew_diagnostics(len(r), 0.97))         # n_eff, half life
```

- If the file has a date column, drop it first: `r = r.drop(columns="Date")`. If it holds prices, use `return_calculate` before anything else.
- **Why combine?** Variances react quickly to new information, so they get a smaller λ. Correlations need more data to estimate, so they get a larger λ and a longer memory. Compare the half-life ln0.5/lnλ: 11.2 days for 0.94 and 22.8 days for 0.97.

### C. A given matrix: eigenvalues, Higham, Rebonato–Jäckel, comparison
*(2026 Q3)*

```python
A = pd.read_csv("problem3.csv")               # must already be square!
print(np.linalg.eigvalsh(A), rl.psd_status(A))
H, RJ = rl.higham_psd(A), rl.near_psd(A)
for name, M in [("Higham", H), ("RJ", RJ)]:
    print(name, np.linalg.eigvalsh(M), rl.frobenius(A, M), rl.psd_status(M))
```

- Both algorithms are defined on the **correlation** matrix (W03 §6). Repair C, then rebuild Σ = D C D with each variable's full-history sd (W03 §4 item 7).
- Before anything else, check `A.shape`. If it's (500, 5), the file is **raw data**, not a matrix: go to recipe D.
- If the CSV has a label column, drop it first: `A = pd.read_csv(..., index_col=0)`.
- Both repairs keep the diagonal (the variances) unchanged. Higham gives the nearest matrix **in correlation space**. Compare there with `rl.frobenius(rl.cov2cor(A)[0], rl.cov2cor(H)[0])`. If A is a covariance, RJ can come out *closer* in covariance space (it did for the released problem1.csv). RJ is a single eigen-clip and is faster; Higham is iterative.
- A repaired matrix is **singular**: its smallest eigenvalue is 0. Don't invert it or condition on it (see the pitfalls). Use the complete-case matrix, or `near_psd(S, eps=1e-8)`, when you need a PD matrix.
- After a repair, the smallest eigenvalue prints as something like −5e-15. That is **zero**: the answer key's −3e-13 is the same answer.

### D. Raw data with blanks: pairwise covariance, PD/PSD check, Higham, PCA
*(Fall 2025 Q5)*

```python
m = pd.read_csv("problem5.csv")                # x1..x5 with blanks
print(rl.joint_counts(m))                      # rows used by each pair
S_pw = rl.missing_cov(m, skip_missing=False)   # PAIRWISE covariance
print(rl.min_eigenvalue(S_pw), rl.psd_status(S_pw))
S_fix = rl.higham_psd(S_pw) if rl.psd_status(S_pw) == "Non-Definite" else S_pw.values
print(rl.psd_status(S_fix))
print(rl.pca_explained(S_fix))                 # eigenvalue, explained, cumulative
```

- `skip_missing=False` gives pairwise estimates and `True` gives complete cases only. Complete-case estimates are always PSD; pairwise estimates may not be.
- For a correlation matrix, use `rl.missing_corr(m, skip_missing=False)`. The function name says "cov", but `missing_cov(..., corr=True)` gives the same result.
- PCA on the **covariance** weights high-variance series more heavily. PCA on the **correlation** treats every series equally. Use whichever matrix the question names.

### E. Portfolio from prices: holdings, returns, fits, Gaussian copula VaR/ES
*(Fall 2025 Q6: 3 stocks × 100 shares, 5%. 2026 Q4: $100k equal-weight, 5 stocks, 1%.)*

```python
prices = pd.read_csv("problem4.csv")                     # Date, SPY, AAPL, ...
hold = rl.equal_weight_shares(prices, 100_000)           # price, dollars, shares
# fixed shares instead:  dollars = rl.holdings_value(prices, 100).values

ret = rl.return_calculate(prices, "arithmetic", "Date").drop(columns="Date")
print(ret.head(), ret.tail())
ret = ret - ret.mean()                                   # only if asked

dollars = hold["dollars"].values
cop, R, margins = rl.copula_var_es(ret, dollars, alphas=(0.01,), margin="t",
                                   method="spearman", n_sims=100_000, seed=545)
hist = rl.hist_var_es(ret, dollars, alphas=(0.01,))
print(R); print(pd.concat({"copula": cop, "historical": hist}, axis=1))
```

- `margin="t"`, `"normal"` or `"best"` (picks by AICc for each stock). Run both `"t"` and `"normal"` when the question asks for the normal and t models.
- `copula_var_es` returns one row per stock plus a `Portfolio` row, in dollars.
- Diversification check: the portfolio VaR should be **less than** the sum of the stock VaRs. For example, the Fall 2025 practice run gave $476 vs $593.
- Copula VaR is a simulation, so it changes slightly with the seed and `n_sims`. Expect about ±1–2% against the key at 100k draws. Historical VaR is exact.
- Watch for a fitted ν in the thousands (e.g. BABA ν≈4300 in the sample data). That margin is effectively normal. Say so in your answer.

### E2. Books / groups of positions and who drives the risk
*(released problem3.csv + holdings)*

```python
hold = pd.read_csv("problem3_holdings.csv")                 # Stock, Book, Holding
last = prices.iloc[-1].drop("Date").astype(float)
hold["value"] = hold["Holding"] * hold["Stock"].map(last)
books = hold.groupby("Book")["Stock"].apply(list).to_dict()
ret = ret[hold["Stock"]]                                    # SAME ORDER as dollars!

pnl, info = rl.copula_simulate_pnl(ret, hold["value"].values, margin="best",
                                   method="spearman", n_sims=100_000, seed=545)
print(rl.group_var_es(pnl, books, (0.05, 0.01)))   # books, Total, Sum, Diversification
print(rl.group_var_es(rl.hist_pnl(ret, hold["value"].values), books, (0.05, 0.01)))
```

### E3. Stock on the market: regression, conditional normal, model-based simulation
*(released problem2.csv; W02 §2.4, §4.3; W05 §2, §7)*

```python
o = rl.ols(r["XYZ"], r["SPY"])                       # alpha, beta, r2
rn, rt = rl.fit_regression_normal(r["XYZ"], r["SPY"]), rl.fit_regression_t(r["XYZ"], r["SPY"])
print(rl.compare_fits(rn, rt))                       # k = 3 vs 4 (W02 §5.3)
S = r[["SPY", "XYZ"]].cov().values                   # XYZ given SPY = -3%:
mean, cov, beta = rl.mvn.conditional_normal(r[["SPY","XYZ"]].mean().values, S, [0], [-0.03])
# W05 §7: SPY ~ normal, eps ~ t, Gaussian copula on [SPY, eps], rebuild XYZ = b*SPY + eps
X = pd.DataFrame({"SPY": r.SPY - r.SPY.mean(), "eps": rt.residuals})
margins = [rl.fit_normal(X.SPY), rl.fit_t(X.eps)]
U = rl.margin_uniforms(X, margins)
_, Xs = rl.simulate_returns(rl.fit_gaussian_copula(U, rl.copula_corr(U)), margins, 100_000, seed=545)
```

When every piece is normal, this gives the same answer as delta normal. "OLS is the same thing as just using a joint, multivariate distribution" (W05 §2.1).

### E4. Choosing the copula the course way (W05 §6.5)

```python
tab, g, t = rl.choose_copula(U)     # same Kendall R for both, k_G = 0, k_t = 1, 2 < nu <= 100
print(tab)                          # take the t only if dBIC = 2(l_t - l_G) - ln m > 0
```

Use `copula_corr(U, "spearman")` for the **Gaussian copula you simulate with** (W05 §4.1). Use Kendall's τ for the **comparison**, so both copulas get the same R. A fitted ν stuck at 100 means "Gaussian".

### E5. Skewed data: the NIG (W01 §5.5, Table 6)

```python
f = rl.fit_nig(x)      # params alpha, beta (sign = skew), delta, mu; k = 4
```

Table 6: skew ≈ 0 with excess kurtosis ≈ 0 → normal; skew ≈ 0 with excess kurtosis > 0 → t; skew ≠ 0 with excess kurtosis > 0 → NIG. A NIG fit can be used as a copula margin, just like a normal or t fit.

### F. Essay questions
*(Fall 2025 Q1: risk vs forecasting models)*

No code is needed. A risk model cares about the **whole distribution, especially the tails** (VaR/ES, extreme co-movement). A forecasting model cares about the **conditional mean** and point accuracy. As a result, a model can forecast well and still be a poor risk model, for example a normal fit to fat-tailed data.

---

## 3. Map of past midterm questions to functions

| Exam | Q | Task | Functions |
|---|---|---|---|
| Fall 2025 | 1 | Risk vs forecasting (essay) | — |
| Fall 2025 | 2 | Moments, choice, fit | `first4_moments`, `fit_normal`, `fit_t`, `compare_fits` |
| Fall 2025 | 3 | VaR/ES 5%, distance from 0 | `fitted_var_es_table` (or `var_normal`, `var_t`, `es_*`) |
| Fall 2025 | 4 | EW corr 0.94, var 0.97, combine | `ew_cov(corr=True)`, `ew_var`, `ew_cov_mixed` |
| Fall 2025 | 5 | Missing data, pairwise, PD/PSD, Higham, PCA | `missing_cov(skip_missing=False)`, `psd_status`, `higham_psd`, `pca_explained` |
| Fall 2025 | 6 | 100 shares each, t margins, Gaussian copula, 5% $ VaR/ES | `holdings_value`, `return_calculate`, `copula_var_es(margin="t")` |
| 2026 | 1 | Moments, fits, 5%/1% VaR/ES, historical | `first4_moments`, `fit_*`, `fitted_var_es_table`, `var_es_table` |
| 2026 | 2 | EW corr 0.97, var 0.94, combine | `ew_cov`, `ew_var`, `ew_cov_mixed` |
| 2026 | 3 | Given matrix: eigen, Higham, RJ | `psd_status`, `higham_psd`, `near_psd`, `frobenius` |
| 2026 | 4 | $100k equal weight, normal and t, copula 1%, historical | `equal_weight_shares`, `return_calculate`, `copula_var_es`, `hist_var_es` |

---

## 4. Pitfalls checklist (each one happened in practice)

- [ ] **Wrong object type.** `fit_*` and `first4_moments` take **one column** (`df["X"]`), not the DataFrame. `fit_regression_t(y, X)` takes **two** arguments.
- [ ] **A 1×1 matrix** means you passed a single column. Correlation and covariance need ≥ 2 columns.
- [ ] **`LinAlgError: last 2 dimensions must be square`** means you called `eigvalsh` on raw data. Build the matrix first with `missing_cov`.
- [ ] **A date column** got into the returns or the covariance. Use `.drop(columns="Date")`, or `index_col=0` for matrix CSVs.
- [ ] **Prices vs returns.** Did the question give prices? Use `return_calculate(prices, "arithmetic", "Date")`, unless it says log returns.
- [ ] **α = 0.95** ✗ → use `0.05`.
- [ ] **Negative VaR** when you expected a loss means the sign convention is wrong. risklib already returns losses as positive numbers.
- [ ] **λs swapped** between the variance and the correlation. Read the question twice.
- [ ] **Demean.** Do it when asked, once, *before* fitting.
- [ ] **$ vs %.** Report the units the question asks for. In a $ table, a return VaR of 0.02 means nothing.
- [ ] **Dollar weights.** Use the latest price × shares, not the first price.
- [ ] **Conditioning on a repaired matrix.** Higham/RJ output is singular, so the conditional variance comes out as ≈0 or negative and its sqrt is `nan`. Condition on a PD matrix instead.
- [ ] **A fitted ν like 1e12** means the t collapsed to the normal. Report "normal", not "t with ν = 2e12".
- [ ] **Holdings order ≠ returns column order.** Reindex the returns: `ret = ret[hold["Stock"]]`.
- [ ] **Historical VaR > copula VaR** with strongly negative skew is real, not a bug: t margins are symmetric.
- [ ] `$` in matplotlib titles must be escaped as `\$`.

---

## 5. Matching the answer key

| You see | Why | Is it fine? |
|---|---|---|
| Eigenvalue −5e-15 vs key −3e-13 | Floating-point noise around 0 | Yes, both are 0 |
| Copula VaR off by 1–2% | Different random draws | Yes. State your seed and n |
| EW numbers differ in the 3rd digit | Weighted vs simple mean | Use the default `mean="simple"` |
| Copula correlation slightly different | Spearman vs Kendall vs Pearson of normal scores | Use `method="spearman"` unless the question names another method |
| t parameters differ in the 4th digit | Optimizer tolerance or starting values | Yes |
| Historical VaR differs a lot | A different quantile definition, or you forgot to demean | Check the demean step; risklib uses the linear quantile |

---

## 6. Faster workflow on the day

1. Copy `examples/exam_template.py` to `midterm.py`. Make one block per question and keep the CSV loads at the top of each block.
2. Run often. Print shapes (`df.shape`) right after every `read_csv`.
3. Put every reported number into a small DataFrame and print it once. This avoids copying errors from scrolling output.
4. Write the interpretation sentence straight after each print, while the numbers are on screen.
