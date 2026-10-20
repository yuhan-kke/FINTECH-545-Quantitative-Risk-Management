from pathlib import Path
import sys

from scipy.special import gammaln

from scipy.special import gammaln
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "risklib" / "src"))   # use risklib without pip install
import numpy as np
import pandas as pd
from scipy import optimize, stats
import risklib as rl
pd.set_option("display.float_format", "{:.6g}".format)
pd.set_option("display.width", 160)


# -----------Question 1 ------------
print("Question 1 starts here.")

#1(a)----- Calculate the pairwise missing correlation
df1 = pd.read_csv(HERE / "data" / "problem1.csv")
print("shape", df1.shape)
df1_pw = rl.missing_corr(df1, skip_missing=False)
print(df1_pw)


#1(b)----- Calculate the variance of each series
df1_var = df1.var()
print("(b) variance per column")
print(df1_var)

#1(c)-----Is the correlation matrix positive semi-definite?
eigvals = np.linalg.eigvals(df1_pw)
print("(c) Eigenvalues of the correlation matrix:", eigvals)
is_psd = np.all(np.linalg.eigvals(df1_pw) >= 0)
print("(c) Is the correlation matrix positive semi-definite?", is_psd)

#1(d)-----Use higham's method to find the nearest positive semi-definite matrix
df1_corr_psd = rl.higham_psd(df1_pw)
print("(d) Nearest positive semi-definite correlation matrix:")
print(df1_corr_psd)
eigvals_psd = np.linalg.eigvals(df1_corr_psd)
print("(d) Eigenvalues of the nearest positive semi-definite correlation matrix:", eigvals_psd)

#which entry changed the most?
diff_matrix = np.abs(df1_corr_psd - df1_pw)
max_change = np.max(diff_matrix)
max_change_index = np.unravel_index(np.argmax(diff_matrix), diff_matrix.shape)
print(diff_matrix)
print("(d) Maximum change in the correlation matrix:", max_change)
print("(d) Location of the maximum change (row, column):", max_change_index)


#1(e)-----Combine the repaired correlation matrix with the original variances to get the covariance matrix
df1_sd_full = df1.std()                                    # each column's own full history sd
df1_cov_fix = pd.DataFrame(rl.cor2cov(df1_corr_psd, df1_sd_full.values), df1.columns, df1.columns)
print("(e) Repaired covariance matrix:")
print(df1_cov_fix)

#-------Question 2 starts here
#-------2(a) and 2(b) combined
print("Question 2 starts here.")    

df2 = pd.read_csv(HERE / "data" / "problem2.csv")
print("shape", df2.shape)
ret = rl.return_calculate(df2, "arithmetic", "Date").drop(columns="Date")
spy, xyz = ret["SPY"].values, ret["XYZ"].values

# fit r_XYZ = alpha + beta * r_SPY + e
reg = rl.fit_regression_normal(xyz, spy)
alpha, beta = reg.params["alpha"], reg.params["beta"]
e = reg.residuals
sigma_e = e.std(ddof=1)
print(reg.summary())
print(f"alpha {alpha:.6f}  beta {beta:.4f}  sigma_e {sigma_e:.6f}  R2 {rl.ols(xyz, spy)['r2']:.4f}")

# SPY ~ N(mu_spy, sd_spy), e ~ N(0, sigma_e), independent (OLS)
mu_spy, sd_spy = spy.mean(), spy.std(ddof=1)
print(f"SPY: mu {mu_spy:.6f}  sd {sd_spy:.6f}")
print("SPY normality: skew", round(stats.skew(spy), 4), " excess kurt", round(stats.kurtosis(spy), 4))
print("e   normality: skew", round(stats.skew(e), 4), " excess kurt", round(stats.kurtosis(e), 4))

# position: 100 shares of XYZ
shares = 100
S0 = df2["XYZ"].iloc[-1]
PV = shares * S0
print(f"position: {shares} x {S0:.4f} = ${PV:,.2f}")

# closed form: r_XYZ ~ N(alpha + beta*mu_spy, beta^2*sd_spy^2 + sigma_e^2)
mu_x = alpha + beta * mu_spy
sd_x = np.sqrt(beta ** 2 * sd_spy ** 2 + sigma_e ** 2)

# simulation
rng = np.random.default_rng(545)
n = 100_000
r_spy = rng.normal(mu_spy, sd_spy, n)
eps = rng.normal(0.0, sigma_e, n)
r_xyz = alpha + beta * r_spy + eps
pnl_sim = PV * r_xyz
pnl_hist = PV * xyz

#report VaR and ES for different confidence levels in percent
rows = {}
for a in (0.05, 0.01):
    rows[f"{a:.0%}"] = {
        "VaR delta normal %": 100 * rl.var_normal(sd_x, mu_x, a),
        "ES delta normal %": 100 * rl.es_normal(sd_x, mu_x, a),
        "VaR historical %": 100 * rl.var_hist(xyz, a),
        "ES historical %": 100 * rl.es_hist(xyz, a),
        "VaR simulated %": 100 * rl.var_hist(r_xyz, a),
        "ES simulated %": 100 * rl.es_hist(r_xyz, a)

    }
print(pd.DataFrame(rows).round(4))


#------ End of Question 2 ------

#------problem 3 starts here
print("Problem 3 starts here.")

prices = pd.read_csv(HERE / "data" / "problem3.csv")
hold = pd.read_csv(HERE / "data" / "problem3_holdings.csv")
ret = prices.drop(columns="Date").pct_change().dropna()

prices = pd.read_csv(HERE / "data" / "problem3.csv")
hold = pd.read_csv(HERE / "data" / "problem3_holdings.csv")
ret = prices.drop(columns="Date").pct_change().dropna()
ret = ret[hold["Stock"]]                    # same order as the holdings

# =============================== 3(a) ===============================
print("--- problem 3(a) ---")
def aicc(ll, k, n):
    return 2 * k - 2 * ll + (2 * k * k + 2 * k) / (n - k - 1)

def bic(ll, k, n):
    return k * np.log(n) - 2 * ll

rows = []
for stock in ret.columns:
    x = ret[stock].to_numpy()
    n = len(x)
    pn = stats.norm.fit(x)
    pt = stats.t.fit(x)
    pg = stats.norminvgauss.fit(x)
    a, b, loc, sc = pg
    ll_n = stats.norm.logpdf(x, *pn).sum()
    ll_t = stats.t.logpdf(x, *pt).sum()
    ll_g = stats.norminvgauss.logpdf(x, *pg).sum()
    rows.append({"Stock": stock,
                 "Normal mu": pn[0], "Normal sigma": pn[1],
                 "t nu": pt[0], "t mu": pt[1], "t sigma": pt[2],
                 "NIG alpha": a / sc, "NIG beta": b / sc, "NIG delta": sc, "NIG mu": loc,
                 "loglik normal": ll_n, "loglik t": ll_t, "loglik NIG": ll_g,
                 "AICc normal": aicc(ll_n, 2, n), "AICc t": aicc(ll_t, 3, n), "AICc NIG": aicc(ll_g, 4, n),
                 "BIC normal": bic(ll_n, 2, n), "BIC t": bic(ll_t, 3, n), "BIC NIG": bic(ll_g, 4, n),
                 "skew": stats.skew(x), "excess kurt": stats.kurtosis(x)})
res_a = pd.DataFrame(rows).set_index("Stock")
groups = {
    "Normal":   ["Normal mu", "Normal sigma"],
    "Student t": ["t nu", "t mu", "t sigma"],
    "NIG":      ["NIG alpha", "NIG beta", "NIG delta", "NIG mu"],
    "Shape":    ["skew", "excess kurt"],
    "loglik":   ["loglik normal", "loglik t", "loglik NIG"],
    "AICc":     ["AICc normal", "AICc t", "AICc NIG"],
    "BIC":      ["BIC normal", "BIC t", "BIC NIG"],
}
big = pd.concat({g: res_a[cols] for g, cols in groups.items()}, axis=1)
big.columns = pd.MultiIndex.from_tuples(
    [(g, c.replace("Normal ", "").replace("t ", "").replace("NIG ", "")
              .replace("loglik ", "").replace("AICc ", "").replace("BIC ", ""))
     for g, c in big.columns])

pd.set_option("display.width", 400)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.6g}".format)
print(big.to_string())

# =============================== 3(b) ===============================
print("\n--- problem 3(b) ---")
# YOUR part (a) choice for each stock: "normal", "t" or "nig"
choice = {"S1": "normal", "S2": "t", "S3": "nig", "S4": "normal", "S5": "t",
          "S6": "nig", "S7": "normal", "S8": "t", "S9": "nig", "S10": "t"}

class NIGMargin:                            # scipy NIG with a fast, robust ppf
    def __init__(self, x):
        self.d = stats.norminvgauss(*stats.norminvgauss.fit(x))
        lo, hi = x.min() - 10 * x.std(), x.max() + 10 * x.std()
        g = np.linspace(lo, hi, 20001)
        f = self.d.pdf(g)
        c = np.concatenate([[0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(g))])
        self.g, self.c = g, c / c[-1]
    def cdf(self, v):
        return np.interp(v, self.g, self.c)
    def ppf(self, q):
        return np.interp(q, self.c, self.g)

class Margin:                               # lets rl functions read .dist
    def __init__(self, dist):
        self.dist = dist

margins = []
for c in ret.columns:
    x = ret[c].values
    if choice[c] == "normal":
        margins.append(rl.fit_normal(x))
    elif choice[c] == "t":
        margins.append(rl.fit_t(x))
    else:
        margins.append(Margin(NIGMargin(x)))
U = np.clip(rl.margin_uniforms(ret, margins), 1e-12, 1 - 1e-12)

tab_b, g_fit, t_fit = rl.choose_copula(U)   # same Kendall R; k = 0 / 1; 2 < nu <= 100
print(tab_b)
print("fitted t-copula nu:", round(t_fit.nu, 4))
print("min eigenvalue of R:", rl.min_eigenvalue(t_fit.R))
print("R (Kendall):\n", pd.DataFrame(t_fit.R, ret.columns, ret.columns).round(4))

# ===============================3 (c) ===============================
print("\n--- problem 3(c) ---")
COPULA = "gaussian"                      

last = prices.iloc[-1].drop("Date").astype(float)
hold["price"] = hold["Stock"].map(last)
hold["value"] = hold["Holding"] * hold["price"]
hold["weight"] = hold["value"] / hold["value"].sum()
books = hold.groupby("Book")["Stock"].apply(list).to_dict()
dollars = hold["value"].values
print(hold)

if COPULA == "gaussian":
    cop_fit = rl.fit_gaussian_copula(U, rl.copula_corr(U, "spearman"))
else:
    cop_fit = t_fit
print("min eig of simulation R:", round(rl.min_eigenvalue(cop_fit.R), 4))

N = 100_000                                 # >= 5,000 draws
_, sim = rl.simulate_returns(cop_fit, margins, N, seed=545)
pnl = pd.DataFrame(sim * dollars, columns=ret.columns)          # $ P&L per stock

alpha = 0.05                                # 95% level
stock_tab = rl.var_es_table({c: pnl[c].values for c in pnl}, (alpha,))
book_tab = rl.group_var_es(pnl, books, (alpha,))
table = pd.concat([stock_tab, book_tab])
h = hold.set_index("Stock")
table.insert(0, "Book", [h["Book"].get(i, "") for i in table.index])
table.insert(1, "Value $", [h["value"].get(i, np.nan) for i in table.index])
table.loc["A", "Value $"] = hold.loc[hold.Book == "A", "value"].sum()
table.loc["B", "Value $"] = hold.loc[hold.Book == "B", "value"].sum()
table.loc["Total", "Value $"] = hold["value"].sum()
table["VaR % of value"] = 100 * table["VaR 5%"] / table["Value $"]
table["ES % of value"] = 100 * table["ES 5%"] / table["Value $"]
pd.set_option("display.float_format", "{:,.2f}".format)
print(table)
print("sum of stock VaRs:", round(stock_tab["VaR 5%"].sum(), 2),
      "  sum of stock ES:", round(stock_tab["ES 5%"].sum(), 2))