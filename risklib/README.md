# risklib

A small quantitative risk management library for the FINTECH 545 assignments.

## Install

```bash
pip install -e .          # from this folder
pip install -e ".[dev]"   # plus pytest
pytest                    # 24 tests
```

## What's in it

| Module | Functions |
|---|---|
| `moments` | `first4_moments`, `moments_table`, `drop_most_extreme`, `mixture_excess_kurtosis`, `tail_exceedances` |
| `fitting` | `fit_normal`, `fit_t`, `ols`, `fit_regression_normal`, `fit_regression_t`, `aic`, `aicc`, `bic`, `compare_fits` |
| `covariance` | `pearson`, `spearman`, `joint_counts`, `missing_cov` (complete-case or pairwise), `ew_diagnostics`, `ew_std`, `ew_cov`, `ew_cov_mixed`, `cov2cor`, `cor2cov` |
| `psd` | `is_psd`, `min_eigenvalue`, `chol_psd` (works on singular PSD), `try_cholesky`, `near_psd` (Rebonato–Jäckel), `higham_psd` (Higham, optional weights), `frobenius` |
| `mvn` | `conditional_normal`, `band_coverage`, `simulate_normal`, `simulate_pca` |
| `timeseries` | `fit_arma`, `select_arma` (AICc ranking), `model_aicc`, `acf`, `pacf`, `variance_breakpoint` |
| `returns` | `return_calculate` (arithmetic / log / diff) |
| `risk` | `var_hist`, `es_hist`, `var_normal`, `es_normal`, `var_t`, `es_t`, `delta_normal_var`, `portfolio_variance`, `var_es_table`, `normal_var_es_table`, `subadditivity`, `ew_var_standard_error` |
| `copula` | `rank_uniforms`, `joint_tail_counts`, `fit_margins`, `margin_uniforms`, `kendall_to_pearson`, `fit_gaussian_copula`, `fit_t_copula`, `compare_copulas`, `simulate_returns`, `outer_tail_mask`, `t_tail_dependence` |
| `factor` | `fit_factor_model`, `factor_var_table` |
| `plots` | `plot_series`, `plot_hists`, `plot_pairs`, `plot_bars` |

Every fit returns a `FitResult` with `.params`, `.loglik`, `.aic`, `.aicc`, `.bic`, `.ppf(p)` and `.summary()`.
`near_psd` and `higham_psd` accept either a correlation or a covariance matrix. For a covariance they keep the variances unchanged.

## Assignment 1 in a few lines

```python
import pandas as pd, risklib as rl

# Q1 moments and normal tails
x = pd.read_csv("problem1.csv")["x"]
print(rl.first4_moments(x)); print(rl.tail_exceedances(x))

# Q2 regression, Normal vs t errors
d = pd.read_csv("problem2.csv")
fn, ft = rl.fit_regression_normal(d.y, d.x), rl.fit_regression_t(d.y, d.x)
print(rl.compare_fits(fn, ft)); print(fn.ppf(0.995), ft.ppf(0.995))

# Q3 correlations
d3 = pd.read_csv("problem3.csv"); print(rl.pearson(d3)); print(rl.spearman(d3))

# Q4 conditional normal x2 | x1
d4 = pd.read_csv("problem4.csv")
mean, cov, beta = rl.conditional_normal(d4.mean(), d4.cov(), [0], d4.x1.values)
print(beta, cov, rl.band_coverage(d4.x2, mean, cov[0, 0] ** 0.5))

# Q5 AR/MA selection
print(rl.select_arma(pd.read_csv("problem5.csv")["x"]).drop(columns="result"))
```

`examples/problem1_missing_psd.py` works through the missing-data / PSD problem.

## Notes

- `select_arma` fits every model, AR included, by exact MLE with `ARIMA`. All models share the same sample and likelihood, and k counts sigma², so the AICc values are comparable.
  The notebook fit AR with `AutoReg` and MA with `ARIMA`, so it used different nobs and k definitions. Expect small differences from the notebook's numbers.
- `fit_t` and `fit_regression_t` constrain ν > 2 so the variance stays finite.
