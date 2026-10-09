# Assignment 2: Covariance, VaR, and Copulas

Every number in `answers.pdf` is produced by the scripts in this folder.
Each script calls the `risklib` package directly. All the statistics live in the library.

## Setup (once)

1. **Extract the zip first.** Don't run the scripts from inside the zip preview.
2. Keep `risklib/` and `Assignment2/` side by side in the same folder. The scripts find the library there automatically, so no `pip install` is needed.
3. Install the dependencies:

```bash
pip install numpy pandas scipy matplotlib statsmodels
```

Optionally, `pip install -e risklib` installs the library so you can `import risklib` from anywhere.

Tested with Python 3.11, numpy 2.5, scipy 1.18, pandas 3.0, statsmodels 0.15.

## Reproduce everything

```bash
python Assignment2/run_all.py          # runs problems 1-5, ~10 s
python Assignment2/problem3.py         # or any single problem
```

The scripts work from any working directory, including VS Code's Run button.

`run_all.py` prints every result and writes:

- `output/problem1.txt` … `output/problem5.txt`, the full numeric output of each problem;
- `figures/*.png`, every plot used in the write-up.

To run the library tests, use `cd ../risklib && pytest`.

## Files

| File | Contents |
|---|---|
| `problem1.py` … `problem5.py` | One short script per problem. Each one loads its CSV and calls `risklib` functions directly. |
| `run_all.py` | Runs all five scripts and saves their output to `output/` |
| `answers.md` / `answers.pdf` | Written responses |
| `data/` | The five CSVs |

Main `risklib` functions used, by problem:

| Problem | Functions |
|---|---|
| 1 | `joint_counts`, `missing_cov`, `try_cholesky`, `cor2cov`, `portfolio_variance`, `near_psd`, `higham_psd`, `min_eigenvalue`, `frobenius` |
| 2 | `return_calculate`, `variance_breakpoint`, `ew_diagnostics`, `first4_moments`, `ew_std`, `fit_t`, `var_normal`, `var_t`, `var_hist`, `mixture_excess_kurtosis`, `ew_var_standard_error` |
| 3 | `moments_table`, `var_es_table`, `normal_var_es_table`, `subadditivity` |
| 4 | `moments_table`, `drop_most_extreme`, `rank_uniforms`, `joint_tail_counts`, `fit_margins`, `margin_uniforms`, `kendall_matrix`, `kendall_to_pearson`, `fit_gaussian_copula`, `fit_t_copula`, `compare_copulas`, `simulate_returns`, `var_es_table`, `outer_tail_mask`, `t_tail_dependence` |
| 5 | `return_calculate`, `fit_factor_model`, `factor_var_table`, `delta_normal_var` |

## Conventions

These choices are documented in the docstrings of the library functions that make them.

- VaR and ES are positive for a loss. Historical and simulated VaR use numpy's linear-interpolated quantile, and ES is the mean of outcomes at or below that quantile.
- Standard deviations and covariances use ddof = 1. Normal and t MLE fits use the MLE scale.
- EW weights are $(1-\lambda)\lambda^{i-1}$, normalised over the sample. Problem 2 removes the full-sample mean before anything else, as the problem asks.
- The t distribution is location-scale with ν > 2, fitted by MLE.
- Copulas use $R = \sin(\pi\tau/2)$ from Kendall's τ, held fixed. Only the t copula's ν is fitted by likelihood. Parameter counts are 3 (Gaussian) and 4 (t).
- Problem 5 simulates with zero expected returns, so α and the market mean are dropped.
- All simulations are seeded and reproduce exactly.

To regenerate the PDF:

```bash
pandoc answers.md -o answers.pdf --pdf-engine=xelatex
```
