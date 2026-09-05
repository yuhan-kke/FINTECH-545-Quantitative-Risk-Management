# FinTech 545 — Assignment 1

## Contents

```
fintech545_assignment1.ipynb   # the analysis (already executed; outputs are saved in the file)
problem1.csv ... problem5.csv  # data, one file per question
```

All five `problemN.csv` files must sit in the **same directory** as the notebook — every
`pd.read_csv("problemN.csv")` call uses a relative path with no folder prefix.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
python -m ipykernel install --user --name python3 --display-name "Python 3"
```

## Running it

Run every cell top to bottom (the notebook is stateful — later questions rely on variables
defined in earlier ones, e.g. Q2's cell 9 relies on `df2` from cell 7). Either:

- Open `fintech545_assignment1.ipynb` in Jupyter and **Run All**, or
- Reproduce the whole thing headlessly from the command line:

```bash
jupyter nbconvert --to notebook --execute fintech545_assignment1.ipynb \
    --output fintech545_assignment1.ipynb \
    --ExecutePreprocessor.timeout=180 \
    --ExecutePreprocessor.kernel_name=python3
```

This overwrites the notebook in place with freshly computed outputs (numbers + figures)
identical to the ones reported below. Runtime is well under a minute.

## Reproducing every number in the write-up

Every number below comes directly from the printed output of the referenced cell. Cell
numbers refer to the notebook's code-cell order (0-indexed, matching `jupyter nbconvert`'s
own numbering / the executed `.ipynb`'s `execution_count`).

### Question 1 — distribution fit (`problem1.csv`)

| Number | Value | Cell |
|---|---|---|
| n | 1000 | 2 |
| mean | 0.001104 | 2 |
| variance | 0.000096 | 2 |
| std dev | 0.009811 | 2 |
| skewness (bias-corrected) | -0.669837 | 2 |
| excess kurtosis (bias-corrected) | 2.357614 | 2 |
| fitted Normal 1% quantile | -0.021719 | 3 |
| observed count below 1% quantile | 26 (expected 10.0) | 3 |
| left/right tail counts at p = {0.001, 0.005, 0.01, 0.02, 0.05, 0.95, 0.99, 0.995, 0.999} | see printed table | 4 |
| histogram + fitted PDF and QQ-plot figure | `figures/normal_fit_plot.png` | 5 |

### Question 2 — regression under Normal vs. Student-t errors (`problem2.csv`)

| Number | Value | Cell |
|---|---|---|
| scatter plot of y vs x | `figures/y_vs_x_scatter.png` | 8 |
| OLS: alpha, beta, SEs, sigma | 1.5607 (SE 0.0963), 2.9908 (SE 0.0966), 1.3514 | 9 |
| MLE-Normal: alpha, beta, sigma, log-lik, AIC, AICc | 1.5607, 2.9907, 1.3446, -343.0110, 692.0221, 692.1445 | 9 |
| MLE-Student-t: alpha, beta, scale, nu, log-lik, AIC, AICc | 1.5480, 3.0192, 0.9355, 3.6322, -328.5612, 665.1224, 665.3275 | 9 |
| preferred model by AICc | Student-t (665.33 < 692.14) | 9 |
| 95%/99.5% quantile comparison, Normal vs. Student-t | 2.2117 vs 2.0538 (95%); 3.4636 vs 4.6201 (99.5%) | 10 |

### Question 3 — correlation structure (`problem3.csv`)

| Number | Value | Cell |
|---|---|---|
| pairplot of x1–x4 | inline figure | 13 |
| Pearson correlation matrix | printed 4×4 table | 14 |
| Spearman correlation matrix | printed 4×4 table | 14 |

### Question 4 — bivariate Normal conditioning (`problem4.csv`)

| Number | Value | Cell |
|---|---|---|
| sample covariance matrix of (x1, x2) | Var(x1)=1.099875, Cov=1.696902, Var(x2)=4.104749 | 17 |
| conditional-mean coefficient (beta) | 1.542813 | 18 |
| conditional variance / std dev | 1.486746 / 1.219322 | 18 |
| fraction of observations inside the 95% conditional band | 0.935 | 18 |
| conditional expectation + 95% band plot | inline figure | 19 |
| coverage within 1 SD / between 1–2 SD / beyond 2 SD of x1's mean | 95.65% (n=759) / 85.79% (n=190) / 90.20% (n=51) | 20 |

### Question 5 — time series model selection (`problem5.csv`)

| Number | Value | Cell |
|---|---|---|
| time series, ACF, PACF plots | inline figures | 22 |
| AICc: AR(1), AR(2), AR(3) | 1416.3654, 1368.3393, 1368.4267 | 23 |
| AICc: MA(1), MA(2), MA(3) | 1389.0843, 1381.2316, 1374.1165 | 23 |
| selected model (lowest AICc) | AR(2) | 23 |
