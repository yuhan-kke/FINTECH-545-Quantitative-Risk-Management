"""risklib - quantitative risk management toolkit.

Modules
-------
moments     first4_moments, tail_exceedances
fitting     fit_normal, fit_t, ols, fit_regression_normal, fit_regression_t,
            aic, aicc, bic, compare_fits
covariance  pearson, spearman, joint_counts, missing_cov, ew_weights,
            ew_diagnostics, ew_std, ew_cov, ew_cov_mixed, cov2cor, cor2cov
psd         is_psd, min_eigenvalue, chol_psd, try_cholesky, near_psd,
            higham_psd, frobenius
mvn         conditional_normal, band_coverage, simulate_normal, simulate_pca
risk        var_hist, es_hist, var_normal, es_normal, var_t, es_t, var_dist,
            es_dist, delta_normal_var, delta_normal_es
copula      rank_uniforms, margin_uniforms, joint_tail_counts, kendall_matrix,
            kendall_to_pearson, fit_gaussian_copula, fit_t_copula,
            simulate_copula, uniforms_to_margins, t_tail_dependence
factor      fit_factor_model, FactorModel
timeseries  fit_arma, select_arma, model_aicc, acf, pacf
returns     return_calculate
"""
from .moments import (Moments, first4_moments, tail_exceedances, mixture_excess_kurtosis,
                      moments_table, drop_most_extreme)
from .fitting import (FitResult, aic, aicc, bic, compare_fits, fit_normal, fit_t,
                      ols, fit_regression_normal, fit_regression_t)
from .covariance import (cov2cor, cor2cov, pearson, spearman, joint_counts,
                         missing_cov, ew_weights, ew_diagnostics, ew_std, ew_cov,
                         ew_cov_mixed)
from .psd import (is_psd, min_eigenvalue, frobenius, chol_psd, try_cholesky,
                  near_psd, higham_psd)
from .mvn import conditional_normal, band_coverage, simulate_normal, simulate_pca
from .risk import (var_hist, es_hist, var_es_hist, var_normal, es_normal, var_t,
                   es_t, var_dist, es_dist, delta_normal_var, delta_normal_es,
                   portfolio_variance, var_es_table, normal_var_es_table,
                   subadditivity, ew_var_standard_error)
from .copula import (CopulaFit, rank_uniforms, margin_uniforms, joint_tail_counts,
                     kendall_matrix, kendall_to_pearson, gaussian_copula_logpdf,
                     t_copula_logpdf, fit_gaussian_copula, fit_t_copula,
                     simulate_copula, uniforms_to_margins, t_tail_dependence,
                     fit_margins, compare_copulas, simulate_returns, outer_tail_mask)
from .factor import FactorModel, fit_factor_model, factor_var_table
from .timeseries import fit_arma, model_aicc, select_arma, acf, pacf, variance_breakpoint
from .returns import return_calculate

from . import plots

__version__ = "0.3.0"
