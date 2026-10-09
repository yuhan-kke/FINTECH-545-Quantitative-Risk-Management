"""PSD checks, PSD-safe Cholesky, and non-PSD repair (Rebonato-Jaeckel,
Higham nearest correlation)."""
from __future__ import annotations

import numpy as np

from .covariance import cov2cor, cor2cov


def _arr(A):
    return np.asarray(A, dtype=float)


def is_psd(A, tol: float = 1e-8) -> bool:
    """True if the smallest eigenvalue is >= -tol."""
    return bool(np.linalg.eigvalsh(_arr(A))[0] >= -tol)


def min_eigenvalue(A) -> float:
    return float(np.linalg.eigvalsh(_arr(A))[0])


def frobenius(A, B) -> float:
    return float(np.linalg.norm(_arr(A) - _arr(B), "fro"))


def chol_psd(A, tol: float = 1e-8) -> np.ndarray:
    """Lower-triangular L with L L' = A for a PSD (possibly singular) matrix.

    Unlike numpy's cholesky this tolerates zero eigenvalues: a zero pivot sets
    its column to 0. Raises ValueError if A is meaningfully non-PSD.
    """
    A = _arr(A)
    n = A.shape[0]
    L = np.zeros_like(A)
    for j in range(n):
        s = A[j, j] - L[j, :j] @ L[j, :j]
        if -tol <= s <= 0:
            s = 0.0
        elif s < -tol:
            raise ValueError(f"Matrix is not PSD (pivot {j} = {s:.3e})")
        L[j, j] = np.sqrt(s)
        if L[j, j] == 0.0:
            continue
        ir = 1.0 / L[j, j]
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - L[i, :j] @ L[j, :j]) * ir
    return L


def try_cholesky(A):
    """(success, L or error message) using numpy's strict Cholesky."""
    try:
        return True, np.linalg.cholesky(_arr(A))
    except np.linalg.LinAlgError as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Repairs
# ---------------------------------------------------------------------------
def _as_corr(A):
    """If A is a covariance, return (correlation, sd); else (A, None)."""
    A = _arr(A)
    if np.allclose(np.diag(A), 1.0):
        return A.copy(), None
    R, sd = cov2cor(A)
    return R, sd


def near_psd(A, eps: float = 0.0) -> np.ndarray:
    """Rebonato & Jaeckel (1999) spectral repair.

    Clip eigenvalues at eps, rebuild B = V sqrt(L+), rescale rows of B to unit
    length so the diagonal is exactly 1, return B B'. Covariance input is
    converted to correlation first and scaled back afterwards.
    """
    R, sd = _as_corr(A)
    lam, V = np.linalg.eigh(R)
    lam = np.maximum(lam, eps)
    B = V * np.sqrt(lam)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    out = B @ B.T
    return out if sd is None else cor2cov(out, sd)


def higham_psd(A, weights=None, tol: float = 1e-10, max_iter: int = 10_000,
               return_iterations: bool = False):
    """Higham (2002) nearest correlation matrix (weighted Frobenius norm)
    via alternating projections with Dykstra's correction.

    weights: optional vector w; minimises ||W^1/2 (A - X) W^1/2||_F.
    Covariance input is converted to correlation and scaled back.
    """
    R, sd = _as_corr(A)
    n = R.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    wh, wih = np.sqrt(w), 1 / np.sqrt(w)

    def proj_psd(X):
        Xw = wh[:, None] * X * wh[None, :]
        lam, V = np.linalg.eigh(Xw)
        Xw = (V * np.maximum(lam, 0)) @ V.T
        return wih[:, None] * Xw * wih[None, :]

    Y, dS = R.copy(), np.zeros_like(R)
    it = 0
    for it in range(1, max_iter + 1):
        Rk = Y - dS
        X = proj_psd(Rk)
        dS = X - Rk
        Y_new = X.copy()
        np.fill_diagonal(Y_new, 1.0)
        done = np.linalg.norm(Y_new - Y, "fro") < tol
        Y = Y_new
        if done:
            break
    Y = (Y + Y.T) / 2
    out = Y if sd is None else cor2cov(Y, sd)
    return (out, it) if return_iterations else out
