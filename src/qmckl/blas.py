"""The one row-major convenience over QMCkl's column-major BLAS (PLAN.md §7.2b).

``qmckl.dgemm`` accepts NumPy (row-major) arrays and packs the column-major
buffers ``qmckl_dgemm_safe`` expects; every other function in the package is
a faithful passthrough.  The generated ``_core.dgemm`` and
``_core.dgemm_safe`` remain the raw column-major calls.
"""

import numpy as np

from ._core import dgemm_safe

__all__ = ["dgemm"]


def _op(x, trans, name):
    if not isinstance(trans, str):
        raise ValueError(f"{name} must be 'N' or 'T', got {trans!r}")
    trans = trans.upper()
    if trans == "N":
        return x
    if trans == "T":
        return x.T
    raise ValueError(f"{name} must be 'N' or 'T', got {trans!r}")


def dgemm(ctx, A, B, *, transa="N", transb="N", alpha=1.0):
    """``alpha * op(A) @ op(B)`` on row-major arrays.

    ``op(X)`` is ``X`` for ``"N"`` and ``X.T`` for ``"T"``; ``A`` is
    ``(m, k)`` and ``B`` is ``(k, n)`` after the transpose.  The result is a
    new C-contiguous ``(m, n)`` array.
    """
    a = np.asarray(A, dtype=np.float64)
    b = np.asarray(B, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"A and B must be 2-D, got {a.ndim}-D and {b.ndim}-D")
    op_a = _op(a, transa, "transa")
    op_b = _op(b, transb, "transb")
    m, k = op_a.shape
    k2, n = op_b.shape
    if k != k2:
        raise ValueError(
            f"incompatible shapes: op(A) is {op_a.shape}, op(B) is {op_b.shape}"
        )
    if m == 0 or n == 0 or k == 0:
        raise ValueError("qmckl_dgemm_safe requires m, n and k to be positive")
    buf_a = np.ascontiguousarray(op_a.T)  # column-major (m, k), lda = m
    buf_b = np.ascontiguousarray(op_b.T)  # column-major (k, n), ldb = k
    buf_c = np.zeros((n, m), dtype=np.float64)  # column-major (m, n), ldc = m
    dgemm_safe(ctx, "N", "N", m, n, k, alpha, buf_a, m, buf_b, k, 0.0, buf_c, m)
    return buf_c.T.copy()
