"""M5: the three distance kernels against NumPy (PLAN.md §8).

All three share the buffer conventions: ``'N'`` takes a C-contiguous
``(m, 3)`` array with ``lda = 3``; ``'T'`` takes ``(3, m)`` with
``lda = m`` (verified in ``src/qmckl_distance_f.F90``: ``'T'`` reads
``A(i, 1..3)``, i.e. all x's, then all y's, then all z's).  ``C`` is the
column-major ``(ldc, n)`` buffer with ``ldc = m``, so a C-contiguous
``(n, m)`` array is exactly that layout and ``buf_c.T`` is the result —
the same convention as ``dgemm``.

``n`` is pinned to 3 on purpose.  In ``qmckl_distance`` (not in
``distance_sq`` nor ``distance_rescaled``) the LDB checks are gated on
``iand(transab,1)`` — transa's bit — instead of ``iand(transab,2)``, so they
demand ``ldb >= 3`` when ``transa == 'N'`` and ``ldb >= n`` when
``transa == 'T'``.  With the packed layouts above (``ldb`` is 3 for ``'N'``
and ``n`` for ``'T'``) only ``n == 3`` satisfies both, whatever ``m`` is;
any other ``n`` loses one flag combination to a spurious
``QMCKL_INVALID_ARG_9``.  ``m = 4`` keeps the two sets distinguishable.
"""

import numpy as np
import pytest

import qmckl

M, N = 4, 3


def _run(fn, ctx, a, b, transa, transb, *extra):
    """Call a distance kernel and return the (m, n) result."""
    m, n = a.shape[0], b.shape[0]
    if transa == "N":
        buf_a, lda = np.ascontiguousarray(a), 3
    else:
        buf_a, lda = np.ascontiguousarray(a.T), m
    if transb == "N":
        buf_b, ldb = np.ascontiguousarray(b), 3
    else:
        buf_b, ldb = np.ascontiguousarray(b.T), n
    buf_c = np.zeros((n, m), dtype=np.float64)
    fn(ctx, transa, transb, m, n, buf_a, lda, buf_b, ldb, buf_c, m, *extra)
    return buf_c.T


def _pair():
    rng = np.random.default_rng(11)
    return rng.normal(size=(M, 3)), rng.normal(size=(N, 3))


def test_distance_sq():
    ctx = qmckl.Context()
    a, b = _pair()
    ref = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
    for transa in "NT":
        for transb in "NT":
            c = _run(qmckl.distance_sq, ctx, a, b, transa, transb)
            assert c.shape == (M, N)
            assert np.allclose(c, ref, atol=1e-14)


def test_distance():
    ctx = qmckl.Context()
    a, b = _pair()
    ref = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    for transa in "NT":
        for transb in "NT":
            c = _run(qmckl.distance, ctx, a, b, transa, transb)
            assert c.shape == (M, N)
            assert np.allclose(c, ref, atol=1e-14)


def test_distance_rescaled():
    ctx = qmckl.Context()
    a, b = _pair()
    kappa = 2.5
    ref = (1.0 - np.exp(-kappa * np.linalg.norm(
        a[:, None, :] - b[None, :, :], axis=-1))) / kappa
    for transa in "NT":
        for transb in "NT":
            c = _run(qmckl.distance_rescaled, ctx, a, b, transa, transb, kappa)
            assert c.shape == (M, N)
            assert np.allclose(c, ref, atol=1e-14)


def test_distance_errors():
    # The raw layer is a passthrough: bad arguments raise QMCKLStatusError,
    # not ValueError (measured: INVALID_ARG_1 for the flag, _4 for m,
    # _7 for lda).
    ctx = qmckl.Context()
    a, b = _pair()
    buf_a = np.ascontiguousarray(a)
    buf_b = np.ascontiguousarray(b)
    buf_c = np.zeros((N, M))
    with pytest.raises(qmckl.QMCKLStatusError):
        qmckl.distance(ctx, "X", "N", M, N, buf_a, 3, buf_b, 3, buf_c, M)
    with pytest.raises(qmckl.QMCKLStatusError):
        qmckl.distance(ctx, "N", "N", 0, N, buf_a, 3, buf_b, 3, buf_c, M)
    with pytest.raises(qmckl.QMCKLStatusError):
        qmckl.distance(ctx, "N", "N", M, N, buf_a, 2, buf_b, 3, buf_c, M)
