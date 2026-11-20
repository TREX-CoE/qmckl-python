"""M3: the generated bindings work (PLAN.md §8, test_generated.py).

- ``test_precision_64`` — value return, no context (§5.4 rule 4);
- ``dgemm_safe`` — a faithful column-major BLAS passthrough (§5.4 rule
  7): column-major buffers, one non-square and one square shape, all
  four transa/transb;
- ``get_point_num`` / ``get_ao_basis_shell_num`` — the scalar-out class
  plus the error path (§5.5(a));
- ``ao_gaussian_vgl`` / ``ao_slater_vgl`` — the §5.5(e) ABI override:
  ``n``/``num_slater`` by value, ``A`` as ``const double *``, writable
  ``VGL``; the documented values fail if ``A`` were bound as int64;
- a read-only ``VGL`` is refused with ``TypeError`` (§5.5(e));
- a ``float32`` array is refused (``noconvert``, §5.2).
"""

import math

import numpy as np
import pytest

import qmckl


def test_test_precision_64():
    # double precision is 53 significant bits; the C routine returns the
    # measured precision, which must be at least that for correct hardware.
    assert qmckl.test_precision_64(1.0, 1.0) >= 53


@pytest.mark.parametrize("m,n,k", [(3, 2, 4), (2, 2, 2)])
@pytest.mark.parametrize("transa,transb",
                         [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")])
def test_dgemm_safe_column_major_passthrough(m, n, k, transa, transb):
    # §5.4 rule 7: the binding is a faithful column-major passthrough,
    # so the arrays are the documented column-major buffers.  Row-major
    # A of shape (k, m) when transa='T' and (m, k) otherwise is handed
    # over as bufA = np.ascontiguousarray(A.T): C-contiguous memory that
    # BLAS reads column-major as A itself (same for B, transb).  bufC =
    # np.zeros((n, m)) is the column-major (m, n) result, read back as
    # bufC.T.  Non-square and non-symmetric payloads keep a transposed
    # reading distinguishable.
    ctx = qmckl.Context()
    if transa == "N":
        A = np.arange(1, m * k + 1, dtype=np.float64).reshape(m, k)
    else:
        A = np.arange(1, k * m + 1, dtype=np.float64).reshape(k, m)
    if transb == "N":
        B = np.arange(1, k * n + 1, dtype=np.float64).reshape(k, n)
    else:
        B = np.arange(1, n * k + 1, dtype=np.float64).reshape(n, k)
    bufA = np.ascontiguousarray(A.T)
    bufB = np.ascontiguousarray(B.T)
    bufC = np.zeros((n, m), dtype=np.float64)
    alpha = 1.0
    qmckl.dgemm_safe(
        ctx, transa, transb, m, n, k, alpha,
        bufA, bufA.shape[1], bufB, bufB.shape[1], 0.0, bufC, m)
    opA = A if transa == "N" else A.T
    opB = B if transb == "N" else B.T
    np.testing.assert_allclose(bufC.T, opA @ opB, rtol=1e-12, atol=1e-12)


def test_scalar_out_class_and_error_path():
    # §8: get_point_num on a fresh context returns 0 — its C body has no
    # NOT_PROVIDED check, it just copies point.num — while
    # get_ao_basis_shell_num, also scalar-out, raises a
    # QMCKLStatusError starting [QMCKL_NOT_PROVIDED] (the error path).
    ctx = qmckl.Context()
    assert qmckl.get_point_num(ctx) == 0
    with pytest.raises(qmckl.QMCKLStatusError) as ei:
        qmckl.get_ao_basis_shell_num(ctx)
    assert str(ei.value).startswith("[QMCKL_NOT_PROVIDED]")


def test_ao_gaussian_vgl_documented_values():
    # §5.5(e): the header declares n as `const int64_t *` and A as
    # `const int64_t *`; the true bind(C) prototype takes n by value and
    # A as `const double *` (src/qmckl_ao_f.F90).  A bound as an int64
    # array, QMCkl would read the bit pattern as doubles (1 -> 5e-324)
    # and every value would be silently wrong, so these documented
    # values fail against the header's types.
    ctx = qmckl.Context()
    X = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    R = np.zeros(3, dtype=np.float64)
    A = np.array([1.0, 2.0], dtype=np.float64)
    ldv = 2
    # C-contiguous (5, 2): the flat memory is exactly the Fortran
    # VGL(ldv, 5) column-major layout; VGL[:, i] is orbital i.
    vgl = np.zeros((5, 2), dtype=np.float64)
    qmckl.ao_gaussian_vgl(ctx, X, R, 2, A, vgl, ldv)
    e1, e2 = math.exp(-0.25), math.exp(-0.5)
    np.testing.assert_allclose(vgl[0, 0], e1, rtol=1e-12, atol=0)
    np.testing.assert_allclose(vgl[0, 1], e2, rtol=1e-12, atol=0)
    np.testing.assert_allclose(vgl[1, 0], -e1, rtol=1e-12, atol=0)
    # laplacian of exp(-a r^2): (4 a^2 r^2 - 6 a) g; (4 a r^2 - 6) g coincides only at a=1
    np.testing.assert_allclose(vgl[4, 0], (4 * 1**2 * 0.25 - 6 * 1) * e1,
                               rtol=1e-12, atol=0)
    np.testing.assert_allclose(vgl[4, 1], (4 * 2**2 * 0.25 - 6 * 2) * e2,
                               rtol=1e-12, atol=0)


def test_ao_slater_vgl_documented_value():
    # The second §5.5(e) override: num_slater by value, A as
    # `const double *`, VGL writable.  N = 1, A = 1.0, |X - R| = 0.5
    # gives VGL[0] = r * exp(-a * r) = 0.5 * exp(-0.5).
    ctx = qmckl.Context()
    X = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    R = np.zeros(3, dtype=np.float64)
    N = np.array([1], dtype=np.int64)
    A = np.array([1.0], dtype=np.float64)
    vgl = np.zeros((5, 1), dtype=np.float64)
    qmckl.ao_slater_vgl(ctx, X, R, 1, N, A, vgl, 1)
    np.testing.assert_allclose(vgl[0, 0], 0.5 * math.exp(-0.5),
                               rtol=1e-12, atol=0)


def test_ao_gaussian_vgl_readonly_vgl_refused():
    # VGL is `const double *` in the header but `double *`
    # (intent(out)) in the true prototype, so it is bound writable
    # (§5.5(e)): a writable buffer is accepted, a read-only one is
    # refused with TypeError instead of being scribbled on.
    ctx = qmckl.Context()
    X = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    R = np.zeros(3, dtype=np.float64)
    A = np.array([1.0, 2.0], dtype=np.float64)
    vgl = np.zeros((5, 2), dtype=np.float64)
    qmckl.ao_gaussian_vgl(ctx, X, R, 2, A, vgl, 2)
    vgl.setflags(write=False)
    with pytest.raises(TypeError):
        qmckl.ao_gaussian_vgl(ctx, X, R, 2, A, vgl, 2)


def test_float32_array_refused():
    # noconvert(): a wrong-dtype array must be refused, not silently
    # copied to the expected dtype.
    ctx = qmckl.Context()
    A32 = np.ones((2, 2), dtype=np.float32)
    B = np.ones((2, 2), dtype=np.float64)
    C = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(TypeError):
        qmckl.dgemm_safe(ctx, "N", "N", 2, 2, 2, 1.0, A32, 2, B, 2, 0.0, C, 2)
