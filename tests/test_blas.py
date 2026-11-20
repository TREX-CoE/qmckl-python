"""M4: the row-major dgemm (§7.2b) and the raw adjugate_safe (PLAN.md §8)."""

import numpy as np
import pytest

import qmckl


def test_dgemm_all_transposes():
    ctx = qmckl.Context()
    rng = np.random.default_rng(7)
    # op(A) is (3, 2), op(B) is (2, 4): the inputs are non-square and a
    # transposed reading of either buffer cannot pass.
    op_a = rng.normal(size=(3, 2))
    op_b = rng.normal(size=(2, 4))
    for transa in ("N", "T"):
        for transb in ("N", "T"):
            a = op_a if transa == "N" else op_a.T
            b = op_b if transb == "N" else op_b.T
            c = qmckl.dgemm(ctx, a, b, transa=transa, transb=transb, alpha=1.7)
            assert c.shape == (3, 4)
            assert np.allclose(c, 1.7 * op_a @ op_b, atol=1e-12)


def test_dgemm_defaults():
    ctx = qmckl.Context()
    a = np.arange(6.0).reshape(3, 2)
    b = np.arange(8.0).reshape(2, 4) + 0.5
    assert np.allclose(qmckl.dgemm(ctx, a, b), a @ b)


def test_dgemm_shape_mismatch():
    ctx = qmckl.Context()
    with pytest.raises(ValueError):
        qmckl.dgemm(ctx, np.zeros((3, 2)), np.zeros((3, 4)))


def test_dgemm_bad_flag():
    ctx = qmckl.Context()
    a = np.zeros((3, 2))
    b = np.zeros((2, 4))
    with pytest.raises(ValueError):
        qmckl.dgemm(ctx, a, b, transa="C")
    with pytest.raises(ValueError):
        qmckl.dgemm(ctx, a, b, transb="x")


def test_dgemm_empty():
    ctx = qmckl.Context()
    with pytest.raises(ValueError):
        qmckl.dgemm(ctx, np.zeros((0, 2)), np.zeros((2, 3)))


def test_adjugate_safe():
    # The raw layer stays column-major: pack A.T, read B.T back.
    ctx = qmckl.Context()
    a = np.array([[4.0, 1.0, 2.0], [0.5, 3.0, -1.0], [2.0, 0.0, 5.0]])
    buf_a = np.ascontiguousarray(a.T)
    buf_b = np.empty((3, 3))
    det = np.empty(1)
    qmckl.adjugate_safe(ctx, 3, buf_a, 3, buf_b, 3, det)
    assert det[0] == pytest.approx(np.linalg.det(a), rel=1e-12)
    assert np.allclose(buf_b.T, np.linalg.inv(a) * det[0], atol=1e-12)
