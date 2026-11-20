"""M4: the §7.2 properties — shapes, round-trips, refusals.

The basis-dependent properties need TREXIO and the committed data file;
they skip on a ``--without-trexio`` build (PLAN.md §8).
"""

import pathlib

import numpy as np
import pytest

import qmckl

DATA = pathlib.Path(__file__).parent / "data" / "h2o_5z.h5"


def _molecule():
    """2 nuclei, 3 electrons, 2 walkers: coordinates and distances."""
    ctx = qmckl.Context()
    qmckl.set_nucleus_num(ctx, 2)
    qmckl.set_nucleus_charge(ctx, np.array([1.0, 1.0]))
    ctx.nucl_coord = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    qmckl.set_electron_num(ctx, 2, 1)
    coord = np.zeros((2, 3, 3))
    coord[0] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    coord[1] = [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6], [1.7, 1.8, 1.9]]
    ctx.electron_coord = coord
    return ctx, coord


def _basis():
    if not hasattr(qmckl, "trexio_read"):
        pytest.skip("QMCkl was built without TREXIO")
    ctx = qmckl.Context()
    ctx.trexio_read(DATA)
    return ctx


def test_integers():
    ctx = qmckl.Context()
    assert ctx.point_num == 0
    assert ctx.walk_num == 0
    with pytest.raises(qmckl.QMCKLStatusError):
        ctx.nucl_num
    with pytest.raises(qmckl.QMCKLStatusError):
        ctx.elec_num
    mol, _ = _molecule()
    assert (mol.nucl_num, mol.elec_num, mol.walk_num, mol.point_num) == (2, 3, 2, 6)


def test_point_roundtrip():
    ctx = qmckl.Context()
    pts = np.array([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]])
    ctx.point = pts
    assert ctx.point_num == 3
    assert np.array_equal(ctx.point, pts)


def test_point_shape_refused():
    ctx = qmckl.Context()
    with pytest.raises(ValueError):
        ctx.point = np.zeros((2, 2))
    with pytest.raises(ValueError):
        ctx.point = np.zeros(6)


def test_nucl_coord_roundtrip():
    ctx = qmckl.Context()
    c = np.array([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]])
    ctx.nucl_coord = c  # sets nucleus_num implicitly
    assert ctx.nucl_num == 2
    assert np.array_equal(ctx.nucl_coord, c)
    ctx.nucl_coord = c + 1.0  # same count: re-set
    assert np.array_equal(ctx.nucl_coord, c + 1.0)
    with pytest.raises(ValueError):
        ctx.nucl_coord = np.zeros((3, 3))


def test_electron_coord_roundtrip():
    ctx, coord = _molecule()
    assert ctx.electron_coord.shape == (2, 3, 3)
    assert np.allclose(ctx.electron_coord, coord)
    with pytest.raises(ValueError):
        ctx.electron_coord = np.zeros((2, 4, 3))


def test_distance_shapes():
    ctx, _ = _molecule()
    assert ctx.ee_distance.shape == (2, 3, 3)
    assert ctx.en_distance.shape == (6, 2)
    assert ctx.ee_potential.shape == (2,)
    assert ctx.en_potential.shape == (2,)
    assert np.isfinite(ctx.ee_distance).all()
    assert np.isfinite(ctx.en_potential).all()


def test_touch_recomputes():
    ctx, _ = _molecule()
    before = ctx.ee_distance
    ctx.touch()
    after = ctx.ee_distance
    assert np.array_equal(before, after)


def test_basis_integers_and_shapes():
    ctx = _basis()
    assert ctx.ao_num > 0 and ctx.mo_num > 0
    assert ctx.prim_num > 0 and ctx.shell_num > 0
    ctx.point = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    n = ctx.point_num
    assert ctx.ao_value.shape == (n, ctx.ao_num)
    assert ctx.ao_vgl.shape == (n, 5, ctx.ao_num)
    assert ctx.mo_value.shape == (n, ctx.mo_num)
    assert ctx.mo_vgl.shape == (n, 5, ctx.mo_num)
    assert np.isfinite(ctx.ao_vgl).all()
    assert np.isfinite(ctx.mo_value).all()


def test_basis_needs_points():
    # PLAN.md §11: with no points the provider allocates zero bytes, which
    # corrupts QMCkl 1.1.0's allocation bookkeeping and aborts an
    # assert-enabled qmckl_context_destroy.
    ctx = _basis()
    assert ctx.point_num == 0
    for get in (lambda: ctx.ao_value, lambda: ctx.ao_vgl,
                lambda: ctx.mo_value, lambda: ctx.mo_vgl):
        with pytest.raises(ValueError):
            get()
