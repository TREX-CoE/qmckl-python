"""M5: TREXIO round-trip on QMCkl's own H2O test file (PLAN.md §8).

The whole module skips on a ``--without-trexio`` build.
"""

import pathlib

import numpy as np
import pytest

import qmckl

pytestmark = pytest.mark.skipif(
    not hasattr(qmckl, "trexio_read"), reason="QMCkl was built without TREXIO"
)

DATA = pathlib.Path(__file__).parent / "data" / "h2o_5z.h5"


@pytest.fixture(scope="module")
def ctx():
    c = qmckl.Context()
    c.trexio_read(DATA)
    return c


def test_counts_and_geometry(ctx):
    # Facts of the file: H2O in a 5-zeta basis.
    assert (ctx.elec_num, ctx.nucl_num) == (10, 3)
    assert (ctx.ao_num, ctx.mo_num) == (266, 201)
    charge = np.empty(ctx.nucl_num, dtype=np.float64)
    qmckl.get_nucleus_charge(ctx, charge)
    assert np.array_equal(charge, np.array([8.0, 1.0, 1.0]))
    coord = ctx.nucl_coord
    assert coord.shape == (3, 3)
    # O at the origin, two O-H bonds of ~1.809 bohr.
    assert np.array_equal(coord[0], np.zeros(3))
    oh = np.linalg.norm(coord - coord[0], axis=1)
    assert np.allclose(oh[1:], 1.809, atol=0.01)


def test_vgl_value_identity(ctx):
    # upstream tests/test_qmckl_mo.c asserts exactly these identities
    # (that file carries no hardcoded numbers).
    ctx.electron_coord = np.random.default_rng(5).normal(scale=0.7, size=(2, 10, 3))
    assert np.allclose(ctx.mo_vgl[:, 0, :], ctx.mo_value, atol=1e-12)
    assert np.allclose(ctx.ao_vgl[:, 0, :], ctx.ao_value, atol=1e-12)


def test_potentials_match_numpy(ctx):
    coords = np.random.default_rng(3).normal(scale=0.7, size=(2, 10, 3))
    ctx.electron_coord = coords
    charge = np.empty(ctx.nucl_num, dtype=np.float64)
    qmckl.get_nucleus_charge(ctx, charge)
    nucl = ctx.nucl_coord
    ref_en = np.array([
        sum(-sum(charge[k] / np.linalg.norm(x - nucl[k])
                 for k in range(ctx.nucl_num))
            for x in coords[w])
        for w in range(ctx.walk_num)
    ])
    assert np.allclose(ctx.en_potential, ref_en, atol=1e-12)
    ref_ee = np.array([
        sum(1.0 / np.linalg.norm(coords[w][i] - coords[w][j])
            for i in range(ctx.elec_num)
            for j in range(i + 1, ctx.elec_num))
        for w in range(ctx.walk_num)
    ])
    assert np.allclose(ctx.ee_potential, ref_ee, atol=1e-12)
