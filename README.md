# qmckl-python

Python bindings for [QMCkl](https://github.com/TREX-CoE/qmckl), generated with
[nanobind](https://nanobind.readthedocs.io).

Every public function of the installed `qmckl.h` is bound automatically at build
time, so the Python surface can never drift from the C library. On top of that
raw surface sits a small, hand-written, Pythonic layer (`Context` properties,
typed exceptions).

```python
import numpy as np
import qmckl

with qmckl.Context() as ctx:
    ctx.trexio_read("h2o.h5")
    print("electrons:", ctx.elec_num, "nuclei:", ctx.nucl_num)

    a = np.ascontiguousarray(np.random.rand(4, 3))
    b = np.ascontiguousarray(np.random.rand(4, 3))
    d = np.empty((4, 4))
    # C signature: (ctx, transa, transb, m, n, A, lda, B, ldb, C, ldc)
    qmckl.distance(ctx, "N", "N", 4, 4, a, 3, b, 3, d, 4)
    assert np.allclose(d, np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1))
```

## Requirements

* Python >= 3.10
* NumPy >= 1.23
* QMCkl **1.1 or newer, but not 2.x**, installed with a working `qmckl.pc`

The bindings are generated from *your* header and linked against *your*
library, so the QMCkl version you build with is the version you get.

```sh
# build QMCkl once (a Python-facing build should not spawn OpenMP threads)
./configure --prefix=$HOME/qmckl --without-openmp
make -j && make install

export PKG_CONFIG_PATH=$HOME/qmckl/lib/pkgconfig:$PKG_CONFIG_PATH
pkg-config --modversion qmckl      # must print 1.1.x or 1.x, never 0.x
```

Then:

```sh
pip install qmckl-python
# development install
pip install -e . --no-build-isolation
```

`pkg-config` must find **1.1 or newer**. If the build stops with
`found qmckl 0.3.1, but qmckl-python requires >= 1.1`, an old `qmckl.pc` is
shadowing the new one: inspect `pkg-config --variable=pcfiledir qmckl`.

## The generated layer

Names drop the `qmckl_` prefix and keep the C argument order:

| C | Python |
|---|---|
| `qmckl_get_mo_basis_mo_value(ctx, out, size_max)` | `qmckl.get_mo_basis_mo_value(ctx, out)` |
| `qmckl_dgemm_safe(ctx, tA, tB, m, n, k, alpha, A, size_max_A, B, size_max_B, beta, C, size_max_C)` | `qmckl.dgemm_safe(ctx, tA, tB, m, n, k, alpha, A, B, beta, C)` |

* Array arguments are `numpy` arrays with the exact element type of the C
  declaration (`float64`, `int32`, `int64`, ...) and must be C-contiguous.
  They are passed as `noconvert()`: a non-contiguous or wrongly-typed array is
  **rejected** rather than silently copied (a silent copy would swallow your
  output).
* `size_max` is not in the Python signature: it is derived from the array
  (`.size`), or from `len(file_name)` for `trexio_read`. QMCkl validates it and
  returns `QMCKL_INVALID_ARG_*`, so an undersized buffer raises instead of
  corrupting memory.
* The GIL is released around every C call.
* Functions that return a scalar take no output argument and return the value:
  `ctx.elec_num` calls `qmckl_get_electron_num(ctx)`.

## Errors

Failed calls raise `qmckl.QMCKLStatusError` (a `RuntimeError`) whose message
starts with the symbolic code:

```
qmckl.QMCKLStatusError: [QMCKL_INVALID_ARG_3] qmckl_get_ao_basis_ao_value: ...
```

`qmckl.errors.ERROR_CODES` maps names to codes and
`qmckl.errors.error_name(code)` does the reverse.

## Contexts

`qmckl.Context` owns one `qmckl_context`. It is destroyed when the object is
garbage-collected (no `close()` is needed, and `with` only calls
`qmckl_context_touch` on exit). `Context.copy()` performs a `qmckl_context_copy`
deep copy; the two contexts are then fully independent.

Integer properties (thin wrappers over the scalar getters): `elec_num`,
`nucl_num`, `point_num`, `walk_num`, `ao_num`, `mo_num`, `prim_num`,
`shell_num`.

Array properties:

| attribute | shape | settable |
|---|---|---|
| `point` | `(point_num, 3)` | yes |
| `nucl_coord` | `(nucl_num, 3)` | yes |
| `electron_coord` | `(walk_num, elec_num, 3)` | yes |
| `ao_value` | `(point_num, ao_num)` | no |
| `ao_vgl` | `(point_num, 5, ao_num)` | no |
| `mo_value` | `(point_num, mo_num)` | no |
| `mo_vgl` | `(point_num, 5, mo_num)` | no |
| `ee_distance` | `(walk_num, elec_num, elec_num)` | no |
| `en_distance` | `(point_num, nucl_num)` | no |
| `ee_potential` | `(walk_num,)` | no |
| `en_potential` | `(walk_num,)` | no |

Setters accept any array-like, convert it to a C-contiguous `float64` array,
and raise `ValueError` on a wrong shape; the `'N'` layout is used (the `'T'`
layout stays in the raw layer only).  `Context.trexio_read(path)` loads a
TREXIO file into the context and raises `NotImplementedError` on a build
without TREXIO.

## Row-major BLAS

`qmckl.dgemm(ctx, A, B, transa="N", transb="N", alpha=1.0)` computes
`alpha * op(A) @ op(B)` on ordinary row-major NumPy arrays and returns a new
C-contiguous `(m, n)` array; `op(X)` is `X` for `"N"` and `X.T` for `"T"`.
It packs the column-major buffers `qmckl_dgemm_safe` expects, so `lda`/`ldb`/
`ldc` are never user-visible, and it raises `ValueError` on a shape mismatch,
a bad flag, or a zero dimension. It shadows the generated `_core.dgemm`; the
raw column-major functions remain available as `qmckl._core.dgemm` and
`qmckl._core.dgemm_safe`.

## Traps

1. **Version coupling.** The bindings are generated from the installed
   `qmckl.h`. After upgrading QMCkl you must reinstall this package
   (`pip install --force-reinstall --no-build-isolation .`).
2. **Shared library only.** `qmckl.pc` carries no `Requires:` and no usable
   `Libs.private`, so static linking against `libqmckl.a` is not supported here.
3. **Shapes are not checked** in the raw layer. A wrongly shaped but large
   enough array is reinterpreted with C semantics. Use the `Context`
   properties, which allocate and check the documented shapes.
4. **TREXIO.** `qmckl.trexio_read` exists only if the linked library was
   built with TREXIO support: QMCkl 1.1.0 exports a `qmckl_trexio_read`
   stub even without TREXIO, so the build probe (not the symbol) decides.
   Check with `hasattr(qmckl, "trexio_read")`; see PLAN §11 trap 4.
5. **OpenMP.** A QMCkl built with `--enable-openmp` starts threads inside a
   single Python call. Prefer `--without-openmp` and thread in Python; the GIL
   is released, so NumPy/threading concurrency works.
6. `qmckl_string_of_error` is broken in the QMCkl 1.1.0 release tarball (empty
   switch, always "Unknown error"). This package carries its own code table.
7. **The two potential getters carry no `size_max`**
   (`qmckl_get_electron_ee_potential`, `qmckl_get_electron_en_potential`).
   The raw layer cannot check the buffer it is given; use the
   `Context.ee_potential` / `en_potential` properties, which allocate it.
8. **Zero points.** The four point-evaluated properties (`ao_value`, `ao_vgl`,
   `mo_value`, `mo_vgl`) refuse an empty point set with `ValueError`: with
   `point_num == 0` the provider allocates zero bytes, QMCkl 1.1.0 records
   that as a free slot, the next allocation overwrites the record, and the
   context's allocation counter never returns to zero.  An assert-enabled
   `qmckl_context_destroy` then aborts.
9. **`qmckl_distance` checks LDB with transa's bit** (QMCkl 1.1.0,
   `qmckl_distance_f.F90`; `distance_sq` and `distance_rescaled` are fine).
   With `transa='T', transb='N'` it spuriously demands `ldb >= n` and raises
   `QMCKL_INVALID_ARG_9` for the usual packed `ldb = 3`.  The test suite pins
   `n = 3`, the only size that passes all four flag combinations.

## Development

```sh
pytest -q                                   # needs a QMCkl prefix in PKG_CONFIG_PATH

# inspect what would be generated
python csrc/generate.py --header $(pkg-config --variable=includedir qmckl)/qmckl.h \
       --dump-json ~/OpenCode/tmp/qmckl_api.json
```

The generator is the only place that knows about C types. It parses the header,
classifies the declarations, and writes `generated_bindings.cpp` and
`_core.pyi` into the build directory; both are installed next to the extension.

## Continuous integration

CI builds QMCkl 1.1.0 from the pinned release tarball against a conda-forge
TREXIO/gfortran environment (micromamba) and runs the suite on ubuntu
py3.10/py3.12 and macos-14 py3.12, plus one `--without-trexio` job, which
verifies `trexio_read` is not bound.

## License

BSD-3-Clause, see `LICENSE`.
