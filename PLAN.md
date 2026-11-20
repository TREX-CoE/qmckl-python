# qmckl-python — implementation spec (v1)

Status: agreed design, ready to implement.
Audience: a build agent that has this file and nothing else.
Scope: nanobind bindings for **all** public QMCkl functions, generated at build
time from the installed `qmckl.h`, plus a thin Pythonic layer.

---

## 1. Decisions (do not re-litigate)

| Topic | Decision |
|---|---|
| Binding technology | nanobind (`nb::ndarray` for arrays) |
| Coverage | every `qmckl_*` declaration in the installed header, mechanically |
| Library provisioning | **system** `libqmckl` via `pkg-config qmckl >= 1.1`; no vendored C sources |
| Bindings | **generated at build time** from the header that will be linked |
| Wheels / conda | out of scope for v1 |
| Distribution name | `qmckl-python`; import name `qmckl`; extension `qmckl._core` |
| Public naming | C name minus the `qmckl_` prefix: `qmckl.get_mo_basis_mo_value` |
| Shape checking | none in the generated layer (`nb::c_contig` + dtype only); shapes are checked by the Pythonic layer for the documented properties |
| Errors | one C++ exception type `QMCKLStatusError(RuntimeError)`, message carries the symbolic code |

Rationale for build-time generation: the header and the library can never
disagree, and a `--without-trexio` build simply loses `qmckl_trexio_read`.

---

## 2. Verified facts (do not spend time re-deriving)

These were measured on QMCkl **v1.1.0** (`qmckl-1.1.0.tar.gz`):

- The release tarball ships already-tangled C (`src/*.c`, 15 files) and builds
  with `./configure --without-openmp && make && make install`; `make check` is
  15/15 PASS. No IRPF90, no emacs, no org needed.
- `make install` produces `include/qmckl.h`, `lib/libqmckl.{a,dylib,so}` and
  `lib/pkgconfig/qmckl.pc` with `Version: 1.1.0`,
  `Cflags: -I${includedir}`, `Libs: -L${libdir} -lqmckl`. There is **no**
  `Requires:` line and no `Libs.private` for TREXIO in practice → static
  linking is not supported by this package.
- `qmckl.h` declares **239** `qmckl_*` functions (count with the parser in
  §5.1): 223 return `qmckl_exit_code`.
- Array/size conventions:
  - **125 functions**: `(ctx, <one array>, size_max)` — the getter/setter shape.
  - `qmckl_dgemm_safe` has `size_max_A/B/C`; `qmckl_adjugate_safe` has
    `size_max_A/B`.
  - `qmckl_trexio_read(ctx, file_name, size_max)` — `size_max` is
    `strlen(file_name)`.
  - 23 functions take exactly one array and no `size_max`: 18 are scalar
    out-parameters, 4 are real arrays (`ee_potential`, `en_potential`,
    `lkpm_combined_index_doc`, `lkpm_combined_index_hpc`), 1 is excluded
    (`qmckl_get_error`).
- `qmckl_context_copy()` performs a **deep copy** (`src/qmckl_context.c:168`),
  so every handle is destroyed exactly once; no refcounting.
- On failure, `qmckl_get_error(ctx, &code, fun, msg)` returns the stored error;
  buffers are `QMCKL_MAX_FUN_LEN` (256) and `QMCKL_MAX_MSG_LEN` (1024)
  (`include/qmckl.h`). `qmckl_get_error` **asserts** that the context is valid —
  guard with `qmckl_context_check()`.
- **Do not use `qmckl_string_of_error`**: in the v1.1.0 release tarball its
  `switch` is empty (always returns `"Unknown error"`). A locally tangled git
  build is correct. The package therefore owns its code→name table. (Upstream
  bug, worth reporting.)
- QMCkl has **no BLAS/LAPACK dependency**; `qmckl_blas.c` is self-contained.
  TREXIO is the only external dependency (`HAVE_TREXIO`).
- The PyPI `trexio` wheel does **not** ship a linkable `libtrexio`; it cannot
  be used as a build dependency.

---

## 3. Repository layout

```
qmckl-python/
├── pyproject.toml
├── CMakeLists.txt
├── README.md
├── PLAN.md                 # this file
├── LICENSE                 # BSD-3-Clause, copied verbatim from QMCkl
├── csrc/
│   ├── module.cpp          # NB_MODULE; registers Context + generated defs
│   ├── context.hpp         # qmckl::Context (RAII), qmckl::check()
│   ├── context.cpp
│   └── generate.py         # header parser + C++ and .pyi emitters
├── src/qmckl/
│   ├── __init__.py
│   ├── context.py          # class Context(_core._Context) + convenience API
│   └── errors.py           # ERROR_CODES, error_name()
├── tests/
│   ├── data/h2o_5z.h5      # 4.6 MB, from the QMCkl tarball (share/qmckl/test_data)
│   ├── test_surface.py
│   ├── test_context.py
│   ├── test_errors.py
│   ├── test_blas.py
│   ├── test_distance.py
│   ├── test_trexio.py
│   └── test_shapes.py
└── .github/workflows/ci.yml
```

---

## 4. Build

### 4.1 `pyproject.toml`

Already committed. Key points: `scikit-build-core>=0.10` +
`nanobind>=3.1` build requirements, `requires-python = ">=3.10"`,
runtime dependency `numpy>=1.23`, `wheel.packages = ["src/qmckl"]`.

NumPy headers are **not** needed at compile time (nanobind's ndarray support is
framework-agnostic).

### 4.2 `CMakeLists.txt`

Written and validated on 2026-11-20 against QMCkl 1.1.0. It:

1. `pkg_check_modules(QMCKL QUIET IMPORTED_TARGET "qmckl>=1.1")`, and on
   failure a `FATAL_ERROR` that names the version actually found
   (`found qmckl 0.3.1, but qmckl-python requires >= 1.1`) plus the
   `PKG_CONFIG_PATH` recipe. Hard error if `QMCKL_VERSION >= 2.0`.
   *Both branches were exercised.*
2. Locates `qmckl.h` via `find_path(... PATHS ${QMCKL_INCLUDE_DIRS}
   NO_DEFAULT_PATH)` → `QMCKL_HEADER`.
3. Detects whether the linked library exports `qmckl_trexio_read` by
   concatenating the output of `nm -D` **and** `nm -g` on the resolved
   `libqmckl` and searching the union. Neither flag works everywhere
   (verified: on macOS `nm -D` fails with "File format has no dynamic symbol
   table" while `nm -g` finds 25 occurrences; on a stripped Linux `.so` it is
   the other way round). `-DQMCKL_HAS_TREXIO=AUTO|ON|OFF` overrides; if neither
   `nm` run succeeds the configure fails and asks for the explicit flag rather
   than guessing. *Verified ON against the 1.1.0 install and OFF via the
   override.*
4. Runs `csrc/generate.py` as a custom command producing
   `${CMAKE_CURRENT_BINARY_DIR}/generated_bindings.cpp` and
   `${CMAKE_CURRENT_BINARY_DIR}/_core.pyi`.
5. `nanobind_add_module(_core NB_STATIC csrc/module.cpp csrc/context.cpp
   ${generated})`, links `PkgConfig::QMCKL`.
6. `install(TARGETS _core LIBRARY DESTINATION qmckl RUNTIME DESTINATION qmckl)`
   and `install(FILES _core.pyi DESTINATION qmckl)` — the stub lands next to
   the extension in both wheel and editable installs, with no dependency on
   `SKBUILD_*` variables. *Verified end to end with a throwaway project
   (nanobind 3.1.0, scikit-build-core, `wheel.packages = ["src/pkg"]`): the
   wheel contains `pkg/{__init__.py,_core.*.so,_core.pyi}`, and in an editable
   install `pkg.__path__` is
   `[<site-packages>/pkg, <src>/pkg]` so `_core.pyi` is found.*

Environment note (macmini, 2026-11-20, already fixed — do not re-diagnose): the
C++ toolchain was broken. CLT 27.2 shipped
`/Library/Developer/CommandLineTools/usr/include/c++/v1` as a 2020 leftover
containing only `__cxx_version`, so every C++ compile died on
`fatal error: 'cstddef' file not found`, while C compiles (QMCkl itself) worked
fine. `-isysroot` and `SDKROOT` do not help: the clang 21 driver searches the
toolchain directory and never the SDK's copy. Repaired by re-pointing that
directory at the default SDK's headers
(`v1 -> ../../../SDKs/MacOSX.sdk/usr/include/c++/v1`; the old directory is kept
as `v1.stale-20261120-150333`, removable later). Verified afterwards with a
nanobind + scikit-build-core editable install and a `<vector>`/`<string>`
compile-link-run. No `CXXFLAGS` workaround is needed.

---

## 5. Code generator — `csrc/generate.py`

Standalone CLI, stdlib only:

```
generate.py --header PATH/qmckl.h --cpp OUT.cpp --pyi OUT.pyi [--trexio|--no-trexio]
generate.py --header PATH/qmckl.h --dump-json OUT.json     # for review/diffing
```

### 5.1 Parsing

1. Read the header; delete `/* … */` and `//…` comments.
2. Delete preprocessor lines (`^#…`) and everything inside `{…}` (inline
   function bodies), so only declarations remain.
3. Split on `;`; keep declarators whose function name starts with `qmckl_`.
4. Normalise each argument: strip `const`, collapse whitespace, derive
   `base type` and `pointer level`. **Beware `double* const C`** (in
   `qmckl_adjugate_safe`, `qmckl_distance`, `qmckl_dgemm_safe`, ...): the
   pointer is const, the pointee is not, so it is an *output* array. Decide
   const-ness from the declarator as a whole (`const` before `*`), never with a
   bare `"const" in arg` test.

Sanity assertions (fail loudly, do not guess):

- `len(functions) >= 230`
- `"qmckl_dgemm_safe" in names` (this symbol marks the ≥ 1.1 API)
- every pointer parameter is classified by §5.3/§5.5 rules
- no Python keyword remains among argument names after suffixing with `_`

### 5.2 Type mapping

| C parameter | stub annotation | C++ lambda parameter |
|---|---|---|
| `qmckl_context` (first arg) | `Context` | `qmckl::Context &ctx` |
| `int64_t`, `int32_t`, `int16_t`, `int` | `int` | same C type |
| `double`, `float` | `float` | same C type |
| `_Bool` | `bool` | `bool` |
| `char` (`transp`, `transa`, `transb`, `basis_type`) | `str` | `char` |
| `const char *` | `str` | `std::string` |
| `T *` (non-const) | `numpy.typing.NDArray[T]` | `nb::ndarray<T, nb::c_contig>` |
| `const T *` | `numpy.typing.NDArray[T]` | `nb::ndarray<const T, nb::c_contig>` |

Return mapping: `qmckl_exit_code` → `None` (raise on non-zero); `_Bool` →
`bool`; `int32_t` → `int`; `double` → `float`; `const char *` → `str`; the four
hand-written cases in §5.6 are not generated.

**Mandatory**: every array argument gets `nb::arg("<name>").noconvert()`.
Without it nanobind silently copies non-contiguous or wrongly-typed arrays, and
outputs would be written into a temporary. This is the single most important
line in the generator.

Verified in the nanobind 3.1.0 sources (`include/nanobind/ndarray.h`,
`src/nb_ndarray.cpp`) — do not re-litigate:

- `noconvert()` reaches the importer as `cast_flags::convert == 0`, so
  dtype/contiguity conversions are refused instead of silently applied.
- `nb::ndarray<const T>` sets `cfg.ro = true` and only *advertises*
  `writable=False`; a writable NumPy array is still accepted. Use `const T` for
  inputs.
- `nb::ndarray<T>` sets `cfg.ro = false` and **rejects** a read-only array
  ("Cannot provide writable access to a read-only ndarray"). That is the right
  behaviour for output parameters (`np.frombuffer(...)` is not a valid output
  buffer).

### 5.3 `size_max` rule

For each `size_max*` parameter, drop it from the Python signature and compute it
from the array it describes:

- `size_max_X` → the parameter named `X` (or ending in `_X`): `(int64_t) X.size()`.
- bare `size_max` → the unique array parameter: `(int64_t) arr.size()`.
- `qmckl_trexio_read` → `(int64_t) std::strlen(file_name)` (explicit exception).

QMCkl itself validates `size_max` and returns `QMCKL_INVALID_ARG_*`, so an
undersized buffer raises cleanly — never an out-of-bounds write.

### 5.4 Emitted call

Every generated binding has exactly this shape (GIL released only around the C
call; pointers and sizes captured first):

```cpp
m.def("get_mo_basis_mo_value",
      [](qmckl::Context &ctx,
         nb::ndarray<double, nb::c_contig> mo_value) {
          double        *p = mo_value.data();
          const int64_t  n = static_cast<int64_t>(mo_value.size());
          qmckl_exit_code rc;
          {
              nb::gil_scoped_release nogil;
              rc = ::qmckl_get_mo_basis_mo_value(ctx.h, p, n);
          }
          qmckl::check(ctx.h, rc, "qmckl_get_mo_basis_mo_value");
      },
      "mo_value"_a.noconvert());
```

All `m.def`s are emitted inside `void qmckl_python::register_generated(nb::module_ &m)`;
`csrc/module.cpp` declares and calls it. No docstrings in v1.

### 5.5 Special cases (explicit tables in the generator)

**(a) Scalar out-parameters** — one pointer parameter, scalar element type, no
`size_max`; return the value instead of taking a buffer:

```
qmckl_get_point_num                      qmckl_get_mo_basis_mo_num
qmckl_get_nucleus_num                    qmckl_get_jastrow_champ_aord_num
qmckl_get_nucleus_repulsion              qmckl_get_jastrow_champ_bord_num
qmckl_get_electron_num                   qmckl_get_jastrow_champ_cord_num
qmckl_get_electron_up_num                qmckl_get_jastrow_champ_type_nucl_num
qmckl_get_electron_down_num              qmckl_get_jastrow_champ_rescale_factor_ee
qmckl_get_electron_walk_num              qmckl_get_jastrow_champ_dim_c_vector
qmckl_get_ao_basis_shell_num             qmckl_get_jastrow_champ_spin_independent
qmckl_get_ao_basis_prim_num
qmckl_get_ao_basis_ao_num
```

**(b) Array outputs without `size_max`** — keep as `nb::ndarray` parameters
(the caller allocates; QMCkl writes a fixed number of elements):

```
qmckl_get_electron_ee_potential      [walk_num]
qmckl_get_electron_en_potential      [walk_num]
qmckl_compute_lkpm_combined_index_doc
qmckl_compute_lkpm_combined_index_hpc
```

**(c) Hand-written bindings** (excluded from codegen):

```
qmckl_context_create / _copy / _check / _destroy   -> qmckl::Context
qmckl_context_touch                                -> Context.touch()
qmckl_lock / qmckl_unlock                          -> not exposed
qmckl_get_error / _set_error / _failwith           -> not exposed
qmckl_last_error                                   -> qmckl.last_error(ctx) -> str
qmckl_get_ao_basis_type                            -> qmckl.get_ao_basis_type(ctx) -> str
```

**(d) TREXIO** — when the linked library does not export `qmckl_trexio_read`,
skip that binding entirely (do not emit a stub for it either).

### 5.6 Stub file `_core.pyi`

One `def` per generated symbol with the mapped annotations, plus:

```python
class _Context:
    def __init__(self) -> None: ...
    def __init__(self, other: _Context) -> None: ...
    def touch(self) -> None: ...
    def valid(self) -> bool: ...
    @property
    def handle(self) -> int: ...

class QMCKLStatusError(RuntimeError): ...
```

The module must expose nothing else at top level.

---

## 6. C++ runtime

### 6.1 `csrc/context.hpp`

```cpp
namespace qmckl {

struct Context {
    int64_t h;

    Context();                        // qmckl_context_create(); throw if NULL
    Context(const Context &other);    // qmckl_context_copy(); throw if NULL (deep copy)
    Context &operator=(const Context &) = delete;
    ~Context();                       // qmckl_context_destroy(h); errors swallowed
};

struct QMCKLStatusError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] void check(int64_t ctx, int32_t rc, const char *fname);
const char *code_name(int32_t rc);    // generated table, see §5.6/§7
}
```

`check(ctx, rc, fname)` behaviour:

- `rc == 0` → return.
- otherwise build `"[<CODE_NAME>] <fname>: <message>"`, where `<message>` comes
  from `qmckl_get_error(ctx, &code, fun, msg)` when
  `qmckl_context_check(ctx) != 0`, and from `code_name(rc)` alone otherwise.
- throw `QMCKLStatusError`.

`code_name` is emitted next to the generated bindings from the
`#define QMCKL_*` macros of the header (single source of truth). Do **not** call
`qmckl_string_of_error` (see §2).

### 6.2 `csrc/module.cpp`

```cpp
NB_MODULE(_core, m) {
    nb::register_exception<qmckl::QMCKLStatusError>(
        m, "QMCKLStatusError", PyExc_RuntimeError);

    nb::class_<qmckl::Context>(m, "_Context")
        .def(nb::init<>())
        .def(nb::init<const qmckl::Context &>())
        .def("touch", [](qmckl::Context &c) {
            qmckl::check(c.h, ::qmckl_context_touch(c.h), "qmckl_context_touch");
        })
        .def("valid", [](const qmckl::Context &c) {
            return ::qmckl_context_check(c.h) != QMCKL_NULL_CONTEXT;
        })
        .def_prop_ro("handle", [](const qmckl::Context &c) { return c.h; });

    m.def("last_error", ...);            // 2048-byte buffer, no GIL issues
    m.def("get_ao_basis_type", ...);     // char buffer -> str

    qmckl_python::register_generated(m);
}
```

Order matters: the exception and `_Context` must be registered before the
generated functions reference them in signatures.

---

## 7. Python layer

### 7.1 `src/qmckl/__init__.py`

```python
from ._core import *              # generated functions, _Context, QMCKLStatusError
from .context import Context
from .errors import ERROR_CODES, error_name
__all__ = [...]
```

`Context` subclasses `_core._Context`; generated functions accept it because the
parameter type is `qmckl::Context &`.

### 7.2 `src/qmckl/context.py`

```python
class Context(_core._Context):
    @classmethod
    def copy(cls, other): return cls(other)          # deep copy

    def __enter__(self): return self
    def __exit__(self, *exc): self.touch()            # GC owns destruction
```

Integers (thin wrappers over the scalar-out getters):
`elec_num, nucl_num, point_num, walk_num, ao_num, mo_num, prim_num, shell_num`.

Arrays — every getter allocates a C-contiguous `float64` buffer of the stated
shape and returns it; setters accept any array-like via
`np.ascontiguousarray(x, dtype=np.float64)`:

| attribute | shape | C calls |
|---|---|---|
| `point` (get/set) | `(point_num, 3)` | `get_point`/`set_point`, `transp` inferred from `x.shape[-1] == 3` |
| `electron_coord` | `(walk_num, elec_num, 3)` | `get/set_electron_coord` |
| `nucl_coord` | `(nucl_num, 3)` | `get/set_nucleus_coord` |
| `ao_value` | `(point_num, ao_num)` | `get_ao_basis_ao_value` |
| `ao_vgl` | `(point_num, 5, ao_num)` | `get_ao_basis_ao_vgl` |
| `mo_value` | `(point_num, mo_num)` | `get_mo_basis_mo_value` |
| `mo_vgl` | `(point_num, 5, mo_num)` | `get_mo_basis_mo_vgl` |
| `ee_distance` | `(walk_num, elec_num, elec_num)` | `get_electron_ee_distance` |
| `en_distance` | `(point_num, nucl_num)` | `get_electron_en_distance` |
| `ee_potential` | `(walk_num,)` | `get_electron_ee_potential` |
| `en_potential` | `(walk_num,)` | `get_electron_en_potential` |

Implementation pattern — no shape table, no JSON:

```python
def _get(self, getter, shape):
    buf = np.empty(shape, dtype=np.float64)
    getter(self, buf)
    return buf

mo_value = property(lambda self: self._get(
    _core.get_mo_basis_mo_value, (self.point_num, self.mo_num)))
```

Setters: `set_point(coords)` derives `num = coords.shape[0]` and `transp` from
the layout; `set_nucleus_coord`, `set_electron_coord` likewise.

`trexio_read(path)` → `_core.trexio_read(self, str(path))`; raise
`NotImplementedError` when the symbol is absent (checked once at import).

Shapes above are verified against the getters' `size_max` checks and the org
argument tables in `org/qmckl_*.org` of the QMCkl sources.

### 7.3 `src/qmckl/errors.py`

`ERROR_CODES: dict[str, int]` and `error_name(code) -> str`, both from the
table generated into the extension. `ERROR_CODES` covers `QMCKL_SUCCESS`,
`QMCKL_INVALID_ARG_1..20`, `QMCKL_FAILURE`, `QMCKL_ERRNO`,
`QMCKL_INVALID_CONTEXT`, `QMCKL_ALLOCATION_FAILED`,
`QMCKL_DEALLOCATION_FAILED`, `QMCKL_NOT_PROVIDED`, `QMCKL_OUT_OF_BOUNDS`,
`QMCKL_ALREADY_SET`, `QMCKL_INVALID_EXIT_CODE`.

---

## 8. Tests

| file | asserts |
|---|---|
| `test_surface.py` | every name in `_core.pyi` is present in `dir(_core)`; ≥ 230 public callables; `get_mo_basis_mo_value` and `dgemm_safe` present; `qmckl.__all__` importable |
| `test_context.py` | create/GC; `Context.copy()` is independent (modify one, the other unchanged); `valid()`; `touch()` after `set_point` re-derives arrays; destroyed handle is not reachable |
| `test_errors.py` | undersized buffer on `get_ao_basis_ao_value` raises `QMCKLStatusError` whose message contains `QMCKL_INVALID_ARG_`; unprovided context yields `QMCKL_NOT_PROVIDED`; `qmckl.last_error(ctx)` returns a str |
| `test_blas.py` | `dgemm_safe` vs `A @ B` (all four `transa`/`transb`); `adjugate_safe` vs `np.linalg.inv(A) * det` for a small matrix |
| `test_distance.py` | `distance`, `distance_sq`, `distance_rescaled` vs NumPy, including transposed layouts |
| `test_trexio.py` | `ctx.trexio_read("tests/data/h2o_5z.h5")`; `elec_num == 10`, `nucl_num == 2`, `ao_num`/`mo_num` > 0; `mo_value` compared to reference numbers copied from the upstream C tests (`tests/test_qmckl_mo.c` in the QMCkl release tarball); whole module skipped when `qmckl.trexio_read` is absent |
| `test_shapes.py` | each property in §7.2 has the documented shape; set/get round-trip of `point` and `nucl_coord` |

`tests/data/h2o_5z.h5` is copied from `share/qmckl/test_data/` of the release
tarball and committed (4.6 MB) so the suite runs offline.

No test may depend on the user's QMCkl prefix beyond `pkg-config`.

---

## 9. CI

`.github/workflows/ci.yml`, matrix `ubuntu-latest` (py 3.10, 3.12) and
`macos-14` (py 3.12):

1. `micromamba create -p $PWD/env -c conda-forge trexio hdf5 gfortran`
   (TREXIO prebuilt on both OSes; `gfortran` satisfies QMCkl's `AC_PROG_FC`,
   which macOS runners lack).
2. Build QMCkl 1.1.0 from the release tarball:
   `./configure --prefix=$PWD/qmckl --without-openmp && make -j && make install`.
3. `export PKG_CONFIG_PATH=$PWD/qmckl/lib/pkgconfig:$PWD/env/lib/pkgconfig`
   plus the matching `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH`.
4. `pip install -e . --no-build-isolation` then `pytest -q`.
5. Cache `env/` and `qmckl/` on the pinned versions.
6. Extra job: same, but QMCkl built `--without-trexio`; must build, must pass,
   and `hasattr(qmckl, "trexio_read")` must be `False`.

---

## 10. Milestones and acceptance

| # | Deliverable | Accepted when |
|---|---|---|
| M0 | `pyproject.toml`, `CMakeLists.txt`, `README.md`, `LICENSE`, `.gitignore`, placeholder `src/qmckl/__init__.py` | **done, configure-verified**; `pip install -e .` now stops at `qmckl.cmake` missing / `csrc/generate.py` missing, never at pkg-config for a valid 1.x install |
| M1 | `generate.py` parser + classification + `--dump-json` | 239 functions classified, zero unclassified, JSON diffable |
| M2 | `context.hpp/.cpp`, `module.cpp`, exception | `test_context.py`, `test_errors.py` green |
| M3 | C++ + `.pyi` emitters, stub installed | `import qmckl` exposes the surface; `test_surface.py` green |
| M4 | §7 Python layer + `errors.py` table | `test_shapes.py` green |
| M5 | full test suite | green on macOS arm64 against QMCkl 1.1.0 |
| M6 | CI | green on ubuntu + macos, including the `--without-trexio` job |

---

## 11. Traps (also to be written in README)

1. **Version coupling** — bindings are generated from the installed header; a
   QMCkl upgrade requires reinstalling the package.
2. **Shared library only** — `qmckl.pc` does not propagate TREXIO, so
   `libqmckl.a` will not link.
3. **Raw layer** — a wrongly shaped array with sufficient size is silently
   reinterpreted (C semantics). Use the `Context` properties.
4. **TREXIO absence** — `qmckl.trexio_read` does not exist in such a build;
   `hasattr` is the documented check.
5. **`qmckl_string_of_error` is broken in 1.1.0 tarball builds**; the package
   uses its own table.
6. **OpenMP** — a QMCkl built with `--enable-openmp` spawns threads inside a
   Python call. Recommended build for Python use: `--without-openmp`, and
   thread in Python (the GIL is released around every call).
7. **`noconvert()`** — never remove it from array arguments; without it
   nanobind copies, and outputs are lost.

---

## 12. Out of scope for v1

Wheels and cibuildwheel; conda-forge recipe; docstrings generated from the org
sources; typed exception attributes (`.code`, `.function`); asynchronous or
threaded execution helpers; GPU/verificarlo.
