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
- `qmckl.h` declares **239** `qmckl_*` declarations but only **237 unique
  names**: `qmckl_get_jastrow_champ_grad` (line 2251 and 2256) and
  `qmckl_compute_ao_polynomial_hessian_doc` (line 1700 and 1716) appear twice,
  byte-identical, a tangling artifact of the release tarball. 223 return
  `qmckl_exit_code`; the other 16 are `bool` (5), `int32_t` (4),
  `qmckl_context` (3: create/copy/check), `void` (2: lock/unlock),
  `const char*` (1: `qmckl_string_of_error`) and `double` (1:
  `qmckl_get_numprec_epsilon`).
- `qmckl_test_precision_64` and `_32` take **no context** argument at all —
  the emitted wrapper must not assume `ctx` as the first parameter.
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
  the buffers are `QMCKL_MAX_FUN_LEN` (256) and `QMCKL_MAX_MSG_LEN` (1024),
  defined in the tarball's **private** `src/qmckl_error_private_type.h` and
  **not installed**, so `context.cpp` re-defines them behind `#ifndef`.
  `qmckl_get_error` **asserts** that the context is valid — guard with
  `qmckl_context_check()`.
- **Do not use `qmckl_string_of_error`**: in the v1.1.0 release tarball its
  `switch` is empty (always returns `"Unknown error"`). A locally tangled git
  build is correct. The package therefore owns its code→name table. (Upstream
  bug, worth reporting.)
- **The header disagrees with the `bind(C)` ABI on exactly two functions**,
  `qmckl_ao_gaussian_vgl` and `qmckl_ao_slater_vgl`. Their installed Fortran
  interface (`_install/include/qmckl_f.F90`) and their bodies
  (`src/qmckl_ao_f.F90`) agree with each other and disagree with `qmckl.h`:

  | parameter | `qmckl.h` | real ABI |
  |---|---|---|
  | `gaussian.n` | `const int64_t *` | `int64_t`, by value |
  | `gaussian.A` | `const int64_t *` | `const double *` (`real(c_double) :: A(n)`) |
  | `slater.num_slater` | `const int64_t *` | `int64_t`, by value |
  | both `VGL` | `const double *` | `double *` (`intent(out)`) |

  Measured on the installed 1.1.0 library: calling through the header
  prototype returns `QMCKL_INVALID_ARG_7` and writes nothing (the pointer's
  bit pattern is read as `n`, so `ldv < n` short-circuits); calling through
  the true prototype returns `QMCKL_SUCCESS` with the documented values. The
  reference `python/src/qmckl_wrap.c` calls both through the header, so SWIG
  is broken here too. Binding `A` as `const int64_t *` is the worst case: the
  call **succeeds** and QMCkl reads integers as doubles (`1` → `5e-324`), so
  every value is silently wrong. §5.5(e) overrides these two prototypes.

  Verified exhaustively: the `intent(out|inout)` and declared type of all 85
  `src/*.F90` definitions were compared against the header (no other
  const/value/type mismatch, no pointer argument with an unknown intent), and
  the 162 C definitions matched by name (zero type or const mismatch).
  Elsewhere the header can be trusted.

  `VGL` is a **column-major** `(ldv, 5)` buffer, like every multi-dimensional
  array in QMCkl (§2 BLAS): element `(i, j)` sits at `j * ldv + i`. A
  C-contiguous `np.zeros((5, ldv))` has exactly that layout, and `VGL[:, i]`
  is orbital `i`'s five components.
- QMCkl has **no BLAS/LAPACK dependency**; `qmckl_blas.c` is self-contained.
  TREXIO is the only external dependency (`HAVE_TREXIO`).
- The PyPI `trexio` wheel does **not** ship a linkable `libtrexio`; it cannot
  be used as a build dependency.
- nanobind 3.1.0: the C++ namespace is `nanobind`; `nb` no longer exists, so
  every TU spells `namespace nb = nanobind;` and the `nb::` code of this spec
  compiles verbatim. pybind11's `nb::register_exception<T>(...)` is gone; the
  equivalent is the `nb::exception<T>(m, "Name", PyExc_RuntimeError)`
  constructor, which creates the Python type and registers the translator.

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
│   ├── errors.py           # ERROR_CODES, error_name()
│   ├── py.typed            # empty, PEP 561 (the stub below is only honoured if present)
│   └── _core.pyi           # generated at build time, installed next to the .so
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
3. Detects whether the linked library was **built with TREXIO support** by
   concatenating the output of `nm -D` **and** `nm -g` on the resolved
   `libqmckl` and searching the union for `trexio_open`. The obvious
   candidate, `qmckl_trexio_read`, cannot be the predicate: `qmckl.h`
   declares it unconditionally and the 1.1.0 release tarball defines it even
   under `--without-trexio`, as a stub returning `QMCKL_FAILURE`
   ("QMCkl was compiled without TREXIO"). Measured counts: the no-TREXIO
   build's symbol table holds exactly one trexio symbol (`T
   _qmckl_trexio_read`, the stub) and `trexio_open` **zero** times; the
   with-TREXIO build holds 29 trexio symbols, `trexio_open` twice (once as
   the imported `U _trexio_open`, once inside `T _qmckl_trexio_open_X`).
   The search is a substring, not a `U` anchor, because a **statically
   linked** TREXIO appears as a defined `T` symbol; `qmckl_trexio_open_X`
   also matches but exists only under `HAVE_TREXIO`. Neither flag works
   everywhere (verified: on macOS `nm -D` fails with "File format has no
   dynamic symbol table" while `nm -g` finds the symbols; on a stripped
   Linux `.so` it is the other way round). `-DQMCKL_HAS_TREXIO=AUTO|ON|OFF`
   overrides; if neither `nm` run succeeds the configure fails and asks for
   the explicit flag rather than guessing. *Re-verified against real builds
   after the M6 predicate fix: the two §9 dry runs print
   `TREXIO I/O bound = ON` (with-TREXIO) and `= OFF` (no-TREXIO).*
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

- `len(functions) >= 230` declarations
- `"qmckl_dgemm_safe" in names` (this symbol marks the ≥ 1.1 API)
- every pointer parameter is classified by §5.3/§5.5 rules
- no Python keyword remains among argument names after suffixing with `_`
- the buckets partition the declarations exactly (sum == the number of
  declarations parsed; 239 on QMCkl 1.1.0), and any unclassified function is
  printed by name and makes the run exit 1
- the declaration count is **reported, not asserted**: a 1.1.x header that adds
  or de-duplicates a declaration must still build, and new declarations land in
  `generated` through §5.2. Drift in the hand-audited §5.5 tables is what is
  fatal (a table name absent from the header), because the classification of
  those names was audited by hand.

**Duplicates.** Two names are declared twice, byte-identically. The IR keeps
one record per declaration in header order, but the emitters (and the stub)
must **de-duplicate by name, first occurrence wins**; otherwise nanobind
registers two identical overloads and `dir(_core)` lies about the surface.

**No context.** `qmckl_test_precision_64/_32` take no context. The IR carries
`has_context`; the emitter must omit the context parameter, the `ctx.h`
argument and the `qmckl::check(ctx.h, ...)` call — those two functions return
an `int32_t` used directly.

### 5.2 Type mapping

| C parameter | stub annotation | C++ lambda parameter |
|---|---|---|
| `qmckl_context` (first arg) | `Context` | `qmckl::Context &ctx` |
| `int64_t`, `int32_t`, `int16_t`, `int` | `int` | same C type |
| `double`, `float` | `float` | same C type |
| `bool`, `_Bool` | `bool` | `bool` |
| `char` (`transp`, `transa`, `transb`, `basis_type`) | `str` | `char` |
| `const char *` | `str` | `std::string` |
| `T *` (non-const) | `numpy.typing.NDArray[T]` | `nb::ndarray<T, nb::c_contig>` |
| `const T *` | `numpy.typing.NDArray[T]` | `nb::ndarray<const T, nb::c_contig>` |

Return mapping: `qmckl_exit_code` → `None` (raise on non-zero); `bool` →
`bool`; `int32_t` → `int`; `double` → `float`; `const char *` → `str`; the 18
scalar-out functions of §5.5(a) return the pointee type (`int`/`float`) instead
of taking a buffer; the hand-written cases of §5.5(c) are not generated.

One correction the emitter applies on top of the table, driven by the table of
§5.5(e) (never by a name test inline in the emitter):

- **Header/ABI divergence** (§5.5(e)): for a table entry, the argument list the
  emitter builds the descriptor from is the override prototype, not the
  header's. Everything downstream — `python_name`, scalar vs array, dtype,
  const-ness, writability, `nb::arg`, the `.pyi` annotations and the call
  expression — then follows from that list by the ordinary §5.2–§5.4 rules, so
  there is one code path, not two. The only other difference is at the call
  site: `qmckl_<name>_abi(...)` instead of `::qmckl_<name>(...)`.
- The override is applied to the **emitted C++ and stub only**. The IR keeps
  what the header says; `--dump-json` changes only in `qmckl_check`'s bucket
  (§5.5c), which is why the M1 baseline is re-cut at M3.

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

- `size_max_X` → the parameter named `X` (or ending in `_X`): `X.size()`.
- bare `size_max` → the unique array parameter: `arr.size()`.
- `qmckl_trexio_read` → `std::strlen(file_name)` (the IR's `size_rule`, §5.4
  rule 2).

The cast is to the declared C type of the `size_max` parameter, which is not
always `int64_t` (§5.4 rule 2). QMCkl itself validates `size_max` and returns
`QMCKL_INVALID_ARG_*`, so an undersized buffer raises cleanly — never an
out-of-bounds write.

### 5.4 Emitted call

Every generated binding has exactly this shape (GIL released only around the C
call; string arguments already converted by nanobind):

```cpp
m.def("get_mo_basis_mo_value",
      [](qmckl::Context &ctx,
         nb::ndarray<double, nb::c_contig> mo_value) {
          qmckl_exit_code qmckl_rc;
          {
              nb::gil_scoped_release nogil;
              qmckl_rc = ::qmckl_get_mo_basis_mo_value(
                  ctx.h, mo_value.data(),
                  static_cast<int64_t>(mo_value.size()));
          }
          qmckl::check(ctx.h, qmckl_rc, "qmckl_get_mo_basis_mo_value");
      },
      "mo_value"_a.noconvert());
```

The only local is `qmckl_rc`: the C API has parameters literally named `n`,
`m`, `k`, `a`, `b`, `c`, `value`, `size_max`, so a bare local name could
collide with some function's parameter. Pointers and sizes are inlined —
`.data()` and `.size()` are pure reads of the nanobind view struct and are safe
inside the released region.

All `m.def`s are emitted inside `void qmckl_python::register_generated(nb::module_ &m)`;
`csrc/module.cpp` declares and calls it. No docstrings in v1. The translation
unit gains `<nanobind/ndarray.h>`, `<nanobind/stl/string.h>` and `<cstring>`
(the `trexio_read` `strlen`) over M2's includes, plus
`using namespace nb::literals;` for `_a`.

Rules, all mechanical — the emitter loops over the IR and never special-cases a
name inline (only the §5.5 tables do):

1. **Signature.** The Python parameter list is the C one minus every
   `size_max*` parameter, in header order. The context, when present, is first
   (verified for every declaration). `nb::arg` names are the IR's `python_name`s, which equal the C
   names on 1.1.0: none is a Python keyword, none a C++ keyword, and none
   repeats inside a function. Assert all three before emitting, and use the same
   names in the stub so keyword calls follow the header.
2. **`size_max`.** The value comes from the IR: the parameter named by
   `size_target`, cast to the declared type of the `size_max` parameter —
   `static_cast<int64_t>` everywhere except `int64_t const` in
   `qmckl_get_jastrow_champ_ee_distance_rescaled` and `const size_t` in
   `qmckl_get_en_rescaled_single_gl`. The IR's `size_rule` says how:
   `"strlen"` → `std::strlen(file_name.c_str())` (only `qmckl_trexio_read`),
   anything else → `.size()`. A `size_max` with no `size_target`, or a
   `size_rule` the emitter does not know, is a generator bug → exit 1. Never
   take a `size_max` from Python, and never derive one from `lda`/`ldc`.
3. **`.noconvert()` on `nb::ndarray` arguments only.** Scalars keep nanobind's
   default conversion: `float` accepts an `int`, `char` accepts a 1-character
   `str` (and an `int`, which QMCkl then rejects with a clean status).
4. **Return class** — exactly one of four, chosen from the IR. On 1.1.0 the
   223 bindings split 195 exit-code-with-context / 18 scalar-out / 8
   value-with-context / 2 value-without-context:
   - *exit code, context present* (195 functions): as the template above.
   - *exit code, no context*: `qmckl::check(QMCKL_NULL_CONTEXT, rc, "qmckl_NAME")`
     — `check` skips the `qmckl_get_error` read when the context is NULL (§6.1).
     No such function exists on 1.1.0; keep the branch so a future header cannot
     produce an unchecked call.
   - *value return*: `bool` (the five `*_provided`), `int32_t`
     (`get_numprec_precision`, `get_numprec_range`, `test_precision_32`,
     `test_precision_64`) and `double` (`get_numprec_epsilon`) — call and
     `return` the C value, no `check()`. Drop the context parameter when the IR
     says `has_context` is false (the two `test_precision_*`).
   - *scalar out* (§5.5(a), 18 functions): the pointer parameter disappears from
     the Python signature; declare a local of the pointee type, pass its
     address, `check()`, `return` it.

```cpp
m.def("get_point_num",
      [](qmckl::Context &ctx) {
          int64_t qmckl_value{};
          qmckl_exit_code qmckl_rc;
          {
              nb::gil_scoped_release nogil;
              qmckl_rc = ::qmckl_get_point_num(ctx.h, &qmckl_value);
          }
          qmckl::check(ctx.h, qmckl_rc, "qmckl_get_point_num");
          return qmckl_value;
      });
```

5. **De-duplicate by name, first occurrence wins** (§5.1): the 225 emitted
   declarations collapse to **223** bindings. The two repeated declarations are
   `qmckl_get_jastrow_champ_grad` and `qmckl_compute_ao_polynomial_hessian_doc`
   (§2); neither is a §5.5 table entry, so the de-duplication and the override
   cannot interact.
6. **Nothing else is scalarised.** The `INOUT_SCALAR_POINTER` table
   (`int64_t *n` in the polynomial/hessian functions, `double *det_l` in
   `qmckl_adjugate*`) documents the dump only: §5.2 binds a non-const `T *` to a
   one-element writable array in both directions (`np.zeros(1, dtype=np.int64)`).
   Reproducing the C signature exactly is worth more than a prettier return.
   The only scalars beyond the header are the two §5.5(e) ABI overrides.
7. **Column-major stays column-major.** QMCkl stores every multi-dimensional
   array column-major (§2), including the `qmckl_dgemm_safe` operands, and
   documents it (`org/qmckl_blas.org`). The emitted binding is a faithful
   passthrough: a C-contiguous array is interpreted column-major, exactly as
   the C API says. **Do not swap operands or transpose inside the generated
   layer** — that would silently contradict the documented API and make the
   documented behaviour unreachable. The row-major convenience is a §7 wrapper.
8. **One descriptor, two emitters.** The C++ emitter and the `.pyi` emitter read
   the same per-function descriptor (order, names, types, return class), so the
   signature cannot drift between them; `test_surface.py` compares the two
   surfaces against each other and against `dir(_core)`. A `const char *`
   argument is received as `std::string`, converted by nanobind while the GIL is
   still held; `c_str()` inside the released region is then a pure read.

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
qmckl_string_of_error                              -> not exposed (see §2: the
     released 1.1.0 tarball ships it with an empty switch and it returns
     "Unknown error" for every code; qmckl.errors.error_name() is the
     replacement and is built from the header macros)
qmckl_check                                        -> not exposed (see §2-adjacent
     note below: it prints, and it returns the wrong code)
```

`qmckl_check(context, rc)` is unusable as a Python API, for three independent
reasons (all verified in `src/qmckl_error.c`):

1. it prints `qmckl_string_of_error(rc)` to stderr, which always says
   `"Unknown error"` on 1.1.0;
2. it passes a **non-NUL-terminated, otherwise-uninitialised**
   `char message[QMCKL_MAX_MSG_LEN]` to `fprintf("%s", …)`; when the context
   carries no stored error, `qmckl_get_error` `memcpy`s zero bytes and the
   stack buffer is printed as-is (uninitialised read, possible overread);
3. it passes its own `rc` argument as the `exit_code` **out**-parameter of
   `qmckl_get_error`, which overwrites it with the context's stored code, so
   it can return `QMCKL_SUCCESS` for a non-zero `rc` — an error silently
   turned into success.

Every generated exit-code function already raises through `qmckl::check`
(§6.1), which reads the context only when the stored code matches and formats
the message itself, so nothing in the package needs the C function. Upstream
bug, worth reporting alongside `qmckl_string_of_error`.

Counts as measured by the M1 parser (QMCkl 1.1.0, TREXIO on): generated 204,
scalar_out 18, array_no_size_max 4, handwritten 7, excluded 5, trexio 1 →
239. Two moves to the excluded set, `qmckl_string_of_error` and
`qmckl_check`, give generated **202**, excluded **7**, **225 emitted
declarations**, and **223 distinct generated bindings** after de-duplicating
the two repeated declarations (222 when `qmckl_trexio_read` is dropped, see
(d)). The generator must print this partition and exit non-zero if the buckets
do not sum to the declaration count. `qmckl_check` leaving `generated` is the
only change `--dump-json` sees relative to the M1 baseline, so the M1
acceptance dump is re-cut at M3 and the byte-identical comparison runs against
the re-cut file.

**(d) TREXIO** — when the linked library was built without TREXIO support
(§4.2 item 3: the probe looks for `trexio_open`), skip that binding entirely
(do not emit a stub for it either).

**(e) Header/ABI divergence — two functions.** The header is wrong for
`qmckl_ao_gaussian_vgl` and `qmckl_ao_slater_vgl` (§2). The generator carries
their **true `bind(C)` prototype**, and emits both the Python signature and the
call from it:

```python
ABI_OVERRIDE_TABLE = {
    "qmckl_ao_gaussian_vgl": (
        ("context",      "qmckl_context"),
        ("X",            "const double *"),
        ("R",            "const double *"),
        ("n",            "int64_t"),            # header: const int64_t *
        ("A",            "const double *"),     # header: const int64_t *
        ("VGL",          "double *"),           # header: const double *
        ("ldv",          "int64_t"),
    ),
    "qmckl_ao_slater_vgl": (
        ("context",      "qmckl_context"),
        ("X",            "const double *"),
        ("R",            "const double *"),
        ("num_slater",   "int64_t"),            # header: const int64_t *
        ("N",            "const int64_t *"),
        ("A",            "const double *"),
        ("VGL",          "double *"),           # header: const double *
        ("ldv",          "int64_t"),
    ),
}
```

The emitter emits, at file scope in an anonymous namespace, a typedef of that
prototype and a `reinterpret_cast`-initialised pointer, then calls through it:

```cpp
namespace {
using qmckl_ao_gaussian_vgl_abi_t = qmckl_exit_code (*)(
    qmckl_context, const double *, const double *, int64_t,
    const double *, double *, int64_t);
qmckl_ao_gaussian_vgl_abi_t const qmckl_ao_gaussian_vgl_abi =
    reinterpret_cast<qmckl_ao_gaussian_vgl_abi_t>(&::qmckl_ao_gaussian_vgl);
}
```

Not `constexpr`: `reinterpret_cast` of a function pointer is not a constant
expression. If the compiler emits `-Wcast-function-type` (GCC `-Wextra` does),
cast through a `std::memcpy` bit-copy of the function pointer instead of
suppressing the warning; the M3 acceptance requires zero warnings outside
nanobind.

From the prototype the ordinary §5.2 mapping follows mechanically: the `int64_t`
parameters become Python `int`, `const double *`/`const int64_t *` become
read-only arrays, and the non-const `double *VGL` becomes a **writable**
`nb::ndarray<double, nb::c_contig>` — so nanobind rejects a read-only buffer with
`TypeError` instead of letting QMCkl scribble on `np.frombuffer(b"...")` memory.

The override replaces **parameter types only**. Because the names and their
order are identical (guarded below), the IR's `has_context`, return class and
`size_target` bindings still apply unchanged, and `.pyi` keyword names stay the
header's. The type strings use exactly the grammar the header parser already
accepts (`qmckl_context`, `int64_t`, `const double *`, …): reuse the parser's
type classifier to build a synthetic argument list, do not hand-roll a second
type mapping.

Fatal, in both directions: the function must be in the header; the override must
have the same number of parameters, with the **same names in the same order** as
the header (so the table cannot rot into a different function's signature); and
at least one parameter must actually differ from the header (an override that
matches the header is stale and exits 1). Applied to the emitted C++/stub only —
the IR and `--dump-json` keep the header's truth.

Do **not** add the `_hpc` variants: their `VGL` is already `double *` and their
other parameters match the header (verified in `src/qmckl_ao.c`).

`VGL` is column-major (§2): the caller passes a buffer of at least `5 * ldv`
elements; a C-contiguous `np.zeros((5, ldv))` is the natural choice, and
`VGL[:, i]` is orbital `i`.

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

The generated part holds one `def` per binding — same names, order and keyword
names as `register_generated`, after the same de-duplication — preceded by the
imports the annotations need:

```python
import numpy as np
from numpy.typing import NDArray
```

Annotations: context → `_Context`; array → `NDArray[np.float64]`,
`NDArray[np.int64]` or `NDArray[np.int32]` from the pointee base type; `char`
and `const char *` → `str`; integer → `int`; `double`/`float` → `float`;
`bool` → `bool`. Return: `None` for the exit-code class, the value's own type
for the value class, the pointee type for scalar-out. Every parameter is
positional-or-keyword, context first.

The emitter runs `ast.parse()` on the finished text and exits 1 if the stub is
not valid Python: an unparsable stub silently degrades every annotation to
`Any`, which is worse than a failed build.

The emitter owns the **whole** file: the fixed part above is template text in
`generate.py`, not a checked-in stub concatenated by CMake. M2 emits it plus
the `BEGIN/END GENERATED BINDINGS` markers, M3 fills the markers.

### 5.7 Error-code table (emitted in M2, complete at M2)

The code names come from the header, never from a checked-in copy and never
from `qmckl_string_of_error` (§2). A second, independent pass over the **raw**
header text (do not reuse the stripped text of §5.1) scans

```
#define  QMCKL_SUCCESS                  ((qmckl_exit_code) 0)
#define  QMCKL_INVALID_ARG_1            ((qmckl_exit_code) 1)
```

i.e. `^\s*#\s*define\s+(QMCKL_[A-Z0-9_]+)\s+\(\s*\(\s*qmckl_exit_code\s*\)\s*(-?\d+)\s*\)\s*$`.
Note the two spaces after `#define` and the mandatory `((qmckl_exit_code) N)`
cast: a bare `#define QMCKL_X 3` does not occur in 1.1.0 and must not be
guessed at. Assert at least 30 codes, `QMCKL_SUCCESS` present with value 0, no
duplicate name and no duplicate value, and `QMCKL_INVALID_ARG_20` present (that
marks the 1.1 API). Failure = exit 1.

Two emissions, from that one table:

1. `const char *qmckl::code_name(int32_t rc)` in the generated translation unit
   (declared in `context.hpp`): a `switch` over the values, returning the name
   as a string literal; anything else returns `"QMCKL_UNKNOWN"`. Values are
   emitted as bare integers (`case 103:`), not as macro names, so the generated
   file does not depend on macros that a future header could rename.
2. `void qmckl_python::register_error_codes(nb::module_ &m)` in the same file,
   setting `m.attr("ERROR_CODES")` to a `nb::dict` of name → int. `errors.py`
   builds `error_name()` by inverting it.

Both are emitted in M2 together with the skeleton of §10; M3 adds the per
function bindings and does not touch them.

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

void check(int64_t ctx, int32_t rc, const char *fname);   // returns iff rc == 0
const char *code_name(int32_t rc);    // emitted from the header macros, §5.7
std::string last_error_message(int64_t ctx);  // shared with the last_error binding
}
```

`check()` must **not** be declared `[[noreturn]]`: the `rc == 0` path returns
normally, and a `[[noreturn]]` function that returns is undefined behaviour.

`check(ctx, rc, fname)` behaviour:

- `rc == 0` → return.
- otherwise build `"[<CODE_NAME>] <fname>: <message>"` and throw
  `QMCKLStatusError`.
- `<CODE_NAME>` comes from the generated `code_name(rc)` table; it never fails.
- `<message>` comes from `qmckl_get_error(ctx, &stored, fun, msg)` **only when
  `qmckl_context_check(ctx) != QMCKL_NULL_CONTEXT`** (the C function asserts on
  an invalid context) **and `stored == rc`**. The `stored == rc` test matters:
  QMCkl stores the last error per context, and several functions return
  `QMCKL_NULL_CONTEXT` or a `provide_*` failure **without** writing it, so an
  unchecked read would print a stale message from an unrelated earlier call.
  When the two disagree, print the code name alone.
- Suppress the message when it is empty **or exactly `"Unknown error"`**: that
  is the string `qmckl_string_of_error` produces for every code in the released
  tarball, and `qmckl_failwith(..., NULL)` stores it verbatim. So
  `qmckl_get_ao_basis_type` on a fresh context must read
  `[QMCKL_NOT_PROVIDED] qmckl_get_ao_basis_type`, not `… : Unknown error`.
  Resulting formats, both acceptable:
  `"[CODE] fname: message"` and `"[CODE] fname"`.
- Buffer sizes are not optional. `qmckl_get_error` `memcpy`s up to
  `QMCKL_MAX_FUN_LEN` bytes (256) and `QMCKL_MAX_MSG_LEN` bytes (1024) **and
  never writes a NUL terminator**. Use `char fun[QMCKL_MAX_FUN_LEN + 1]{};`
  and `char msg[QMCKL_MAX_MSG_LEN + 1]{};` — zero-initialised, one byte of
  slack. Do not shrink them to "what the message looks like".
- Those two macros are **not** in the installed `qmckl.h` (§2): define them in
  `context.cpp` behind `#ifndef` with the verified 256/1024 values.
- No GIL concern: `check()` runs with the GIL re-acquired (the release scope
  closes before it is called).

`code_name` is emitted next to the generated bindings from the
`#define QMCKL_*` macros of the header (single source of truth). Do **not** call
`qmckl_string_of_error` (see §2).

### 6.2 `csrc/module.cpp`

```cpp
namespace nb = nanobind;   // nanobind 3.x renamed nb -> nanobind (§2)

NB_MODULE(_core, m) {
    nb::exception<qmckl::QMCKLStatusError>(
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

    m.def("last_error", ...);            // see below
    m.def("get_ao_basis_type", ...);     // char buffer -> str

    qmckl_python::register_error_codes(m);   // _core.ERROR_CODES, §5.7
    qmckl_python::register_generated(m);
}
```

`last_error(ctx) -> str` does **not** call `::qmckl_last_error`. That C
function formats its text with `qmckl_string_of_error(last_rc)`, which is the
broken empty switch of §2, so every message would read `Error -- Unknown
error -- in …`. Compose it instead from `qmckl_get_error` (same 257/1025
zero-initialised buffers as `check()`) and `code_name()`:
`"[<CODE_NAME>] <function>: <message>"`, and return `"Null context"` when
`qmckl_context_check(ctx) == QMCKL_NULL_CONTEXT` (matching the C wording).

Order matters: the exception and `_Context` must be registered before the
generated functions reference them in signatures.

---

## 7. Python layer

### 7.1 `src/qmckl/__init__.py`

```python
from . import _core
from ._core import *              # generated functions, QMCKLStatusError, ERROR_CODES
from .context import Context
from .errors import ERROR_CODES, error_name
from .blas import dgemm

# One source of truth: every public name of the extension, plus the Pythonic
# layer. `import *` skips `_Context` (leading underscore), which is intended:
# users construct `Context`.
__all__ = sorted({"Context", "error_name"}
                 | {n for n in dir(_core) if not n.startswith("_")})
```

`Context` subclasses `_core._Context`; generated functions accept it because the
parameter type is `qmckl::Context &`.

`dgemm` is the row-major wrapper of §7.2b; it deliberately shadows the
generated `_core.dgemm`, which is already a public name of `_core`, so
`__all__` is unchanged and the raw column-major calls remain at
`qmckl._core.dgemm` / `qmckl._core.dgemm_safe`.

The package ships an empty `src/qmckl/py.typed` (PEP 561). Without it, type
checkers ignore the installed `_core.pyi` and every generated function is
`Any`. `wheel.packages = ["src/qmckl"]` picks it up automatically.

### 7.2 `src/qmckl/context.py`

The class below, plus the properties and setters of this section.  Shapes are
validated here, because the raw layer cannot validate them (§11, trap 3).

```python
class Context(_core._Context):
    @classmethod
    def copy(cls, other): return cls(other)          # deep copy

    def __enter__(self): return self
    def __exit__(self, *exc): self.touch()            # GC owns destruction

    def trexio_read(self, path): ...                  # see below
```

Integers (thin wrappers over the scalar-out getters):
`elec_num, nucl_num, point_num, walk_num, ao_num, mo_num, prim_num, shell_num`.

Arrays — every getter allocates a C-contiguous `float64` buffer of the stated
shape and returns it; setters accept any array-like via
`np.ascontiguousarray(x, dtype=np.float64)`:

| attribute | shape | C calls |
|---|---|---|
| `point` (get/set) | `(point_num, 3)` | `get_point`/`set_point` with `transp='N'` |
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

def _eval(self, getter, shape):
    if shape[0] == 0:
        raise ValueError("no points in the context: set ctx.point (or "
                         "ctx.electron_coord) before evaluating the basis")
    return self._get(getter, shape)
```

`_eval` is used by the four point-evaluated properties (`ao_value`, `ao_vgl`,
`mo_value`, `mo_vgl`) and refuses `point_num == 0`: with no points the
provider allocates zero bytes, which corrupts QMCkl 1.1.0's allocation
bookkeeping and makes an assert-enabled `qmckl_context_destroy` abort (§11,
trap 9).  Every other array property uses `_get` directly.

Setters: each coordinate setter runs the input through `_coords(x, ndim)` —
`np.ascontiguousarray(x, dtype=np.float64)` plus a rank and last-axis-3 check,
`ValueError` otherwise — and passes `transp='N'` (verified:
`qmckl_get_point`/`qmckl_set_point` with `'N'` are the `(num, 3)` row-major
layouts — internally QMCkl stores a 3×num matrix and transposes for `'N'`;
the `'T'` layout stays available in the raw layer only).

- `point` — `num = coords.shape[0]` → `set_point(ctx, 'N', num, coords)`.
- `nucl_coord` — derives `n = coords.shape[0]`; calls
  `qmckl_set_nucleus_num(ctx, n)` only when `qmckl_get_nucleus_num` reports
  `QMCKL_NOT_PROVIDED` (a fresh context), and raises `ValueError` when the
  stored count differs from `n` — `qmckl_set_nucleus_coord` takes no num and
  would truncate silently otherwise.
- `electron_coord` — requires `elec_num` to be set and `shape[1] == elec_num`
  (else `ValueError`).  Note that `qmckl_set_electron_coord` calls
  `qmckl_set_point`, so it **replaces the point set**: afterwards
  `point_num == walk_num * elec_num`.

`trexio_read(path)` is a method that calls `_core.trexio_read(self, str(path))`
and raises `NotImplementedError` when `_core.trexio_read` is absent (checked
once at import, §5.5(d)).

Shapes above are verified against the getters' `size_max` checks and the org
argument tables in `org/qmckl_*.org` of the QMCkl sources.

### 7.2b `src/qmckl/blas.py` — the one row-major convenience

QMCkl's `qmckl_dgemm_safe` is a passthrough to column-major BLAS and NumPy
arrays are row-major, so the generated binding cannot be called with ordinary
arrays without building column-major buffers (§5.4 rule 7). M4 adds exactly one
wrapper, and nothing else:

```python
def dgemm(ctx, A, B, *, transa="N", transb="N", alpha=1.0):
    """alpha * op(A) @ op(B) with row-major (NumPy) arrays; returns C."""
```

- `op(X)` is `X` for `"N"` and `X.T` for `"T"`; `opA` is `(m, k)`, `opB` is
  `(k, n)`, the result is `(m, n)`. A shape mismatch raises `ValueError`, and
  so does a flag other than `'N'`/`'T'` (either case is accepted).
  `m`, `n` or `k` equal to zero also raises `ValueError`: a zero-row NumPy
  matmul is legal, but `qmckl_dgemm_safe` rejects non-positive dimensions
  with `QMCKL_INVALID_ARG_*`.
- `qmckl.dgemm` shadows the generated `_core.dgemm`, which is already a
  public name of `_core`; `__all__` is therefore unchanged, and the raw
  column-major calls remain at `qmckl._core.dgemm` /
  `qmckl._core.dgemm_safe`.
- Column-major buffers, packed (`ld` = number of rows of the stored matrix):
  `bufA = np.ascontiguousarray(opA.T)` is `(k, m)` with `lda = m`,
  `bufB = np.ascontiguousarray(opB.T)` is `(n, k)` with `ldb = k`, and
  `bufC = np.zeros((n, m))` is the column-major result with `ldc = m`. The
  call is then plain `"N", "N"` — the transpose is done by NumPy, never by
  flag juggling:

  ```python
  _core.dgemm_safe(ctx, "N", "N", m, n, k, alpha,
                   bufA, m, bufB, k, 0.0, bufC, m)
  return bufC.T.copy()
  ```

- This is the *only* place in the package that transposes, and `test_blas.py`
  checks it against `A @ B` for all four `transa`/`transb` (§8). `lda`, `ldb`,
  `ldc`, `beta` and the `size_max` values are never user-visible.
- Do not implement this by swapping the operands of the C call. A C-contiguous
  `(m, k)` buffer *is* a column-major `(k, m)` matrix only if its leading
  dimension is read as `k`, its row length. Swapping operands leaves the
  caller's `lda` untouched, so any `lda != k` silently multiplies the wrong
  elements — and nothing rejects it: `qmckl_dgemm_safe` validates only
  `size_* > 0` and `ld* > 0`, never `lda >= m` (measured in
  `src/qmckl_blas_f.F90`). Reinterpreting `lda` inside the generated layer
  would contradict the documented C API instead. Packed buffers need no such
  reinterpretation.

### 7.3 `src/qmckl/errors.py`

`ERROR_CODES: dict[str, int]` and `error_name(code) -> str`, both from the
table generated into the extension (§5.7): `from ._core import ERROR_CODES`
and invert it for `error_name`. Nothing is retyped by hand; the header is the
single source of truth. `ERROR_CODES` covers `QMCKL_SUCCESS`,
`QMCKL_INVALID_ARG_1..20`, `QMCKL_FAILURE`, `QMCKL_ERRNO`,
`QMCKL_INVALID_CONTEXT`, `QMCKL_ALLOCATION_FAILED`,
`QMCKL_DEALLOCATION_FAILED`, `QMCKL_NOT_PROVIDED`, `QMCKL_OUT_OF_BOUNDS`,
`QMCKL_ALREADY_SET`, `QMCKL_INVALID_EXIT_CODE`.

---

## 8. Tests

| file | asserts |
|---|---|
| `test_surface.py` | **M3**: every top-level name of `_core.pyi` is in `dir(_core)`; the number of generated `def`s in the stub equals the number of generated names in `dir(_core)` and is ≥ 220 (223 with TREXIO, 222 without); `get_mo_basis_mo_value`, `dgemm_safe`, `test_precision_64`, `get_point_num`, `ao_gaussian_vgl` present; `trexio_read` present exactly when the library exports it; `string_of_error`, `get_error`, `check`, `failwith`, `lock` **absent** (deliberate, §5.5c); `_Context` and `ERROR_CODES` present; `qmckl.__all__` equals the public names |
| `test_generated.py` | **M3**: `test_precision_64(1.0, 1.0) >= 53`; `dgemm_safe` is a **faithful column-major passthrough** (§5.4 rule 7) for one non-square and one square shape in all four `transa`/`transb` (arrays, `size_max`, `char` args, GIL release): given row-major `A` of shape `(k, m)` when `transa='T'` and `(m, k)` otherwise, buffers are `bufA = np.ascontiguousarray(A.T)`, `bufB = np.ascontiguousarray(B.T)`, `bufC = np.zeros((n, m))`, leading dimensions `bufA.shape[1]`, `bufB.shape[1]`, `m`, expected `bufC.T == op(A) @ op(B)`; payloads are `[[1,2],[3,4],[5,6]]`-class values so that a transposed reading cannot pass. `get_point_num` on a fresh context returns `0` — its C body has no `NOT_PROVIDED` check, it just copies `point.num` — while `get_ao_basis_shell_num`, also scalar-out, raises `QMCKLStatusError` starting `[QMCKL_NOT_PROVIDED]`; `ao_gaussian_vgl` with `A = np.array([1.0, 2.0])`, `X = (0.5,0,0)`, `R = 0` gives the documented values `VGL[0,0] = exp(-0.25)`, `VGL[0,1] = exp(-0.5)`, `VGL[1,0] = -exp(-0.25)`, and the Laplacians `VGL[4,0] = (4*1**2*0.25 - 6*1)*exp(-0.25) = -5*exp(-0.25)`, `VGL[4,1] = (4*2**2*0.25 - 6*2)*exp(-0.5) = -8*exp(-0.5)`, in a C-contiguous `np.zeros((5, ldv))` whose flat memory is exactly the Fortran `VGL(ldv,5)` column-major layout — this fails if `A` is bound as an integer array (§5.5e). Write the Laplacian as `(4*alpha**2*r2 - 6*alpha)*exp(-alpha*r2)`: the α² and α factors are not decoration, and dropping them happens to go unnoticed at α=1; a read-only `VGL` is refused with `TypeError`; a `float32` array is refused (`noconvert`) |
| `test_context.py` | **M2**: create, `valid()`, `handle`, `touch()`, GC (no crash, no leak report), `Context.copy()` returns a distinct valid context (`copy.handle != ctx.handle`); **M4**: deep independence via a setter (`set_point` then read back); the `touch()` re-derivation check lives in `test_shapes.py` |
| `test_errors.py` | **M2**: `get_ao_basis_type` on a fresh context raises `QMCKLStatusError` whose message starts `[QMCKL_NOT_PROVIDED]` and does **not** contain `Unknown error`; `ERROR_CODES["QMCKL_SUCCESS"] == 0`; `error_name(103) == "QMCKL_INVALID_CONTEXT"`; `last_error(ctx)` returns a `str`; **M3+**: undersized buffer on `get_ao_basis_ao_value` raises with `QMCKL_INVALID_ARG_` |
| `test_blas.py` | **M4**: `qmckl.dgemm(ctx, A, B, transa=…, transb=…)` (§7.2b) equals `alpha * op(A) @ op(B)` for all four combinations, with the *inputs* built per combination (op(A) = (3, 2), op(B) = (2, 4)) so a transposed reading of either buffer cannot pass; `ValueError` on shape mismatch, bad flag and empty operands; `adjugate_safe` vs `np.linalg.inv(A) * det` for a small matrix |
| `test_distance.py` | `distance_sq`, `distance` and `distance_rescaled` against NumPy — `C[i,j] = |A_i - B_j|^2`, `|A_i - B_j|`, `(1 - exp(-kappa*r_ij))/kappa` with `kappa = 2.5` — for non-square `m != n`, in all four `transa`/`transb` combinations. `'N'` takes `A` as C-contiguous `(m, 3)` with `lda = 3`; `'T'` takes `A` as `(3, m)` with `lda = m` (verified in `src/qmckl_distance_f.F90`: `'T'` indexes `A(i,1..3)`, i.e. x's contiguous, then y's, then z's). The `C` buffer is C-contiguous `(n, m)` with `ldc = m`, read back as `.T` — the same convention as `dgemm`. A bad flag, `m == 0` or `lda < 3` raise `QMCKLStatusError` with `QMCKL_INVALID_ARG_*`. `n` is pinned to 3 and `m` to 4: trap 10 makes `n == 3` the only size accepted in all four flag combinations by `qmckl_distance` |
| `test_trexio.py` | `ctx.trexio_read("tests/data/h2o_5z.h5")`; `elec_num == 10`, `nucl_num == 3`, nuclear charges `[8, 1, 1]`, `ao_num == 266`, `mo_num == 201`; the two cross-provider identities `mo_vgl[:, 0, :] == mo_value` and `ao_vgl[:, 0, :] == ao_value` (measured exactly 0.0 — this is what upstream's `tests/test_qmckl_mo.c` asserts; that file carries **no** hardcoded numbers, only such identities and a rescale check); `en_potential[w] == sum_{i in w} sum_k -Z_k/r_ik` and `ee_potential[w] == sum_{i<j} 1/r_ij` recomputed in NumPy from `nucl_coord`/`nucleus_charge` and from a 2-walker `electron_coord` (measured to 1.4e-14 and 7.1e-15); whole module skipped when `qmckl.trexio_read` is absent |
| `test_shapes.py` | each property of §7.2 has the documented shape; set/get round-trip of `point`, `nucl_coord`, `electron_coord`; wrong shapes raise `ValueError`; `touch()` recomputes an unchanged distance; the four point-evaluated properties raise `ValueError` with no points; the basis-dependent ones use `tests/data/h2o_5z.h5` and skip without TREXIO |

`tests/data/h2o_5z.h5` is copied from `share/qmckl/test_data/` of the release
tarball and committed (4.6 MB) so the suite runs offline.

No test may depend on the user's QMCkl prefix beyond `pkg-config`.

---

## 9. CI

`.github/workflows/ci.yml` (written and dry-run-validated 2026-11-20 on the
QMCkl 1.1.0 release tarball, macOS arm64). One `test` job, `fail-fast:
false`, `QMCKL_VERSION` and the tarball's sha256 pinned as job env.

1. Four matrix cells: `ubuntu-latest` py3.10 and py3.12 with TREXIO,
   `macos-latest` py3.12 with TREXIO, and one `ubuntu-latest` py3.12
   `--without-trexio` job (it carries `qmckl_flags: --without-trexio` and
   `trexio: "false"`). The macOS cell was pinned to `macos-14` until the
   first CI run: macOS 14 was retired 2026-11-02 (deprecation began
   2026-07-06), so the cell moved to `macos-latest` — a pinned label is
   exactly what expired.
2. `mamba-org/setup-micromamba@v3` builds `qmckl-env` from conda-forge with
   `trexio hdf5 gfortran openblas` (conda-forge has `osx-arm64` TREXIO
   builds; `gfortran` supplies the Fortran compiler, since QMCkl always
   compiles its `.F90` sources and macOS runners ship none — the build step
   points `FC` at `$ENV_PREFIX/bin/gfortran`, which `AC_PROG_FC` honours;
   `openblas` supplies BLAS and LAPACK, paragraph below). The environment
   is cached (`cache-environment: true`) and its prefix is read from the
   action's `environment-path` output.
3. The release tarball is downloaded with `curl` and verified against the
   pinned sha256 (`sha256sum -c -` on Linux, `shasum -a 256 -c -` on macOS),
   then configured with `--prefix=$PWD/qmckl --without-openmp`
   (§11 trap 6), `--with-blas="-L$ENV_PREFIX/lib -lopenblas"` and
   `--with-lapack="-L$ENV_PREFIX/lib -lopenblas"` (paragraph below), and
   `--with-trexio=$ENV_PREFIX`. The directory form is what `configure.ac`
   turns into `-I$dir/include -L$dir/lib -ltrexio` directly, so the conda
   TREXIO package needs no `.pc` file. The `qmckl/` prefix is cached with
   `actions/cache@v6` keyed on
   `qmckl-<version>-<runner.os>-<runner.arch>-trexio-<flag>`; on a hit the
   build step is skipped.
4. `make -j"$(getconf _NPROCESSORS_ONLN)"` and `make install`. No race on
   generated sources was measured (the tarball ships already-tangled C).
5. A step appends to `$GITHUB_ENV`: `PKG_CONFIG_PATH` (qmckl prefix first,
   then `$ENV_PREFIX/lib/pkgconfig`) and, with the same two prefixes joined
   by `:`, `LD_LIBRARY_PATH` and `DYLD_LIBRARY_PATH` — `qmckl.pc` does not
   propagate TREXIO (§11 trap 2), so the loader must be told where both
   libraries live. The same step exports `PKG_CONFIG_PATH` locally (variables
   from `$GITHUB_ENV` are only visible to later steps) and runs
   `pkg-config --modversion qmckl` so a bad prefix fails immediately with a
   clear message.
6. `pip install --upgrade pip`, then
   `pip install nanobind scikit-build-core cmake ninja numpy pytest` (the
   build toolchain comes from the same line on every runner), then
   `pip install -e . --no-build-isolation`; a one-line presence check
   asserts `hasattr(qmckl, "trexio_read") is (EXPECT_TREXIO == "true")`
   with `EXPECT_TREXIO` from the matrix; finally `pytest -q`. The
   `--without-trexio` job runs the same pipeline against a QMCkl built
   without TREXIO, where the CMake `nm` probe (§4) binds `trexio_read`
   only if the library was built with TREXIO support (the `trexio_open`
   probe, §4.2 item 3).

**BLAS and LAPACK.** QMCkl needs both, and a bare runner supplies
neither: `m4/ax_blas.m4` has a `-framework vecLib` branch, but
`m4/ax_lapack.m4` has none (it only ever tries `-llapack`), so a macOS
runner fails the auto-detection as well. The conda env therefore carries
`openblas`, and the build step tells configure where it is instead of
relying on a search path: `--with-blas="-L${ENV_PREFIX}/lib
-lopenblas" --with-lapack="-L${ENV_PREFIX}/lib -lopenblas"` — one library
provides both symbol sets, on both operating systems. Measured against
the 1.1.0 tarball: configure then exits 0 with `checking for dgemm_ in
-L<env>/lib -lopenblas... yes` and `checking for cheev_ in -L<env>/lib
-lopenblas... yes`, and the Makefile ends up with `LAPACK_LIBS =
-L<env>/lib -lopenblas`, `BLAS_LIBS =` (emptied by the `BLAS_LIBS ==
LAPACK_LIBS` rule at `configure.ac:291`) and `LIBS = -L<env>/lib
-lopenblas -lpthread -lm`.

**First CI run (44789003065) failed all four cells, for two reasons
outside the package.** The `macos-14` cell never ran a step — GitHub
answered `The configuration 'macos-14-arm64-us-default' is not
supported` (macOS 14 retired 2026-11-02, actions/runner-images#13518).
The three ubuntu cells died in the QMCkl build step with `configure:
error: BLAS was not found.`: the runners ship no BLAS/LAPACK, and the
local dry run had borrowed Homebrew's `libblas` and TREXIO, which the
runners lack — the dry run's fidelity gap. Fixed by `macos-latest`,
conda `openblas` in the env, and the explicit `--with-blas`/
`--with-lapack` flags (paragraph above); the dry run was re-mirrored on a
real conda-forge env accordingly. The four actions were bumped to their
current majors in the same change (`actions/checkout@v7`,
`actions/setup-python@v7`, `actions/cache@v6`,
`mamba-org/setup-micromamba@v3`), the previous pins warning about
deprecated Node 20.

**TREXIO probe corrected during M6.** The original premise did not hold:
`qmckl.h` declares `qmckl_trexio_read` unconditionally, and the 1.1.0
release tarball defines it even under `--without-trexio`, as a stub
returning `QMCKL_FAILURE` (the `#else` branch of `src/qmckl_trexio.c`), so
a symbol-presence predicate cannot tell the builds apart. The CMake
predicate was changed from `qmckl_trexio_read` to `trexio_open`
(§4.2 item 3), which appears only when the library was actually built with
TREXIO support, and the `--without-trexio` cell now meets its original
expectation: `trexio_read` unbound, presence check passing with
`EXPECT_TREXIO=false`, and the TREXIO-dependent tests skipping.

---

## 10. Milestones and acceptance

| # | Deliverable | Accepted when |
|---|---|---|
| M0 | `pyproject.toml`, `CMakeLists.txt`, `README.md`, `LICENSE`, `.gitignore`, placeholder `src/qmckl/__init__.py` | **done, configure-verified**; `pip install -e .` now stops at `qmckl.cmake` missing / `csrc/generate.py` missing, never at pkg-config for a valid 1.x install |
| M1 | `generate.py` parser + classification + `--dump-json` | **done**: 239 declarations in 6 buckets summing to 239, 0 unclassified, JSON round-trips, positional `(name, argc)` sequence identical to the independent cross-check |
| M2 | `context.hpp/.cpp`, `module.cpp`, exception, §5.7 error table, generator skeleton emission, `context.py`/`errors.py` skeleton, `test_context.py`, `test_errors.py` | `pytest -q tests` green with zero generated bindings |
| M3 | C++ + `.pyi` emitters (§5.4, §5.5e, §5.6), ABI override, stub installed, `tests/test_surface.py` + `tests/test_generated.py` | `import qmckl` exposes **223** generated names (222 without TREXIO) plus `Context`/`_Context`, `last_error`, `get_ao_basis_type`, `QMCKLStatusError` and `ERROR_CODES`; `qmckl_check` and `string_of_error` are absent; both test files green; the wheel ships `_core.pyi` and `py.typed`; a rebuild is byte-identical to the re-cut `--dump-json` baseline |
| M4 | §7 Python layer + `errors.py` table | `test_shapes.py` **and** `test_blas.py` green |
| M5 | `tests/test_distance.py` + `tests/test_trexio.py`, completing the suite | green on macOS arm64 against QMCkl 1.1.0 |
| M6 | `.github/workflows/ci.yml`, four jobs | green on ubuntu + macos, including the `--without-trexio` job — first CI run 44789003065 failed all four cells (retired `macos-14` label; no BLAS/LAPACK on the runners), fixed with `macos-latest` + conda `openblas` + `--with-blas`/`--with-lapack`; 2026-11-20 dry runs against the 1.1.0 tarball from a conda-forge env: with-TREXIO 56 passed; no-TREXIO 51 passed, 5 skipped |

The M6 acceptance is met once the repository has a GitHub remote and the
workflow has run at least once there, since it cannot be executed locally;
the §9 dry runs are the validation on record.

**Ordering constraint M2 ↔ M3.** The runtime cannot be tested without a
buildable extension, and the extension compiles `generated_bindings.cpp`, which
only the generator can write. So M2 must *replace* `generate.py`'s
"emission is implemented in M3, refusing to emit" guard with a real writer that
produces a **compilable skeleton**:

```cpp
// generated_bindings.cpp
#include <nanobind/nanobind.h>
#include "context.hpp"

namespace qmckl_python {
void register_generated(nb::module_ &m) { (void) m; }
}
```

plus an empty `_core.pyi`. Every intermediate state then builds, `m.def` count
is zero, and `test_context.py` / `test_errors.py` exercise the hand-written
layer for real. M3 grows that skeleton into the per-function loop and the stub,
without changing the CLI, the file names, or the CMake integration.

---

## 11. Traps (also to be written in README)

1. **Version coupling** — bindings are generated from the installed header; a
   QMCkl upgrade requires reinstalling the package.
2. **Shared library only** — `qmckl.pc` does not propagate TREXIO, so
   `libqmckl.a` will not link.
3. **Raw layer** — a wrongly shaped array with sufficient size is silently
   reinterpreted (C semantics). Use the `Context` properties.
4. **TREXIO support is a capability, not a symbol** — `qmckl.h` declares
   `qmckl_trexio_read` unconditionally, and the 1.1.0 release tarball
   defines it even under `--without-trexio`, as a stub returning
   `QMCKL_FAILURE` ("QMCkl was compiled without TREXIO").
   `hasattr(qmckl, "trexio_read")` is a valid check only because the build
   probe (§4.2 item 3) refuses to bind the stub; forcing
   `-DQMCKL_HAS_TREXIO=ON` on such a build binds the stub, and the call
   then raises `QMCKLStatusError` with that message.
5. **`qmckl_string_of_error` is broken in 1.1.0 tarball builds**; the package
   uses its own table.
6. **OpenMP** — a QMCkl built with `--enable-openmp` spawns threads inside a
   Python call. Recommended build for Python use: `--without-openmp`, and
   thread in Python (the GIL is released around every call).
7. **`noconvert()`** — never remove it from array arguments; without it
   nanobind copies, and outputs are lost.
8. **`qmckl_get_electron_ee_potential` / `en_potential` carry no
   `size_max`** — the raw layer cannot check the buffer it is given; use the
   `Context.ee_potential` / `en_potential` properties, which allocate it.
9. **Zero points** — with `point_num == 0` the four point-evaluated providers
   (`ao_value`, `ao_vgl`, `mo_value`, `mo_vgl`) call
   `qmckl_malloc(ctx, {size: 0})`.  QMCkl 1.1.0 stores that record with
   `size == 0`, which its "first free slot" scan treats as an empty slot: the
   next allocation overwrites the record and leaks the block, while
   `n_allocated` stays incremented, so it can never return to zero — an
   assert-enabled `qmckl_context_destroy` aborts.  The properties raise
   `ValueError` instead.
10. **`qmckl_distance` checks LDB with the wrong bit** — in
    `src/qmckl_distance_f.F90` only `qmckl_distance` gates its LDB checks on
    `iand(transab,1)` (transa's bit) instead of `iand(transab,2)`
    (`distance_sq` and `distance_rescaled` are correct; `distance_sq` also
    reports an LDB problem as `INVALID_ARG_7`).  The result: with
    `transa='T', transb='N'` it demands `ldb >= n` and rejects the packed
    `ldb = 3` with `QMCKL_INVALID_ARG_9`, and with `transa='N', transb='T'`
    it demands `ldb >= 3`.  For the packed layouts of §8 only `n == 3`
    satisfies both, so `tests/test_distance.py` pins `n = 3`.  This is an
    upstream defect, not a binding one: the raw layer stays a faithful
    passthrough.

---

## 12. Out of scope for v1

Wheels and cibuildwheel; conda-forge recipe; docstrings generated from the org
sources; typed exception attributes (`.code`, `.function`); asynchronous or
threaded execution helpers; GPU/verificarlo.
