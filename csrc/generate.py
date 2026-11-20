#!/usr/bin/env python3
"""Generate the qmckl-python bindings from an installed qmckl.h.

Milestone M3 of PLAN.md (qmckl-python spec v1): the header parser, the
function classification, ``--dump-json``, the §5.7 error-code table read
from the raw header text, and the M3 emission: ``--cpp`` writes the full
C++ translation unit (the §5.7 error-code table plus one ``m.def`` per
generated binding, PLAN.md §5.4), and ``--pyi`` writes the full ``_core.pyi``
stub (the hand-written surface plus one ``def`` per generated binding,
PLAN.md §5.6).

Pipeline (PLAN.md §5.1):
  1. delete ``/* … */`` and ``//…`` comments;
  2. delete preprocessor lines (``^#…``) and inline function bodies (``{…}``
     blocks whose opening brace directly follows a ``)`` — braces that do not
     follow a ``)``, such as the ``extern "C" {`` wrapper of the installed
     header, are kept);
  3. split on ``;`` and keep declarators whose function name starts with
     ``qmckl_``;
  4. normalise each argument: base type, pointer level, and the const-ness of
     the *pointee*, decided from the declarator as a whole.  A ``const``
     before a ``*`` qualifies the pointee (input array / string); a ``const``
     after a ``*`` qualifies the pointer itself and is dropped.  Hence
     ``double* const C`` is an OUTPUT array.  Never a ``"const" in arg`` test.

Classification buckets (PLAN.md §5.5) form a partition of every parsed
declaration:
  scalar_out         (a) scalar out-parameter getters (explicit table, 18)
  array_no_size_max  (b) array outputs without size_max (explicit table, 4)
  handwritten        (c) bound by hand in csrc/module.cpp (§5.5(c), 7)
  excluded           (c) not exposed at all (§5.5(c), 7)
  trexio             (d) qmckl_trexio_read; codegen only when the linked
                         library exports it (§5.5(d), ``--trexio`` /
                         ``--no-trexio``)
  generated          the ordinary shapes of §5.2/§5.3: everything else,
                     mechanically generated (sub-shapes: getter,
                     getter_transp, safe, other)

``emitted`` marks the declarations that will produce a *generated* binding
(everything except handwritten and excluded; trexio only when enabled).

Hard failures (exit 1, per §5.1 "fail loudly, do not guess"):
  * fewer than 230 declarations;
  * ``qmckl_dgemm_safe`` missing (marks the >= 1.1 API);
  * a §5.5 table name absent from the header, or present with a shape that
    contradicts its table;
  * a ``size_max*`` parameter that §5.3 cannot resolve to exactly one array;
  * a pointer parameter the role classifier cannot classify (unknown base
    type, pointer level >= 2, …);
  * a parameter or return type outside the §5.2 mapping tables;
  * a Python keyword among the argument names after ``_`` suffixing;
  * (emitters only) an argument ``python_name`` that is a C++ keyword, a
    ``size_max`` whose declared type the emitter cannot cast to, or an
    ``ABI_OVERRIDE_TABLE`` entry whose function is absent from the header,
    whose names or arity drifted, whose type is outside the §5.2 mapping,
    whose prototype matches the header (stale), or whose name is not
    emitted (a silently rotting table reintroduces the bug, §5.5(e));
  * (emitters only) a finished ``_core.pyi`` that ``ast.parse`` rejects.

The §2 verified facts (239 declarations, 223 ``qmckl_exit_code`` returns, 125
getter/setter-shape functions) are reported and warned about, not fatal: the
bindings are generated from whatever 1.x header is installed, so a 1.1.1 that
adds or de-duplicates a declaration must still build.  Loss of a §5.5 table
name *is* fatal, which is what protects the hand-audited classification.

Usage:
  generate.py --header PATH/qmckl.h --dump-json OUT.json [--trexio|--no-trexio]
  generate.py --header PATH/qmckl.h --cpp OUT.cpp --pyi OUT.pyi [--trexio|--no-trexio]

M3 grew the M2 skeleton (PLAN.md §10) into the per-function loop without
changing the CLI, the file names, or the CMake integration.  ``--cpp`` emits
one ``m.def`` per generated binding (de-duplicated by name, first occurrence
wins, §5.1/§5.4); ``--pyi`` fills the ``BEGIN/END GENERATED BINDINGS``
markers of ``_core.pyi`` and ``ast.parse``s the finished stub.

On top of the §5.2 type mapping, the emitters consult one explicit table,
applied to the emitted C++/stub only (the IR and ``--dump-json`` keep the
header's truth):

  * ``ABI_OVERRIDE_TABLE`` (§5.5(e)): ``qmckl_ao_gaussian_vgl`` and
    ``qmckl_ao_slater_vgl`` — the installed header's prototypes are wrong
    (§2): ``n``/``num_slater`` are declared ``const int64_t *`` but taken
    by value by the bind(C) symbol, the gaussian's ``A`` is declared
    ``const int64_t *`` but is ``const double *`` in the true prototype,
    and both ``VGL`` are declared ``const double *`` but written by the
    implementation (Fortran ``intent(out)``), so they are ``double *``.
    The emitter builds a synthetic argument list from the table's true
    prototype through the header's own type classifier and derives the
    signature, the ``.pyi`` annotations and the call expression from it
    by the ordinary §5.2–§5.4 rules (one code path, not two); the call
    site uses a file-scope function pointer of the true prototype
    (``reinterpret_cast``-initialised, not ``constexpr``: a
    ``reinterpret_cast`` of a function pointer is not a constant
    expression).

The table is fatal in both directions: a function absent from the header,
a drifted name or arity, a type outside the §5.2 mapping, a prototype
that matches the header (stale), or an override for a name that is not
emitted, exits 1, because a silently rotting table reintroduces the bug.

Standard library only, Python >= 3.10.
"""

from __future__ import annotations

import argparse
import ast
import json
import keyword
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# §5.1 / §2 constants
# ---------------------------------------------------------------------------

MIN_FUNCTIONS = 230           # §5.1 sanity assertion
EXPECTED_FUNCTIONS = 239      # §2 verified fact; M1 acceptance count
EXPECTED_EXIT_CODE_RETS = 223  # §2 verified fact (reported, not fatal)
EXPECTED_GETTER_SHAPE = 125    # §2 verified fact (reported, not fatal)

# ---------------------------------------------------------------------------
# §5.5 explicit tables (authoritative; never derived from the header)
# ---------------------------------------------------------------------------

# (a) scalar out-parameter getters: (ctx, T* x) — return the value instead of
#     taking a buffer.
SCALAR_OUT_TABLE = (
    "qmckl_get_point_num",
    "qmckl_get_mo_basis_mo_num",
    "qmckl_get_nucleus_num",
    "qmckl_get_jastrow_champ_aord_num",
    "qmckl_get_nucleus_repulsion",
    "qmckl_get_jastrow_champ_bord_num",
    "qmckl_get_electron_num",
    "qmckl_get_jastrow_champ_cord_num",
    "qmckl_get_electron_up_num",
    "qmckl_get_jastrow_champ_type_nucl_num",
    "qmckl_get_electron_down_num",
    "qmckl_get_jastrow_champ_rescale_factor_ee",
    "qmckl_get_electron_walk_num",
    "qmckl_get_jastrow_champ_dim_c_vector",
    "qmckl_get_ao_basis_shell_num",
    "qmckl_get_jastrow_champ_spin_independent",
    "qmckl_get_ao_basis_prim_num",
    "qmckl_get_ao_basis_ao_num",
)

# (b) array outputs without size_max: the caller allocates; QMCkl writes a
#     fixed number of elements.
ARRAY_NO_SIZE_MAX_TABLE = (
    "qmckl_get_electron_ee_potential",
    "qmckl_get_electron_en_potential",
    "qmckl_compute_lkpm_combined_index_doc",
    "qmckl_compute_lkpm_combined_index_hpc",
)

# (c) hand-written bindings, excluded from codegen (§5.5(c), §6.2, §7):
#     context lifecycle -> qmckl::Context, context_touch -> Context.touch(),
#     last_error -> qmckl.last_error(ctx), get_ao_basis_type ->
#     qmckl.get_ao_basis_type(ctx).
HANDWRITTEN_TABLE = (
    "qmckl_context_create",
    "qmckl_context_copy",
    "qmckl_context_check",
    "qmckl_context_destroy",
    "qmckl_context_touch",
    "qmckl_last_error",
    "qmckl_get_ao_basis_type",
)

# (d) excluded / not exposed at all.
EXCLUDED_TABLE = (
    "qmckl_lock",
    "qmckl_unlock",
    "qmckl_get_error",
    "qmckl_set_error",
    "qmckl_failwith",
    # The released QMCkl 1.1.0 tarball ships this with an empty switch, so it
    # returns "Unknown error" for every code.  qmckl.errors.error_name() is the
    # replacement (PLAN.md §2, §5.5(c)).
    "qmckl_string_of_error",
    # Unusable as a Python API, three independent reasons (verified in
    # src/qmckl_error.c, PLAN.md §5.5(c)): (1) it prints
    # qmckl_string_of_error(rc) to stderr, which always says "Unknown
    # error" on 1.1.0; (2) it passes a non-NUL-terminated, otherwise
    # uninitialised char[QMCKL_MAX_MSG_LEN] to fprintf("%s", ...) — when
    # the context carries no stored error, qmckl_get_error memcpy's zero
    # bytes and the stack buffer is printed as-is (uninitialised read);
    # (3) it passes its own rc argument as qmckl_get_error's exit_code
    # OUT-parameter, so it can return QMCKL_SUCCESS for a non-zero rc —
    # an error silently turned into success.  Every generated
    # exit-code function already raises through qmckl::check (§6.1), so
    # nothing in the package needs the C function.
    "qmckl_check",
)

# (e) TREXIO-only: skipped entirely when the linked library does not export
#     it (no binding, no stub).
TREXIO_TABLE = (
    "qmckl_trexio_read",
)

ALL_TABLES = (
    SCALAR_OUT_TABLE
    + ARRAY_NO_SIZE_MAX_TABLE
    + HANDWRITTEN_TABLE
    + EXCLUDED_TABLE
    + TREXIO_TABLE
)

# (func, arg) pairs whose pointer parameter is a single-element in/out
# scalar per the org argument tables (input: capacity/initial value, output:
# the result), not a sized array: no size_max exists for it.  §5.2 maps a
# non-const ``T*`` to a writable ndarray, so the M3 emitter will bind these
# as 1-element arrays; the table documents that intent in the dump.
INOUT_SCALAR_POINTER = {
    ("qmckl_adjugate", "det_l"),
    ("qmckl_adjugate_safe", "det_l"),
    ("qmckl_ao_polynomial_vgl", "n"),
    ("qmckl_ao_polynomial_vgl_doc", "n"),
    ("qmckl_ao_polynomial_transp_vgl", "n"),
    ("qmckl_ao_polynomial_transp_vgl_doc", "n"),
    ("qmckl_ao_polynomial_transp_vgl_hpc", "n"),
    ("qmckl_ao_polynomial_hessian", "n"),
    ("qmckl_compute_ao_polynomial_hessian_doc", "n"),
}

BUCKET_ORDER = (
    "generated",
    "scalar_out",
    "array_no_size_max",
    "handwritten",
    "excluded",
    "trexio",
)

# §5.5(e) header/ABI divergence — the two functions whose installed
# qmckl.h prototypes are wrong (§2).  The table carries their TRUE
# bind(C) prototypes; the emitter builds a synthetic argument list from
# them through the header's own type classifier and derives the
# signature, the .pyi annotations and the call expression from it by
# the ordinary §5.2–§5.4 rules (one code path, not two).  Applied to
# the emitted C++/stub ONLY (the IR and --dump-json keep the header's
# truth).  Checked fatal both ways by _check_emit_tables (emitters
# only): name absent from the header, drifted names/arity, a type
# outside the §5.2 mapping, a prototype that matches the header
# (stale), or a name that is not emitted — a silently rotting table
# reintroduces the bug.
#
# Do NOT add the _hpc variants: their VGL is already `double *` and
# their other parameters match the header (verified in
# src/qmckl_ao.c).  There is no gaussian _hpc variant.
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

# C++ keywords (C++17 plus the C++20 additions): the IR's python_name
# becomes a C++ lambda parameter name, so none of these may occur.  Asserted
# before emitting (PLAN.md §5.4 rule 1).
CPP_KEYWORDS = frozenset("""
    alignas alignvalof and asm auto bool break case catch char char8_t
    char16_t char32_t class concept const consteval constexpr constinit
    const_cast continue co_await co_return co_yield default delete do
    double dynamic_cast else enum explicit export extern false float for
    friend goto if inline int long mutable namespace new noexcept not
    nullptr operator or private protected public register reinterpret_cast
    requires return short signed sizeof static static_assert static_cast
    struct switch template this thread_local throw true try typedef typeid
    typename union unsigned using virtual void volatile wchar_t while
""".split())

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """A loud, fatal classification/parsing failure (never guess)."""


INT_BASES = frozenset({
    "int", "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "size_t", "ssize_t",
    "qmckl_exit_code",  # typedef int32_t
})
FLOAT_BASES = frozenset({"float", "double"})

KNOWN_BASES = INT_BASES | FLOAT_BASES | {"_Bool", "bool", "char", "qmckl_context"}


def _scalar_python(base: str) -> str:
    """§5.2 stub annotation for a scalar C type."""
    if base in INT_BASES:
        return "int"
    if base in FLOAT_BASES:
        return "float"
    if base == "_Bool":
        return "bool"
    if base == "char":
        return "str"
    raise ParseError(f"unmapped scalar base type {base!r}")


@dataclass
class Arg:
    c_type: str                       # normalised type part, e.g. "double* const"
    name: str                         # C name
    python_name: str                  # name, "_" suffixed if a Python keyword
    base: str                         # normalised base type ("bool" -> "_Bool")
    pointer_level: int                # 0, 1 (2+ is a hard error)
    is_const_pointee: Optional[bool]  # const before '*'; None at level 0
    is_const_pointer: Optional[bool]  # const between '*' and the name
    role: str                         # context | scalar | size_max | string |
                                      # char_out | in_array | out_array | code_out
    python_type: Optional[str]        # §5.2 stub annotation; None for size_max
    size_rule: Optional[str] = None   # bare | sized | strlen (size_max args only)
    size_target: Optional[str] = None # array argument it describes


@dataclass
class Func:
    name: str
    python_name: str                  # name minus the "qmckl_" prefix (§1)
    return_type: str                  # normalised C return type
    return_python: str                # §5.2 return mapping
    args: list[Arg] = field(default_factory=list)
    bucket: Optional[str] = None      # BUCKET_ORDER
    shape: Optional[str] = None       # getter | getter_transp | safe | other
    emitted: bool = True              # will produce a *generated* binding
    has_context: bool = False
    notes: list[str] = field(default_factory=list)
    emit_args: Optional[list[Arg]] = None  # §5.5(e) synthetic list, cached


# ---------------------------------------------------------------------------
# §5.1 parsing
# ---------------------------------------------------------------------------

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_WS = re.compile(r"\s+")

# A C parameter declarator: [const] base [*] [const-pointer] name [const].
# The leading const qualifies the pointee (input array / string); the const
# between the star and the name qualifies the pointer itself (e.g.
# "double* const C" is an OUTPUT array); the trailing const applies to the
# name (e.g. "int64_t const size_max").
_ARG = re.compile(
    r"^(?:(?P<const>const)\s+)?"
    r"(?P<base>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?P<stars>\*+)?\s*"
    r"(?:(?P<const_mid>const)\s+)?"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)?"
    r"\s*(?:(?P<const2>const)\s*)?$"
)

_RET = re.compile(
    r"^(?:(?P<const>const)\s+)?"
    r"(?P<base>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?P<stars>\*+)?$"
)


def _strip_bodies(text: str) -> str:
    """Remove inline function bodies: ``{…}`` blocks whose opening brace
    directly follows a ``)``.  Other braces (``extern "C" {``) are kept."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "{":
            j = i - 1
            while j >= 0 and text[j] in " \t":
                j -= 1
            if j >= 0 and text[j] == ")":
                depth, k = 1, i + 1
                while k < n and depth:
                    if text[k] == "{":
                        depth += 1
                    elif text[k] == "}":
                        depth -= 1
                    k += 1
                if depth:
                    raise ParseError("unbalanced '{' after ')' while stripping bodies")
                i = k
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _split_top_level(text: str) -> list[str]:
    """Split on top-level commas (paren/bracket aware)."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _matching_close(text: str, open_idx: int) -> int:
    depth = 0
    for k in range(open_idx, len(text)):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                return k
    raise ParseError(f"unbalanced '(' in declarator: {text[:80]!r}…")


def _parse_arg(func_name: str, arg_text: str) -> Arg:
    m = _ARG.match(arg_text)
    if not m:
        raise ParseError(f"{func_name}: cannot parse parameter {arg_text!r}")
    base = m["base"]
    if base == "bool":
        base = "_Bool"  # <stdbool.h> spelling; the API inventory uses _Bool
    if base not in KNOWN_BASES:
        raise ParseError(f"{func_name}: unknown base type {base!r} in {arg_text!r}")
    level = len(m["stars"] or "")
    if level > 1:
        raise ParseError(f"{func_name}: pointer level {level} in {arg_text!r}")
    name = m["name"]
    if not name:
        raise ParseError(f"{func_name}: anonymous parameter {arg_text!r}")
    const_pointee: Optional[bool] = (m["const"] is not None) if level else None
    const_pointer: Optional[bool] = (m["const_mid"] is not None) if level else None
    python_name = name if not keyword.iskeyword(name) else name + "_"
    # The type part of the declarator, exactly as written (whitespace-normalised).
    c_type = _WS.sub(" ", arg_text[: m.start("name")]).strip()

    if level == 0:
        if name.startswith("size_max"):
            if base not in INT_BASES:
                raise ParseError(
                    f"{func_name}: size_max parameter has non-integer type {base!r}"
                )
            role, python_type = "size_max", None
        elif base == "qmckl_context":
            role, python_type = "context", "Context"
        else:
            role = "scalar"
            # §5.2 maps no qmckl_exit_code parameter (the row is gone: on
            # 1.1.0 the only five declarations with such a parameter —
            # check, get_error, set_error, failwith, string_of_error —
            # are all excluded, §5.5(c)).  python_type stays None, so an
            # *emitted* declaration with one is unclassifiable and the
            # run fails (classify); excluded ones are never validated.
            if base == "qmckl_exit_code":
                python_type = None
            else:
                python_type = _scalar_python(base)
    elif base == "char":
        # const char* is a string; bare char* is a single-character buffer.
        role = "string" if const_pointee else "char_out"
        python_type = "str"
    elif base == "qmckl_exit_code":
        role, python_type = "code_out", None  # qmckl_get_error only
    else:
        role = "in_array" if const_pointee else "out_array"
        python_type = f"NDArray[{_scalar_python(base)}]"

    return Arg(c_type, name, python_name, base, level, const_pointee,
               const_pointer, role, python_type)


def _parse_ret(func_name: str, ret_text: str) -> str:
    if ret_text == "void":
        return "None"
    m = _RET.match(ret_text)
    if not m:
        raise ParseError(f"{func_name}: cannot parse return type {ret_text!r}")
    base = m["base"]
    if base == "bool":
        base = "_Bool"
    level = len(m["stars"] or "")
    if level > 1:
        raise ParseError(f"{func_name}: pointer return type {ret_text!r}")
    if level == 0:
        if base not in KNOWN_BASES or base == "char":
            raise ParseError(f"{func_name}: unmapped return type {ret_text!r}")
        if base == "qmckl_exit_code":
            return "None"  # raise on non-zero (§5.2)
        if base == "qmckl_context":
            return "Context"
        return _scalar_python(base)
    if base == "char" and m["const"]:
        return "str"
    raise ParseError(f"{func_name}: unmapped pointer return type {ret_text!r}")


def parse_header(path: Path) -> list[Func]:
    """§5.1: parse the installed header into one Func per declaration."""
    raw = path.read_text(encoding="utf-8")
    text = _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", raw))
    text = "\n".join(ln for ln in text.splitlines()
                     if not ln.lstrip().startswith("#"))
    text = _WS.sub(" ", text)
    text = _strip_bodies(text)

    funcs: list[Func] = []
    for frag in text.split(";"):
        frag = frag.strip()
        if not frag or "(" not in frag:
            continue
        open_idx = frag.index("(")
        close_idx = _matching_close(frag, open_idx)
        if frag[close_idx + 1:].strip():
            raise ParseError(f"trailing text after parameter list in {frag[:60]!r}…")
        head = frag[:open_idx].split()
        if not head:
            continue
        name, ret_tokens = head[-1], head[:-1]
        if not name.startswith("qmckl_"):
            continue
        params = frag[open_idx + 1:close_idx].strip()
        args = [_parse_arg(name, p) for p in
                (_split_top_level(params) if params and params != "void" else [])]
        for idx, a in enumerate(args):
            if a.role == "context" and idx != 0:
                raise ParseError(f"{name}: qmckl_context parameter is not first")
        if len({a.python_name for a in args}) != len(args):
            raise ParseError(f"{name}: duplicate Python argument names after suffixing")
        funcs.append(Func(
            name=name,
            python_name=name[len("qmckl_"):],
            return_type=" ".join(ret_tokens),
            return_python=_parse_ret(name, " ".join(ret_tokens)),
            args=args,
            has_context=any(a.role == "context" for a in args),
        ))
    return funcs


# ---------------------------------------------------------------------------
# §5.3 / §5.5 classification
# ---------------------------------------------------------------------------

def _resolve_size_max(func: Func) -> None:
    """Attach size_rule/size_target to every size_max argument (§5.3)."""
    for a in func.args:
        if a.role != "size_max":
            continue
        if func.name in TREXIO_TABLE:
            strings = [s for s in func.args if s.role == "string"]
            if len(strings) != 1:
                raise ParseError(f"{func.name}: strlen rule needs exactly one "
                                  f"string parameter")
            a.size_rule, a.size_target = "strlen", strings[0].name
        elif a.name == "size_max":
            arrays = [s for s in func.args if s.role in ("in_array", "out_array")]
            if len(arrays) != 1:
                raise ParseError(f"{func.name}: bare size_max must describe "
                                  f"exactly one array, found {[s.name for s in arrays]}")
            a.size_rule, a.size_target = "bare", arrays[0].name
        else:
            x = a.name[len("size_max_"):]
            cands = [s for s in func.args
                     if s.role in ("in_array", "out_array")
                     and (s.name == x or s.name.endswith("_" + x))]
            if len(cands) != 1:
                raise ParseError(f"{func.name}: size_max_{x} resolves to "
                                  f"{[s.name for s in cands]}")
            a.size_rule, a.size_target = "sized", cands[0].name


def _shape(func: Func) -> str:
    roles = [a.role for a in func.args]
    if any(a.size_rule == "sized" for a in func.args):
        return "safe"
    if (len(func.args) == 3 and roles[0] == "context"
            and roles[1] in ("in_array", "out_array") and roles[2] == "size_max"):
        return "getter"
    transp = (len(func.args) in (4, 5) and roles[0] == "context"
              and roles[1] == "scalar" and func.args[1].base == "char")
    if (transp and roles.count("size_max") == 1
            and sum(1 for r in roles if r in ("in_array", "out_array")) == 1):
        return "getter_transp"
    return "other"


def _check_scalar_out(func: Func) -> None:
    if (len(func.args) != 2 or func.args[0].role != "context"
            or func.args[1].role != "out_array"
            or func.return_type != "qmckl_exit_code"):
        raise ParseError(f"{func.name}: §5.5(a) table entry but shape is "
                         f"{[(a.name, a.role) for a in func.args]}")


def _check_array_no_sizemax(func: Func) -> None:
    if (not any(a.role in ("out_array",) for a in func.args)
            or any(a.role == "size_max" for a in func.args)):
        raise ParseError(f"{func.name}: §5.5(b) table entry but shape is "
                         f"{[(a.name, a.role) for a in func.args]}")


def _check_trexio(func: Func) -> None:
    roles = [a.role for a in func.args]
    if roles != ["context", "string", "size_max"]:
        raise ParseError(f"{func.name}: §5.5(d) table entry but shape is {roles}")


def classify(functions: list[Func], trexio: bool) -> None:
    """Assign bucket/shape/emitted to every declaration; fail loudly otherwise."""
    if len(functions) < MIN_FUNCTIONS:
        raise ParseError(f"only {len(functions)} qmckl_* declarations found "
                         f"(< {MIN_FUNCTIONS}); is this the >= 1.1 header?")
    # Not fatal: a 1.1.x that adds (or de-duplicates) a declaration must still
    # build, and a new declaration lands in "generated" through §5.2.  Drift in
    # the hand-audited tables is caught below, by name.
    if len(functions) != EXPECTED_FUNCTIONS:
        print(f"generate.py: note: {len(functions)} declarations, §2 audited "
              f"{EXPECTED_FUNCTIONS} on QMCkl 1.1.0 (different 1.x header?)",
              file=sys.stderr)
    names = {f.name for f in functions}
    if "qmckl_dgemm_safe" not in names:
        raise ParseError("qmckl_dgemm_safe missing: not the >= 1.1 API")
    for table in (SCALAR_OUT_TABLE, ARRAY_NO_SIZE_MAX_TABLE, HANDWRITTEN_TABLE,
                  EXCLUDED_TABLE, TREXIO_TABLE):
        missing = [n for n in table if n not in names]
        if missing:
            raise ParseError(f"§5.5 table names absent from header: {missing}")
    for table in (SCALAR_OUT_TABLE, ARRAY_NO_SIZE_MAX_TABLE, HANDWRITTEN_TABLE,
                  EXCLUDED_TABLE, TREXIO_TABLE):
        if len(set(table)) != len(table):
            raise ParseError("duplicate entry inside a §5.5 table")
    if len(set(ALL_TABLES)) != len(ALL_TABLES):
        raise ParseError("a name appears in more than one §5.5 table")

    for func in functions:
        _resolve_size_max(func)
        for a in func.args:  # §5.1: no Python keyword may survive suffixing
            if keyword.iskeyword(a.python_name):
                raise ParseError(f"{func.name}: argument {a.name!r} is a Python "
                                  f"keyword after suffixing")
        name = func.name
        if name in SCALAR_OUT_TABLE:
            _check_scalar_out(func)
            func.bucket = "scalar_out"
            func.emitted = True
        elif name in ARRAY_NO_SIZE_MAX_TABLE:
            _check_array_no_sizemax(func)
            func.bucket = "array_no_size_max"
            func.emitted = True
        elif name in HANDWRITTEN_TABLE:
            func.bucket = "handwritten"
            func.emitted = False
        elif name in EXCLUDED_TABLE:
            func.bucket = "excluded"
            func.emitted = False
        elif name in TREXIO_TABLE:
            _check_trexio(func)
            func.bucket = "trexio"
            func.emitted = trexio
        else:
            # Ordinary generated shape: every parameter and the return type
            # must map through §5.2, otherwise the declaration is
            # unclassifiable and the run fails.
            for a in func.args:
                if a.role == "size_max" or a.python_type is not None:
                    continue
                if a.role == "context":
                    continue
                raise ParseError(f"{func.name}: unclassifiable argument "
                                  f"{a.name!r} (role {a.role})")
            func.bucket = "generated"
            func.shape = _shape(func)
            func.emitted = True
            for a in func.args:
                if (a.role == "out_array" and a.pointer_level == 1
                        and (name, a.name) in INOUT_SCALAR_POINTER):
                    func.notes.append(
                        f"in/out single-element scalar pointer {a.name!r} "
                        f"(org tables); bound as a 1-element writable array per §5.2")
                if a.role == "size_max" and a.base == "size_t":
                    func.notes.append(
                        f"size_max parameter declared as size_t, not int64_t")
    unclassified = [f.name for f in functions if f.bucket is None]
    if unclassified:
        raise ParseError(f"unclassified declarations: {unclassified}")


# ---------------------------------------------------------------------------
# §5.7 Error-code table (second pass over the raw header text)
# ---------------------------------------------------------------------------

# PLAN.md §5.7: scan the *raw* header (not the stripped §5.1 text, where the
# preprocessor lines are already gone) for the exit-code macros.  The 1.1.0
# header writes exactly two spaces after `#define` and a mandatory
# `((qmckl_exit_code) N)` cast, so a bare `#define QMCKL_X 3` does not occur
# and must not be guessed at.
_ERROR_CODE_RE = re.compile(
    r"^\s*#\s*define\s+(QMCKL_[A-Z0-9_]+)\s+"
    r"\(\s*\(\s*qmckl_exit_code\s*\)\s*(-?\d+)\s*\)\s*$"
)

MIN_ERROR_CODES = 30  # §5.7 sanity assertion


def parse_error_codes(raw: str) -> list[tuple[str, int]]:
    """§5.7: extract the (name, value) exit-code pairs, in header order."""
    codes: list[tuple[str, int]] = []
    seen_name: dict[str, int] = {}
    seen_value: dict[int, str] = {}
    for line in raw.splitlines():
        m = _ERROR_CODE_RE.match(line)
        if not m:
            continue
        name, value = m.group(1), int(m.group(2))
        if name in seen_name:
            raise ParseError(f"duplicate error-code macro {name} "
                             f"(values {seen_name[name]} and {value})")
        if value in seen_value:
            raise ParseError(f"duplicate error-code value {value} "
                             f"({seen_value[value]} and {name})")
        seen_name[name] = value
        seen_value[value] = name
        codes.append((name, value))

    if len(codes) < MIN_ERROR_CODES:
        raise ParseError(f"only {len(codes)} error-code macros found "
                         f"(< {MIN_ERROR_CODES}); is this the >= 1.1 header?")
    if seen_name.get("QMCKL_SUCCESS") != 0:
        raise ParseError("QMCKL_SUCCESS is missing or not 0")
    if "QMCKL_INVALID_ARG_20" not in seen_name:
        raise ParseError("QMCKL_INVALID_ARG_20 missing; not the >= 1.1 API")
    return codes


# ---------------------------------------------------------------------------
# Summary and JSON dump
# ---------------------------------------------------------------------------

def _arg_dict(a: Arg) -> dict:
    d = {
        "c_type": a.c_type,
        "name": a.name,
        "python_name": a.python_name,
        "base": a.base,
        "pointer_level": a.pointer_level,
        "is_const_pointee": a.is_const_pointee,
        "is_const_pointer": a.is_const_pointer,
        "role": a.role,
        "python_type": a.python_type,
    }
    if a.size_rule is not None:
        d["size_rule"] = a.size_rule
        d["size_target"] = a.size_target
    return d


def _func_dict(f: Func) -> dict:
    d = {
        "name": f.name,
        "python_name": f.python_name,
        "return_type": f.return_type,
        "return_python": f.return_python,
        "bucket": f.bucket,
        "shape": f.shape,
        "emitted": f.emitted,
        "has_context": f.has_context,
        "args": [_arg_dict(a) for a in f.args],
    }
    if f.notes:
        d["notes"] = f.notes
    return d


def to_json(functions: list[Func], trexio: bool, header: str) -> str:
    buckets = {b: sum(1 for f in functions if f.bucket == b) for b in BUCKET_ORDER}
    obj = {
        "generator": "csrc/generate.py (M1: parser + classification)",
        "header": header,
        "trexio": trexio,
        "function_count": len(functions),
        "unique_name_count": len({f.name for f in functions}),
        "emitted_count": sum(1 for f in functions if f.emitted),
        "buckets": buckets,
        "functions": [_func_dict(f) for f in functions],
    }
    return json.dumps(obj, indent=2)


def print_summary(functions: list[Func], trexio: bool) -> None:
    import collections
    names = [f.name for f in functions]
    dups = {n: c for n, c in collections.Counter(names).items() if c > 1}
    rets = collections.Counter(f.return_type for f in functions)
    print(f"qmckl.h: declarations {len(functions)} | unique names "
          f"{len(set(names))}"
          + (f" | duplicates: {sorted(dups)}" if dups else ""))
    print("return types: " + " | ".join(f"{k} {v}" for k, v in rets.most_common()))
    print("buckets:")
    for b in BUCKET_ORDER:
        print(f"  {b:<18} {sum(1 for f in functions if f.bucket == b)}")
    print(f"  {'total':<18} {len(functions)}")
    gen = [f for f in functions if f.bucket == "generated"]
    shapes = collections.Counter(f.shape for f in gen)
    print("generated sub-shapes: "
          + " | ".join(f"{k} {shapes[k]}" for k in ("getter", "getter_transp", "safe", "other") if k in shapes))
    getter = shapes.get("getter", 0)
    transp4 = sum(1 for f in gen
                  if f.shape == "getter_transp" and len(f.args) == 4)
    trexio_n = sum(1 for f in functions if f.bucket == "trexio")
    s2 = getter + transp4 + trexio_n
    print(f"§2 check: getter/setter shape '(ctx, one array, size_max)' = "
          f"{getter} + {transp4} (with transp char, no extra scalar) "
          f"+ {trexio_n} (trexio_read) = {s2}"
          + (" [matches §2: 125]" if s2 == EXPECTED_GETTER_SHAPE
             else f" [DIFFERS from §2: {EXPECTED_GETTER_SHAPE}]"))
    n_exit = rets.get("qmckl_exit_code", 0)
    if n_exit != EXPECTED_EXIT_CODE_RETS:
        print(f"warning: {n_exit} qmckl_exit_code returns, "
              f"§2 verified fact is {EXPECTED_EXIT_CODE_RETS}", file=sys.stderr)
    emitted = [f for f in functions if f.emitted]
    distinct = len({f.name for f in emitted})
    print(f"emitted (codegen): {len(emitted)} declarations -> "
          f"{distinct} distinct bindings"
          + ("" if trexio else " [trexio_read dropped: --no-trexio]"))


# ---------------------------------------------------------------------------
# Emitters (M3: per-function bindings — PLAN.md §5.4, §5.5(e), §5.6; the
# CLI, the file names, and the CMake integration are untouched, §10)
# ---------------------------------------------------------------------------

def _check_emit_tables(functions: list[Func]) -> None:
    """Fatal both-ways check of ABI_OVERRIDE_TABLE (§5.5(e)).

    Emitter path only (``--dump-json`` never runs it): the overrides
    apply to the emitted C++/stub only, not to the IR.  Each of these
    exits 1: a name absent from the header, a drifted arity, a
    parameter name or order that differs, an override type outside the
    §5.2 mapping, a prototype that matches the header (stale entry),
    or a name that is not in an emitted bucket — a silently rotting
    table reintroduces the bug.
    """
    by_name: dict[str, Func] = {}
    for f in functions:
        by_name.setdefault(f.name, f)
    for fname, override in sorted(ABI_OVERRIDE_TABLE.items()):
        f = by_name.get(fname)
        if f is None:
            raise ParseError(f"ABI override table: {fname} absent from the header")
        if not f.emitted:
            raise ParseError(f"ABI override table: {fname} is not in an "
                             f"emitted bucket (bucket {f.bucket!r})")
        if len(override) != len(f.args):
            raise ParseError(f"ABI override table: {fname} has "
                             f"{len(override)} override parameters, the "
                             f"header has {len(f.args)}")
        n_diff = 0
        for a, (oname, otype) in zip(f.args, override):
            if a.name != oname:
                raise ParseError(f"ABI override table: {fname} parameter "
                                 f"{oname!r} differs from the header's "
                                 f"{a.name!r}; names and order must match")
            s = _parse_arg(fname, f"{otype} {oname}")
            if s.python_type is None and s.role not in ("size_max", "context"):
                raise ParseError(f"ABI override table: {fname}({oname}) type "
                                 f"{otype!r} is outside the §5.2 mapping")
            # Same parameter type = same base, pointer level and
            # pointee const-ness: a level-0 const (the header writes
            # "const qmckl_context") and a const on the pointer itself
            # are not part of the ABI.
            if (s.base, s.pointer_level, s.is_const_pointee) != \
                    (a.base, a.pointer_level, a.is_const_pointee):
                n_diff += 1
        if n_diff == 0:
            raise ParseError(f"ABI override table: {fname} matches the "
                             f"header exactly; the entry is stale")


def _override_args(f: Func) -> list[Arg]:
    """The synthetic argument list of an ABI_OVERRIDE_TABLE entry: the
    override's types run through the header's own type classifier
    (``_parse_arg``), so there is no second type mapping (§5.2,
    §5.5(e)).  A size_max position inherits the IR's size_rule/
    size_target, so §5.3 and §5.4 rule 2 apply unchanged.  Name and
    order identity with the header is guaranteed by
    _check_emit_tables."""
    args: list[Arg] = []
    for a, (oname, otype) in zip(f.args, ABI_OVERRIDE_TABLE[f.name]):
        s = _parse_arg(f.name, f"{otype} {oname}")
        if a.role == "size_max":
            s.size_rule = a.size_rule
            s.size_target = a.size_target
        args.append(s)
    return args


def _emit_args(f: Func) -> list[Arg]:
    """The argument list the emitters (and the pre-emit assertions)
    read for ``f``: the synthetic override list for ABI_OVERRIDE_TABLE
    entries, the IR's args otherwise.  Everything downstream —
    python_name, scalar vs array, dtype, const-ness, writability,
    ``nb::arg``, the ``.pyi`` annotations and the call expression —
    follows the ordinary §5.2–§5.4 rules from this list: one code
    path, not two (§5.5(e)).
    """
    if f.name in ABI_OVERRIDE_TABLE:
        if f.emit_args is None:
            f.emit_args = _override_args(f)
        return f.emit_args
    return f.args


def _abi_override_lines(functions: list[Func]) -> list[str]:
    """File-scope function pointers of the true bind(C) prototypes of
    the ABI_OVERRIDE_TABLE entries (§5.5(e)), emitted once per
    overridden name after de-duplication, inside an anonymous
    namespace.

    The installed header's own prototype cannot be used to make the
    call: the C++ compiler enforces it, and it is wrong (§2).  The
    pointer is deliberately not constexpr: a reinterpret_cast of a
    function pointer is not a constant expression.  If a compiler
    warns -Wcast-function-type, replace this with a std::memcpy
    bit-copy of the function pointer instead of suppressing the
    warning (M3 acceptance: zero warnings outside nanobind).
    """
    L: list[str] = []
    for f in _emitted_functions(functions):
        if f.name not in ABI_OVERRIDE_TABLE:
            continue
        L.append(f"// {f.name}: the installed qmckl.h prototype is wrong")
        L.append("// (§2, §5.5(e)); the call goes through this function")
        L.append("// pointer of the true bind(C) prototype")
        L.append("// (src/qmckl_ao_f.F90, _install/include/qmckl_f.F90).")
        L.extend(_abi_using_lines(f))
        L.append(f"{f.name}_abi_t const {f.name}_abi =")
        L.append(f"    reinterpret_cast<{f.name}_abi_t>(&::{f.name});")
        L.append("")
    if L:
        L.pop()  # trailing blank line
    return L


def _abi_using_lines(f: Func) -> list[str]:
    types = [t for _, t in ABI_OVERRIDE_TABLE[f.name]]
    one = f"using {f.name}_abi_t = qmckl_exit_code (*)({', '.join(types)});"
    if len(one) <= 88:
        return [one]
    lines = [f"using {f.name}_abi_t = qmckl_exit_code (*)("]
    inner = "    "
    cur = ""
    for t in types:
        piece = t if not cur else ", " + t
        if cur and len(inner + cur + piece) > 88:
            lines.append(inner + cur + ",")
            cur = t
        else:
            cur += piece
    lines.append(inner + cur + ");")
    return lines


def _call_head(f: Func) -> str:
    """The head of the emitted C call.  ABI_OVERRIDE_TABLE entries are
    called through the true-prototype function pointer (see
    _abi_override_lines); everything else through the header's
    prototype."""
    if f.name in ABI_OVERRIDE_TABLE:
        return f"{f.name}_abi("
    return f"::{f.name}("


def _assert_emit_names(functions: list[Func], trexio: bool) -> None:
    """Pre-emit assertions of PLAN.md §5.4 rule 1.

    Every argument ``python_name`` becomes an ``nb::arg`` name (a keyword
    name, echoed by the stub) and a C++ lambda parameter name: none may be
    a Python keyword or a C++ keyword, and none may repeat inside a
    function.
    """
    for f in functions:
        for a in _emit_args(f):
            if keyword.iskeyword(a.python_name):
                raise ParseError(f"{f.name}: argument python_name "
                                 f"{a.python_name!r} is a Python keyword")
            if a.python_name in CPP_KEYWORDS:
                raise ParseError(f"{f.name}: argument python_name "
                                 f"{a.python_name!r} is a C++ keyword")
        names = [a.python_name for a in _emit_args(f)]
        if len(set(names)) != len(names):
            raise ParseError(f"{f.name}: duplicate python_name among arguments")
    for f in functions:
        if f.bucket == "trexio" and f.emitted != trexio:
            raise ParseError(f"{f.name}: trexio bucket's emitted flag "
                             f"inconsistent with --trexio/--no-trexio")


def _emitted_functions(functions: list[Func]) -> list[Func]:
    """The declarations that produce a generated binding, in header order,
    de-duplicated by name, first occurrence wins (§5.1, §5.4 rule 5)."""
    seen: set[str] = set()
    out: list[Func] = []
    for f in functions:
        if not f.emitted or f.name in seen:
            continue
        seen.add(f.name)
        out.append(f)
    return out


# ---------------------------------------------------------------------------
# §5.2 / §5.4 / §5.6 per-argument descriptors, shared by both emitters
# (§5.4 rule 7: one descriptor, two emitters — the signature cannot drift).
# ---------------------------------------------------------------------------

# §5.6 stub array element type, from the pointee base type.
_ARRAY_NP = {
    "double": "float64",
    "float": "float32",
    "int64_t": "int64",
    "int32_t": "int32",
    "int": "int32",
    "size_t": "int64",
    "ssize_t": "int64",
    "uint8_t": "uint8",
    "uint16_t": "uint16",
    "uint32_t": "uint32",
    "uint64_t": "uint64",
}


def _scalar_cpp(base: str) -> str:
    """§5.2 C lambda parameter type for a scalar argument."""
    return "bool" if base == "_Bool" else base


def _array_cpp(fname: str, a: Arg) -> str:
    """§5.2 C lambda parameter type for an array argument.  Const-ness
    (and hence writability) comes from the arg's role, which for
    ABI_OVERRIDE_TABLE entries derives from the override's types."""
    const = "const " if a.role == "in_array" else ""
    return f"nb::ndarray<{const}{a.base}, nb::c_contig>"


def _array_stub(a: Arg) -> str:
    """§5.6 stub annotation for an array argument (const or not)."""
    if a.base not in _ARRAY_NP:
        raise ParseError(f"unmapped array element type {a.base!r}")
    return f"NDArray[np.{_ARRAY_NP[a.base]}]"


def _return_class(f: Func) -> tuple[str, str]:
    """§5.4 rule 4 return class — exactly one of four:

    - 'exit' (``qmckl_exit_code``): raise on non-zero via ``check()``;
    - 'value': call and return the C value, no ``check()``;
    - 'scalar_out' (§5.5(a)): pass ``&qmckl_value``, check, return it;
    with the C type of the returned value (the pointee type for
    scalar_out).  Anything else is a generator bug.
    """
    if f.bucket == "scalar_out":
        p = next(a for a in f.args if a.role == "out_array")
        return "scalar_out", _scalar_cpp(p.base)
    rt = f.return_type
    if rt == "qmckl_exit_code":
        return "exit", ""
    if rt in INT_BASES or rt in FLOAT_BASES or rt in ("_Bool", "bool"):
        return "value", _scalar_cpp(rt)
    raise ParseError(f"{f.name}: no §5.4 return class for return type {rt!r}")


def _size_cast_type(c_type: str) -> str:
    """§5.4 rule 2: the cast target is the declared C type of the size_max
    parameter — ``static_cast<int64_t>`` everywhere except ``int64_t const``
    in ``qmckl_get_jastrow_champ_ee_distance_rescaled`` and ``const size_t``
    in ``qmckl_get_en_rescaled_single_gl``: those two header spellings are
    kept verbatim, only the canonical leading-const spelling of the usual
    case (``const int64_t``) is normalised to ``int64_t``."""
    if c_type == "const int64_t":
        return "int64_t"
    return c_type


def _size_expr(f: Func, a: Arg) -> str:
    """The inlined value of a dropped size_max parameter (§5.3, §5.4 rule
    2): the size_target's ``.size()``, or ``std::strlen`` for the trexio
    string, cast to the size_max parameter's declared type.  Never from
    Python, never derived from lda/ldc."""
    if a.size_rule == "strlen":
        inner = f"std::strlen({a.size_target}.c_str())"
    elif a.size_rule in ("bare", "sized"):
        target = next((x for x in _emit_args(f)
                      if x.name == a.size_target), None)
        if target is None:
            raise ParseError(f"{f.name}: size_target {a.size_target!r} of "
                             f"size_max {a.name!r} not found among arguments")
        inner = f"{target.python_name}.size()"
    else:
        raise ParseError(f"{f.name}: size_max {a.name!r} has unknown "
                         f"size_rule {a.size_rule!r}")
    return f"static_cast<{_size_cast_type(a.c_type)}>({inner})"


def _c_call_exprs(f: Func) -> list[str]:
    """The C call's argument expressions, in the descriptor's order, with
    the size_max positions inlined (§5.4).  For ABI_OVERRIDE_TABLE
    entries the descriptor is the override's synthetic list, and the
    call head (see _call_head) goes through the true-prototype pointer."""
    exprs: list[str] = []
    for a in _emit_args(f):
        if a.role == "context":
            exprs.append("ctx.h")
        elif a.role == "string":
            exprs.append(f"{a.python_name}.c_str()")
        elif a.role == "size_max":
            exprs.append(_size_expr(f, a))
        elif a.role in ("in_array", "out_array"):
            if f.bucket == "scalar_out":
                exprs.append("&qmckl_value")
            else:
                exprs.append(f"{a.python_name}.data()")
        elif a.role == "scalar":
            exprs.append(a.python_name)
        else:
            raise ParseError(f"{f.name}: unemittable argument {a.name!r} "
                             f"(role {a.role!r})")
    return exprs


def _lambda_params(f: Func) -> list[str]:
    """The C lambda parameters: the Python signature (C order minus every
    size_max, minus the scalar_out buffer) with §5.2 types."""
    params: list[str] = []
    if f.has_context:
        params.append("qmckl::Context &ctx")
    for a in _emit_args(f):
        if a.role in ("context", "size_max"):
            continue
        if f.bucket == "scalar_out" and a.role == "out_array":
            continue
        if a.role == "string":
            t = "std::string"
        elif a.role in ("in_array", "out_array"):
            t = _array_cpp(f.name, a)
        else:
            t = _scalar_cpp(a.base)
        params.append(f"{t} {a.python_name}")
    return params


def _arg_descriptors(f: Func) -> list[str]:
    """The trailing ``nb::arg`` descriptors (§5.4 rules 1 and 3): every
    parameter keeps its python_name; ``.noconvert()`` on ``nb::ndarray``
    arguments only — scalars keep nanobind's default conversion."""
    descs: list[str] = []
    for a in _emit_args(f):
        if a.role == "size_max":
            continue
        if f.bucket == "scalar_out" and a.role == "out_array":
            continue
        d = f'"{a.python_name}"_a'
        if a.role in ("in_array", "out_array"):
            d += ".noconvert()"
        descs.append(d)
    return descs


def _wrapped(indent: str, head: str, items: list[str], close: str) -> list[str]:
    """`head<item, item, …close>` on one line when it fits in 88 columns.

    Otherwise the head stays alone on the first line (the §5.4 template
    shape) and the items wrap at indent+4, the closing token on the last
    item's line.  With an empty head (the nb::arg descriptor list) the items
    wrap at indent itself.
    """
    one = f"{indent}{head}{', '.join(items)}{close}"
    if len(one) <= 88:
        return [one]
    lines: list[str] = []
    base = indent
    if head:
        lines.append(f"{indent}{head}")
        base = indent + "    "
    cur = ""
    for e in items:
        piece = e if not cur else ", " + e
        if cur and len(base + cur + piece) > 88:
            lines.append(base + cur + ",")
            cur = e
        else:
            cur += piece
    lines.append(base + cur + close)
    return lines


def _mdef_lines(f: Func) -> list[str]:
    """One m.def per binding: the §5.4 template shape, with the four return
    classes of rule 4.  The only local is ``qmckl_rc`` (plus ``qmckl_value``
    in the value and scalar_out classes, as shown in §5.4's scalar-out
    template); the GIL is released only around the C call."""
    rclass, value_type = _return_class(f)
    indent = " " * 14
    exprs = _c_call_exprs(f)
    body: list[str] = []
    if rclass in ("exit", "scalar_out"):
        if rclass == "scalar_out":
            body.append(f"{indent}{value_type} qmckl_value{{}};")
        body.append(f"{indent}qmckl_exit_code qmckl_rc;")
        body.append(f"{indent}{{")
        body.append(f"{indent}    nb::gil_scoped_release nogil;")
        body.extend(_wrapped(indent + "    ", f"qmckl_rc = {_call_head(f)}",
                             exprs, ");"))
        body.append(f"{indent}}}")
        # The no-context branch exists so a future header cannot produce an
        # unchecked call; check() skips the qmckl_get_error read for a NULL
        # context (§6.1).  No such function exists on 1.1.0.
        ctx = "ctx.h" if f.has_context else "QMCKL_NULL_CONTEXT"
        body.append(f'{indent}qmckl::check({ctx}, qmckl_rc, "{f.name}");')
        if rclass == "scalar_out":
            body.append(f"{indent}return qmckl_value;")
    else:  # "value": call and return the C value, no check()
        body.append(f"{indent}{value_type} qmckl_value{{}};")
        body.append(f"{indent}{{")
        body.append(f"{indent}    nb::gil_scoped_release nogil;")
        body.extend(_wrapped(indent + "    ", f"qmckl_value = {_call_head(f)}",
                             exprs, ");"))
        body.append(f"{indent}}}")
        body.append(f"{indent}return qmckl_value;")

    params = _lambda_params(f)
    lines = [f'    m.def("{f.python_name}",']
    if len(params) == 1:
        lines.append(f"          []({params[0]}) {{")
    else:
        lines.append(f"          []({params[0]},")
        for p in params[1:-1]:
            lines.append(f"            {p},")
        lines.append(f"            {params[-1]}) {{")
    lines.extend(body)
    lines.append("          },")
    lines.extend(_wrapped("          ", "", _arg_descriptors(f), ");"))
    return lines


def _pyi_def_line(f: Func) -> str:
    """§5.6: one stub def per binding — same name, order and keyword names
    as ``register_generated``, after the same de-duplication.  Context
    first; every parameter positional-or-keyword."""
    rclass, _ = _return_class(f)
    ps: list[str] = []
    for a in _emit_args(f):
        if a.role == "size_max":
            continue
        if f.bucket == "scalar_out" and a.role == "out_array":
            continue
        if a.role == "context":
            t = "_Context"
        elif a.role in ("in_array", "out_array"):
            t = _array_stub(a)
        else:
            t = _scalar_python(a.base)
        ps.append(f"{a.python_name}: {t}")
    if rclass == "exit":
        ret = "None"  # raise on non-zero (§5.2)
    elif rclass == "value":
        ret = f.return_python
    else:  # scalar_out: the pointee type (§5.2 return mapping)
        ret = _scalar_python(next(a for a in f.args
                                   if a.role == "out_array").base)
    return f"def {f.python_name}({', '.join(ps)}) -> {ret}: ..."


def emit_cpp(functions: list[Func], trexio: bool,
             codes: list[tuple[str, int]], header: str, path: Path) -> None:
    """M3: emit the full C++ translation unit (PLAN.md §5.4, §5.5, §5.7).

    Contains ``qmckl::code_name`` (a switch over the bare integer values,
    so the generated file does not depend on macros a future header could
    rename), ``qmckl_python::register_error_codes`` (the ``ERROR_CODES``
    dict of the extension) and one ``m.def`` per generated binding inside
    ``qmckl_python::register_generated`` (§5.4).  With ``--no-trexio`` the
    ``qmckl_trexio_read`` binding is not emitted at all (§5.5(d)).
    """
    _assert_emit_names(functions, trexio)
    _check_emit_tables(functions)
    L: list[str] = []
    a = L.append
    a("// generated_bindings.cpp")
    a("//")
    a(f"// Generated by csrc/generate.py from {header} -- DO NOT EDIT.")
    a("//")
    a("// Milestone M3 (PLAN.md §5.4, §5.5, §5.7): the error-code table read")
    a("// from the QMCKL_* macros of the installed header, plus one m.def")
    a("// per generated binding (de-duplicated by name, first occurrence")
    a(f"// wins); the TREXIO binding is {'emitted' if trexio else 'skipped'}.")
    a("")
    a("#include <nanobind/nanobind.h>")
    a("#include <nanobind/ndarray.h>")
    a("#include <nanobind/stl/string.h>")
    a("#include <cstring>")
    a("")
    a('#include "context.hpp"')
    a("")
    a("// nanobind 3.x renamed its namespace from `nb` to `nanobind`.")
    a("namespace nb = nanobind;")
    a("using namespace nb::literals;")
    a("")
    abi_lines = _abi_override_lines(functions)
    if abi_lines:
        a("namespace {")
        a("")
        L.extend(abi_lines)
        a("")
        a("}  // namespace")
        a("")
    a("namespace qmckl {")
    a("")
    a("const char *code_name(int32_t rc) {")
    a("    switch (rc) {")
    for name, value in codes:
        a(f"    case {value}:")
        a(f'        return "{name}";')
    a("    default:")
    a('        return "QMCKL_UNKNOWN";')
    a("    }")
    a("}")
    a("")
    a("}  // namespace qmckl")
    a("")
    a("namespace qmckl_python {")
    a("")
    emitted = _emitted_functions(functions)
    if emitted:
        a("void register_generated(nb::module_ &m) {")
        for f in emitted:
            L.extend(_mdef_lines(f))
            a("")
        a("}")
    else:  # never happens on a >= 1.1 header; keeps the TU warning-free
        a("void register_generated(nb::module_ &m) { (void) m; }")
    a("")
    a("void register_error_codes(nb::module_ &m) {")
    a("    nb::dict codes;")
    for name, value in codes:
        a(f'    codes["{name}"] = {value};')
    a('    m.attr("ERROR_CODES") = codes;')
    a("}")
    a("")
    a("}  // namespace qmckl_python")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def emit_pyi(functions: list[Func], trexio: bool, header: str, path: Path) -> None:
    """M3: emit the _core.pyi stub (PLAN.md §5.6).

    The emitter owns the whole file: the hand-written surface plus one def
    per generated binding between the ``BEGIN/END GENERATED BINDINGS``
    markers (with the imports the annotations need).  The emitter runs
    ``ast.parse()`` on the finished text and exits 1 when the stub is not
    valid Python: an unparsable stub silently degrades every annotation to
    ``Any``, which is worse than a failed build.
    """
    _assert_emit_names(functions, trexio)
    _check_emit_tables(functions)
    L: list[str] = []
    a = L.append
    a(f"# _core.pyi -- generated by csrc/generate.py from {header} -- DO NOT EDIT.")
    a("#")
    a("# Milestone M3 (PLAN.md §5.6): the hand-written surface of the")
    a("# qmckl._core extension, plus one def per generated binding.")
    a("# The module exposes nothing else at top level.")
    a("")
    a("ERROR_CODES: dict[str, int]")
    a("")
    a("")
    a("class _Context:")
    a("    def __init__(self) -> None: ...")
    a("    def __init__(self, other: _Context) -> None: ...")
    a("    def touch(self) -> None: ...")
    a("    def valid(self) -> bool: ...")
    a("    @property")
    a("    def handle(self) -> int: ...")
    a("")
    a("")
    a("class QMCKLStatusError(RuntimeError): ...")
    a("")
    a("")
    a("def last_error(ctx: _Context) -> str: ...")
    a("")
    a("def get_ao_basis_type(ctx: _Context) -> str: ...")
    a("")
    a("")
    a("# BEGIN GENERATED BINDINGS (M3 inserts one def per generated symbol)")
    a("import numpy as np")
    a("from numpy.typing import NDArray")
    a("")
    a("")
    for f in _emitted_functions(functions):
        a(_pyi_def_line(f))
        a("")
    a("# END GENERATED BINDINGS")
    text = "\n".join(L) + "\n"
    try:
        ast.parse(text)
    except SyntaxError as e:
        raise ParseError(f"generated _core.pyi is not valid Python: {e}")
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate.py",
        description="Parse an installed qmckl.h and classify every qmckl_* "
                    "declaration (PLAN.md §5).")
    p.add_argument("--header", required=True, metavar="PATH/qmckl.h",
                   help="installed qmckl.h to parse")
    p.add_argument("--cpp", metavar="OUT.cpp",
                   help="C++ emitter output (§5.7 error table + one m.def "
                        "per generated binding, §5.4)")
    p.add_argument("--pyi", metavar="OUT.pyi",
                   help="stub emitter output (§5.6: the hand-written surface "
                        "plus one def per generated binding)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--trexio", dest="trexio", action="store_true", default=True,
                   help="the linked library exports qmckl_trexio_read (default)")
    g.add_argument("--no-trexio", dest="trexio", action="store_false",
                   help="the linked library lacks qmckl_trexio_read")
    p.add_argument("--dump-json", metavar="OUT.json",
                   help="write the IR (one record per function) as JSON")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.cpp and not args.pyi and not args.dump_json:
        print("generate.py: nothing to do: pass --dump-json and/or "
              "--cpp/--pyi", file=sys.stderr)
        return 2
    try:
        functions = parse_header(Path(args.header))
        classify(functions, trexio=args.trexio)
        # The §5.7 error-code table is a second, independent pass over the
        # *raw* header text (the §5.1 pipeline has already deleted the
        # preprocessor lines), so re-read the file rather than reusing the
        # stripped text.  Needed only by the emitters; --dump-json keeps
        # its M1 behaviour byte for byte.
        codes: Optional[list[tuple[str, int]]] = None
        if args.cpp or args.pyi:
            raw = Path(args.header).read_text(encoding="utf-8")
            codes = parse_error_codes(raw)
    except ParseError as e:
        print(f"generate.py: error: {e}", file=sys.stderr)
        return 1
    print_summary(functions, args.trexio)
    if codes is not None:
        first, last = codes[0], codes[-1]
        print(f"error codes: {len(codes)} macros "
              f"({first[0]}={first[1]} .. {last[0]}={last[1]})")
    if args.dump_json:
        Path(args.dump_json).write_text(
            to_json(functions, args.trexio, args.header) + "\n", encoding="utf-8")
    if args.cpp:
        emit_cpp(functions, args.trexio, codes, args.header, Path(args.cpp))
    if args.pyi:
        emit_pyi(functions, args.trexio, args.header, Path(args.pyi))
    return 0


if __name__ == "__main__":
    sys.exit(main())
