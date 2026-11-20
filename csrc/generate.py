#!/usr/bin/env python3
"""Generate the qmckl-python binding inventory from an installed qmckl.h.

Milestone M2 of PLAN.md (qmckl-python spec v1): the header parser, the
function classification, ``--dump-json``, the §5.7 error-code table read
from the raw header text, and the M2 skeleton emission: ``--cpp`` writes the
C++ translation unit holding ``qmckl::code_name``,
``qmckl_python::register_error_codes`` and an empty
``qmckl_python::register_generated`` (M3 fills it with the per-function
bindings), and ``--pyi`` writes the hand-written ``_core.pyi`` surface.

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
  excluded           (d) not exposed at all (6)
  trexio             (e) qmckl_trexio_read; codegen only when the linked
                         library exports it (``--trexio`` / ``--no-trexio``)
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
  * a Python keyword among the argument names after ``_`` suffixing.

The §2 verified facts (239 declarations, 223 ``qmckl_exit_code`` returns, 125
getter/setter-shape functions) are reported and warned about, not fatal: the
bindings are generated from whatever 1.x header is installed, so a 1.1.1 that
adds or de-duplicates a declaration must still build.  Loss of a §5.5 table
name *is* fatal, which is what protects the hand-audited classification.

Usage:
  generate.py --header PATH/qmckl.h --dump-json OUT.json [--trexio|--no-trexio]
  generate.py --header PATH/qmckl.h --cpp OUT.cpp --pyi OUT.pyi [--trexio|--no-trexio]

In M2, ``--cpp``/``--pyi`` emit the compilable skeleton of PLAN.md §10:
the §5.7 error-code table (``code_name``, ``register_error_codes``) plus an
empty ``register_generated`` / the hand-written stub surface.  M3 grows that
skeleton into the per-function loop without changing the CLI, the file
names, or the CMake integration.

Standard library only, Python >= 3.10.
"""

from __future__ import annotations

import argparse
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
        raise ParseError(f"{func.name}: §5.5(e) table entry but shape is {roles}")


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
    print(f"qmckl.h declarations: {len(functions)} "
          f"({len(set(names))} unique names"
          + (f"; duplicates: {sorted(dups)}" if dups else "") + ")")
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
    print(f"emitted (codegen): {sum(1 for f in functions if f.emitted)}"
          + ("" if trexio else " [trexio_read dropped: --no-trexio]"))


# ---------------------------------------------------------------------------
# Emitters (M2 skeleton; M3 grows them without touching the CLI, the file
# names, or the CMake integration — PLAN.md §10)
# ---------------------------------------------------------------------------

def emit_cpp(codes: list[tuple[str, int]], header: str, path: Path) -> None:
    """M2: emit the C++ translation unit (PLAN.md §5.7, §10).

    Contains ``qmckl::code_name`` (a switch over the bare integer values,
    so the generated file does not depend on macros a future header could
    rename), ``qmckl_python::register_error_codes`` (the ``ERROR_CODES``
    dict of the extension) and an empty ``register_generated`` that M3
    fills with the per-function bindings.
    """
    L: list[str] = []
    a = L.append
    a("// generated_bindings.cpp")
    a("//")
    a(f"// Generated by csrc/generate.py from {header} -- DO NOT EDIT.")
    a("//")
    a("// Milestone M2 (PLAN.md §5.7, §10): the error-code table read from")
    a("// the QMCKL_* macros of the installed header, plus the empty")
    a("// register_generated() that M3 fills with the per-function bindings.")
    a("")
    a("#include <nanobind/nanobind.h>")
    a("")
    a('#include "context.hpp"')
    a("")
    a("// nanobind 3.x renamed its namespace from `nb` to `nanobind`.")
    a("namespace nb = nanobind;")
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
    a("// M2: no per-function bindings yet; M3 emits one m.def per")
    a("// generated symbol here (PLAN.md §5.4).")
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


def emit_pyi(header: str, path: Path) -> None:
    """M2: emit the _core.pyi stub (PLAN.md §5.6).

    The hand-written surface only.  M3 inserts the per-function defs
    between the BEGIN/END GENERATED BINDINGS markers.
    """
    L: list[str] = []
    a = L.append
    a(f"# _core.pyi -- generated by csrc/generate.py from {header} -- DO NOT EDIT.")
    a("#")
    a("# Milestone M2 (PLAN.md §5.6): the hand-written surface of the")
    a("# qmckl._core extension.  The module exposes nothing else at top level.")
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
    a("# END GENERATED BINDINGS")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


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
                   help="C++ emitter output (§5.7 error table + "
                        "register_generated; empty binding set in M2)")
    p.add_argument("--pyi", metavar="OUT.pyi",
                   help="stub emitter output (the hand-written §5.6 surface "
                        "in M2)")
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
        emit_cpp(codes, args.header, Path(args.cpp))
    if args.pyi:
        emit_pyi(args.header, Path(args.pyi))
    return 0


if __name__ == "__main__":
    sys.exit(main())
