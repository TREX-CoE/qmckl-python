"""M3: the generated surface (PLAN.md §8, test_surface.py).

The stub is found next to the installed ``_core`` extension: the
directory of ``qmckl._core.__file__`` holds ``_core.pyi`` in both
editable and wheel installs (verified in M2, PLAN.md §4.2).

"Generated names in ``dir(_core)``" means ``set(dir(_core))`` minus the
five hand-written names, with every dunder name excluded.
"""

import ast
import pathlib

import qmckl
from qmckl import _core

# The hand-written part of the extension surface (PLAN.md §5.6, §6.2).
_HAND_WRITTEN = {
    "ERROR_CODES",
    "QMCKLStatusError",
    "_Context",
    "last_error",
    "get_ao_basis_type",
}


def _stub_text() -> str:
    return (pathlib.Path(_core.__file__).resolve().parent / "_core.pyi").read_text(
        encoding="utf-8"
    )


def _stub_generated_defs() -> list[str]:
    """The def names between the BEGIN/END GENERATED BINDINGS markers."""
    text = _stub_text()
    begin, end = "# BEGIN GENERATED BINDINGS", "# END GENERATED BINDINGS"
    block = text[text.index(begin) : text.index(end)]
    return [n.name for n in ast.parse(block).body if isinstance(n, ast.FunctionDef)]


def _generated_names() -> set[str]:
    return {n for n in dir(_core) if n not in _HAND_WRITTEN and not n.startswith("__")}


def test_stub_names_all_in_module():
    """Every top-level name of _core.pyi is in dir(_core)."""
    tree = ast.parse(_stub_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    assert names, "no top-level names found in _core.pyi"
    missing = names - set(dir(_core))
    assert not missing, f"stub names absent from _core: {sorted(missing)}"


def test_generated_def_count_matches_module():
    stub_defs = _stub_generated_defs()
    gen_names = _generated_names()
    assert len(stub_defs) == len(gen_names)
    # 223 generated bindings with TREXIO, 222 without (§5.4 rule 5,
    # §5.5(d): 225 emitted declarations de-duplicated by name, trexio
    # dropped when the library does not export it).
    expected = 223 if hasattr(_core, "trexio_read") else 222
    assert len(stub_defs) == expected
    # Same names as register_generated (§5.6, §5.4 rule 8: the two
    # surfaces cannot drift apart).
    assert set(stub_defs) == gen_names


def test_required_names_present():
    for name in (
        "get_mo_basis_mo_value",
        "dgemm_safe",
        "test_precision_64",
        "get_point_num",
        "ao_gaussian_vgl",
    ):
        assert name in dir(_core), f"{name} missing from _core"
        assert name in _stub_generated_defs(), f"{name} missing from the stub"


def test_trexio_read_present_exactly_when_exported():
    # The build's nm probe decides whether the linked library exports
    # qmckl_trexio_read; the runtime and the stub must agree with that
    # decision (PLAN.md §5.5(d): no binding AND no stub line otherwise).
    in_runtime = hasattr(_core, "trexio_read")
    in_stub = "trexio_read" in _stub_generated_defs()
    assert in_runtime == in_stub
    if in_runtime:
        assert callable(_core.trexio_read)


def test_deliberately_absent_names():
    # §5.5(c)/§5.5(d): bound by hand in module.cpp or not exposed at
    # all.  qmckl_check is excluded (PLAN.md §5.5(c): it prints, it
    # reads an uninitialised buffer, and it can return success for a
    # non-zero rc).
    for name in ("string_of_error", "get_error", "check", "failwith", "lock"):
        assert name not in dir(_core), f"{name} must not be exposed"
        assert name not in _stub_generated_defs(), f"{name} must not be stubbed"


def test_hand_written_surface_present():
    assert "_Context" in dir(_core)
    assert "ERROR_CODES" in dir(_core)
    assert "QMCKLStatusError" in dir(_core)


def test_package_all_matches_public_names():
    public = {n for n in dir(_core) if not n.startswith("_")}
    assert set(qmckl.__all__) == public | {"Context", "error_name"}
    for name in qmckl.__all__:
        assert hasattr(qmckl, name), f"__all__ name {name} not importable"
