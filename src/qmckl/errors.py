"""Error-code helpers (PLAN.md §7.3).

The table comes from the QMCKL_* macros of the installed header, generated
into the extension by csrc/generate.py (§5.7) — nothing is retyped by hand,
the header is the single source of truth.  ``error_name()`` inverts it.
"""

from ._core import ERROR_CODES

__all__ = ["ERROR_CODES", "error_name"]

_NAME_BY_CODE: dict[int, str] = {code: name for name, code in ERROR_CODES.items()}


def error_name(code: int) -> str:
    """Symbolic name of a QMCkl exit code; ``QMCKL_UNKNOWN(<code>)`` if unknown."""
    return _NAME_BY_CODE.get(code, f"QMCKL_UNKNOWN({code})")
