"""M2: error handling (PLAN.md §8, test_errors.py, M2 rows).

Uses only what exists at M2: no generated bindings yet, no ``set_point``.
"""

import pytest

import qmckl


def test_get_ao_basis_type_fresh_context_raises():
    # A fresh context has no AO basis, so the getter must fail with
    # QMCKL_NOT_PROVIDED, and the message must not leak the broken
    # "Unknown error" default of the released 1.1.0 qmckl_string_of_error.
    ctx = qmckl.Context()
    with pytest.raises(qmckl.QMCKLStatusError) as ei:
        qmckl.get_ao_basis_type(ctx)
    msg = str(ei.value)
    assert msg.startswith("[QMCKL_NOT_PROVIDED]")
    assert "Unknown error" not in msg


def test_error_codes_success_is_zero():
    assert qmckl.ERROR_CODES["QMCKL_SUCCESS"] == 0


def test_error_name_inverts_table():
    # 103 is QMCKL_INVALID_CONTEXT in the installed header.
    assert qmckl.error_name(103) == "QMCKL_INVALID_CONTEXT"
    assert qmckl.error_name(qmckl.ERROR_CODES["QMCKL_SUCCESS"]) == "QMCKL_SUCCESS"


def test_error_name_unknown_code():
    # A value with no macro name renders as QMCKL_UNKNOWN(<code>).
    assert qmckl.error_name(-1) == "QMCKL_UNKNOWN(-1)"


def test_last_error_returns_str():
    ctx = qmckl.Context()
    assert isinstance(qmckl.last_error(ctx), str)
