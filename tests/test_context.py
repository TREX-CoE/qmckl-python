"""M2: the Context lifecycle (PLAN.md §8, test_context.py, M2 rows).

Uses only what exists at M2: no generated bindings yet, no ``set_point``.
"""

import gc

import qmckl


def test_create():
    ctx = qmckl.Context()
    assert ctx.valid()


def test_handle():
    ctx = qmckl.Context()
    assert isinstance(ctx.handle, int)
    assert ctx.handle != 0


def test_touch():
    ctx = qmckl.Context()
    ctx.touch()
    ctx.touch()
    assert ctx.valid()


def test_gc_no_crash():
    # Creating and dropping contexts must not crash: the RAII dtor destroys
    # each handle exactly once (PLAN.md §2: deep copy, no refcounting).
    for _ in range(32):
        qmckl.Context()
    gc.collect()


def test_copy_returns_distinct_valid_context():
    ctx = qmckl.Context()
    copied = qmckl.Context.copy(ctx)
    assert copied is not ctx
    assert copied.valid()
    assert copied.handle != ctx.handle
    assert ctx.valid()
    del copied
    gc.collect()
    assert ctx.valid()


def test_copy_init():
    # The copy __init__ overload of _core._Context.
    ctx = qmckl.Context()
    copied = qmckl.Context(ctx)
    assert copied.valid()
    assert copied.handle != ctx.handle


def test_context_manager():
    # __exit__ calls touch(); destruction is left to the GC.
    with qmckl.Context() as ctx:
        assert ctx.valid()
    gc.collect()
