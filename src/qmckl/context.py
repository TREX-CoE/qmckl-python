"""The Pythonic :class:`Context` (PLAN.md §7.2).

Milestone M2 ships exactly the class below — no properties, no setters.
M4 adds the scalar/array properties and the setters in this same file.
"""

from . import _core

__all__ = ["Context"]


class Context(_core._Context):
    """A QMCkl context: the Pythonic wrapper over :class:`_core._Context`."""

    @classmethod
    def copy(cls, other):
        """Return a deep copy of ``other`` (PLAN.md §2: qmckl_context_copy)."""
        return cls(other)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # The GC owns destruction; touching here marks any derived data dirty.
        self.touch()
