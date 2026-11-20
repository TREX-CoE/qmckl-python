"""Python bindings for QMCkl.

Milestone M2 of PLAN.md: the hand-written layer around the generated core
extension ``qmckl._core`` — the Pythonic :class:`Context` (``context.py``)
and the error-code table (``errors.py``).  The per-function generated
bindings are added by M3; ``from ._core import *`` re-exports them as they
appear.
"""

from . import _core
from ._core import *  # noqa: F401,F403 -- generated functions + exception/table
from .context import Context
from .errors import ERROR_CODES, error_name

# One source of truth: every public name of the extension, plus the Pythonic
# layer.  ``import *`` skips ``_Context`` (leading underscore), which is
# intended: users construct ``Context``.
__all__ = sorted(
    {"Context", "error_name"} | {n for n in dir(_core) if not n.startswith("_")}
)
