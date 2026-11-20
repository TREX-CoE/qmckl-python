"""Python bindings for QMCkl.

The hand-written layer around the generated core extension
``qmckl._core``: the Pythonic :class:`Context` (``context.py``), the
error-code table (``errors.py``) and the row-major ``dgemm``
(``blas.py``).  The per-function generated bindings are re-exported by
``from ._core import *`` exactly as they appear.
"""

from . import _core
from ._core import *  # noqa: F401,F403 -- generated functions + exception/table
from .context import Context
from .errors import ERROR_CODES, error_name
from .blas import dgemm  # noqa: F401 -- shadows the raw _core.dgemm on purpose

# One source of truth: every public name of the extension, plus the Pythonic
# layer.  ``import *`` skips ``_Context`` (leading underscore), which is
# intended: users construct ``Context``.
__all__ = sorted(
    {"Context", "error_name"} | {n for n in dir(_core) if not n.startswith("_")}
)
