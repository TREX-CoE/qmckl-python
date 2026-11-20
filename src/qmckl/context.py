"""The Pythonic :class:`Context` (PLAN.md §7.2).

The integer and array properties wrap the scalar-out getters and the
``size_max``-carrying array getters; the setters accept any array-like and
convert it to a C-contiguous ``float64`` array.  Shapes are validated here,
because the raw layer cannot validate them (PLAN.md §11, trap 3).
"""

import numpy as np

from . import _core

__all__ = ["Context"]

try:  # a QMCkl built without TREXIO does not export qmckl_trexio_read
    _trexio_read = _core.trexio_read
except AttributeError:
    _trexio_read = None


def _coords(x, ndim):
    """C-contiguous ``float64`` coordinates of rank ``ndim``, last axis 3."""
    a = np.ascontiguousarray(x, dtype=np.float64)
    if a.ndim != ndim or a.shape[-1] != 3:
        raise ValueError(
            f"coordinates must have {ndim} dimensions and a last dimension "
            f"of 3, got shape {a.shape}"
        )
    return a


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

    # -- TREXIO (PLAN.md §7.2) ----------------------------------------

    def trexio_read(self, path):
        """Load the TREXIO file at ``path`` into this context."""
        if _trexio_read is None:
            raise NotImplementedError("this QMCkl build has no TREXIO support")
        _trexio_read(self, str(path))

    # -- integers (thin wrappers over the scalar-out getters) ---------

    @property
    def elec_num(self):
        return _core.get_electron_num(self)

    @property
    def nucl_num(self):
        return _core.get_nucleus_num(self)

    @property
    def point_num(self):
        return _core.get_point_num(self)

    @property
    def walk_num(self):
        return _core.get_electron_walk_num(self)

    @property
    def ao_num(self):
        return _core.get_ao_basis_ao_num(self)

    @property
    def mo_num(self):
        return _core.get_mo_basis_mo_num(self)

    @property
    def prim_num(self):
        return _core.get_ao_basis_prim_num(self)

    @property
    def shell_num(self):
        return _core.get_ao_basis_shell_num(self)

    # -- array properties ---------------------------------------------

    def _get(self, getter, shape):
        buf = np.empty(shape, dtype=np.float64)
        getter(self, buf)
        return buf

    def _eval(self, getter, shape):
        # With point_num == 0 the provider allocates zero bytes, and QMCkl
        # 1.1.0 records that as a free slot; the next allocation overwrites
        # the record, so the context's allocation counter never returns to
        # zero and an assert-enabled qmckl_context_destroy aborts
        # (PLAN.md §11, trap 9).  Refuse instead.
        if shape[0] == 0:
            raise ValueError(
                "no points in the context: set ctx.point (or "
                "ctx.electron_coord) before evaluating the basis"
            )
        return self._get(getter, shape)

    @property
    def point(self):
        buf = np.empty((self.point_num, 3), dtype=np.float64)
        _core.get_point(self, "N", buf)
        return buf

    @point.setter
    def point(self, coords):
        coords = _coords(coords, 2)
        _core.set_point(self, "N", coords.shape[0], coords)

    @property
    def nucl_coord(self):
        buf = np.empty((self.nucl_num, 3), dtype=np.float64)
        _core.get_nucleus_coord(self, "N", buf)
        return buf

    @nucl_coord.setter
    def nucl_coord(self, coords):
        coords = _coords(coords, 2)
        n = coords.shape[0]
        try:
            current = _core.get_nucleus_num(self)
        except _core.QMCKLStatusError as exc:
            if not str(exc).startswith("[QMCKL_NOT_PROVIDED]"):
                raise
            # qmckl_set_nucleus_coord takes no num: it uses the stored one.
            _core.set_nucleus_num(self, n)
        else:
            if current != n:
                raise ValueError(
                    f"nucleus_num is already {current}, cannot set {n} "
                    "coordinates; build a new Context to change it"
                )
        _core.set_nucleus_coord(self, "N", coords)

    @property
    def electron_coord(self):
        buf = np.empty((self.walk_num, self.elec_num, 3), dtype=np.float64)
        _core.get_electron_coord(self, "N", buf)
        return buf

    @electron_coord.setter
    def electron_coord(self, coords):
        coords = _coords(coords, 3)
        elec_num = self.elec_num
        if coords.shape[1] != elec_num:
            raise ValueError(
                f"electron_coord has {coords.shape[1]} electrons per walker, "
                f"but elec_num is {elec_num}"
            )
        _core.set_electron_coord(self, "N", coords.shape[0], coords)

    @property
    def ao_value(self):
        return self._eval(_core.get_ao_basis_ao_value, (self.point_num, self.ao_num))

    @property
    def ao_vgl(self):
        return self._eval(
            _core.get_ao_basis_ao_vgl, (self.point_num, 5, self.ao_num)
        )

    @property
    def mo_value(self):
        return self._eval(_core.get_mo_basis_mo_value, (self.point_num, self.mo_num))

    @property
    def mo_vgl(self):
        return self._eval(
            _core.get_mo_basis_mo_vgl, (self.point_num, 5, self.mo_num)
        )

    @property
    def ee_distance(self):
        return self._get(
            _core.get_electron_ee_distance,
            (self.walk_num, self.elec_num, self.elec_num),
        )

    @property
    def en_distance(self):
        return self._get(
            _core.get_electron_en_distance, (self.point_num, self.nucl_num)
        )

    @property
    def ee_potential(self):
        return self._get(_core.get_electron_ee_potential, (self.walk_num,))

    @property
    def en_potential(self):
        return self._get(_core.get_electron_en_potential, (self.walk_num,))
