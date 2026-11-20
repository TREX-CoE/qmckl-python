// csrc/module.cpp
//
// The nanobind module qmckl._core (PLAN.md §6.2): the exception type, the
// RAII _Context class, the two hand-written scalar bindings, and the calls
// into the generated translation unit.
//
// nanobind 3.x renamed its namespace from `nb` to `nanobind`; the alias
// below keeps the PLAN.md `nb::` spelling compiling verbatim.  And
// nanobind 3.1.0 has no pybind11-style nb::register_exception<T>() free
// function — its nb::exception<T> constructor performs the identical task
// (create the Python exception type in the module, subclassing the given
// base, and register the C++-to-Python translator).

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>  // std::string <-> str casters (last_error, get_ao_basis_type)

#include "context.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace qmckl_python {
void register_error_codes(nb::module_ &m);  // generated (PLAN.md §5.7)
void register_generated(nb::module_ &m);    // generated (PLAN.md §5.4; empty in M2)
}  // namespace qmckl_python

NB_MODULE(_core, m) {
    nb::exception<qmckl::QMCKLStatusError>(
        m, "QMCKLStatusError", PyExc_RuntimeError);

    nb::class_<qmckl::Context>(m, "_Context")
        .def(nb::init<>())
        .def(nb::init<const qmckl::Context &>())
        .def("touch", [](qmckl::Context &c) {
            qmckl::check(c.h, ::qmckl_context_touch(c.h), "qmckl_context_touch");
        })
        .def("valid", [](const qmckl::Context &c) {
            return ::qmckl_context_check(c.h) != QMCKL_NULL_CONTEXT;
        })
        .def_prop_ro("handle", [](const qmckl::Context &c) { return c.h; });

    // Does NOT call ::qmckl_last_error: that C function formats through the
    // broken qmckl_string_of_error of the 1.1.0 tarball (PLAN.md §2, §6.2).
    m.def("last_error", [](const qmckl::Context &ctx) {
        std::string msg;
        {
            nb::gil_scoped_release nogil;
            msg = qmckl::last_error_message(ctx.h);
        }
        return msg;
    }, "ctx"_a);

    // The C function writes exactly one character, with no NUL terminator.
    m.def("get_ao_basis_type", [](const qmckl::Context &ctx) {
        char b = 0;
        qmckl_exit_code rc;
        {
            nb::gil_scoped_release nogil;
            rc = ::qmckl_get_ao_basis_type(ctx.h, &b);
        }
        qmckl::check(ctx.h, rc, "qmckl_get_ao_basis_type");
        return std::string(1, b);
    }, "ctx"_a);

    // Order matters: the exception and _Context are registered before the
    // generated functions reference them in signatures (PLAN.md §6.2).
    qmckl_python::register_error_codes(m);
    qmckl_python::register_generated(m);
}
