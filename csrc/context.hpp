// csrc/context.hpp
//
// The RAII wrapper around a QMCkl context handle, the QMCkl status
// exception, and the error-message helpers (PLAN.md §6.1).
//
// This layer uses only the QMCkl public header and the C++ standard
// library; csrc/module.cpp ties it to Python through nanobind.

#ifndef QMCKL_PYTHON_CONTEXT_HPP
#define QMCKL_PYTHON_CONTEXT_HPP

#include <qmckl.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace qmckl {

struct Context {
    int64_t h;

    Context();                        // qmckl_context_create(); throw if NULL
    Context(const Context &other);    // qmckl_context_copy(); throw if NULL (deep copy)
    Context &operator=(const Context &) = delete;
    ~Context();                       // qmckl_context_destroy(h); errors swallowed
};

struct QMCKLStatusError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Throw qmckl::QMCKLStatusError when rc != 0 (PLAN.md §6.1).
//
// The message is "[<CODE_NAME>] <fname>: <message>", or
// "[<CODE_NAME>] <fname>" when the context holds no message to report.
// <CODE_NAME> comes from the generated code_name() table and never fails.
// The stored message is used only when the context is valid *and* the stored
// exit code equals rc: QMCkl stores the last error per context, and several
// functions return a failure code without writing it, so an unchecked read
// would print a stale message from an unrelated earlier call.  An empty
// message, or exactly "Unknown error" (what the released QMCkl 1.1.0
// qmckl_string_of_error produces for every code), is suppressed.
//
// Not [[noreturn]] (PLAN.md §6.1 shows the attribute): the function returns
// normally when rc == 0, and the generated bindings call it unconditionally
// after every C call (PLAN.md §5.4).  A [[noreturn]] function that returns
// is undefined behaviour and a -Wall warning.
void check(int64_t ctx, int32_t rc, const char *fname);

// The symbolic name of an exit code, emitted from the QMCKL_* macros of the
// installed header (single source of truth, PLAN.md §5.7); "QMCKL_UNKNOWN"
// for anything else.
const char *code_name(int32_t rc);

// The error state stored in the context, as
//   "[<CODE_NAME>] <function>: <message>"
// or "[<CODE_NAME>] <function>" when the message is empty or "Unknown
// error"; "Null context" when ctx is not a valid QMCkl context (matching
// the C wording of qmckl_last_error).  Shared by check() and the
// qmckl.last_error() binding.
std::string last_error_message(int64_t ctx);

}  // namespace qmckl

#endif  // QMCKL_PYTHON_CONTEXT_HPP
