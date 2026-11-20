// csrc/context.cpp
//
// Implementation of qmckl::Context, qmckl::check() and
// qmckl::last_error_message() (PLAN.md §6.1).

#include "context.hpp"

namespace qmckl {

// The installed qmckl.h (QMCkl 1.1.0) does not export the buffer lengths
// used by qmckl_get_error(); they are defined in QMCkl's private header and
// measure 256 and 1024 (PLAN.md §2).  qmckl_get_error() memcpy()s up to
// those lengths and never writes a NUL terminator, so the buffers below are
// zero-initialised with one byte of slack.  The #ifndef guards keep working
// if a future header starts exporting the macros.
#ifndef QMCKL_MAX_FUN_LEN
#define QMCKL_MAX_FUN_LEN 256
#endif
#ifndef QMCKL_MAX_MSG_LEN
#define QMCKL_MAX_MSG_LEN 1024
#endif

namespace {

// The message qmckl_failwith(ctx, code, function, NULL) stores verbatim in a
// QMCkl 1.1.0 release-tarball build: qmckl_string_of_error()'s switch is
// empty, so the stored default message is "Unknown error" for every code.
// Reporting it adds nothing but confusion (PLAN.md §2, §6.1).
const char *const kUnknownMessage = "Unknown error";

struct StoredError {
    bool valid = false;       // the context was valid at the time of the read
    int32_t code = 0;
    std::string function;
    std::string message;      // empty when suppressed
};

// One read of the error state stored in the context.  qmckl_get_error()
// asserts on an invalid context, so guard with qmckl_context_check() first.
StoredError read_stored_error(int64_t ctx) {
    StoredError e;
    if (::qmckl_context_check(ctx) == QMCKL_NULL_CONTEXT)
        return e;
    char fun[QMCKL_MAX_FUN_LEN + 1]{};
    char msg[QMCKL_MAX_MSG_LEN + 1]{};
    ::qmckl_get_error(ctx, &e.code, fun, msg);
    e.function = fun;
    e.message = msg;
    if (e.message.empty() || e.message == kUnknownMessage)
        e.message.clear();
    e.valid = true;
    return e;
}

// "[<CODE>] <function>: <message>", with " <function>" dropped for an empty
// function name (a fresh context stores none) and ": <message>" dropped for
// a suppressed message.
std::string format_error(const char *code, const std::string &function,
                         const std::string &message) {
    std::string out = std::string("[") + code + "]";
    if (!function.empty())
        out += " ";
    out += function;
    if (!message.empty())
        out += ": " + message;
    return out;
}

}  // namespace

Context::Context() : h(QMCKL_NULL_CONTEXT) {
    h = ::qmckl_context_create();
    if (h == QMCKL_NULL_CONTEXT)
        throw QMCKLStatusError("qmckl_context_create failed");
}

Context::Context(const Context &other) : h(QMCKL_NULL_CONTEXT) {
    h = ::qmckl_context_copy(other.h);
    if (h == QMCKL_NULL_CONTEXT)
        throw QMCKLStatusError("qmckl_context_copy failed");
}

Context::~Context() {
    if (h != QMCKL_NULL_CONTEXT) {
        (void) ::qmckl_context_destroy(h);  // errors swallowed (PLAN.md §6.1)
        h = QMCKL_NULL_CONTEXT;
    }
}

void check(int64_t ctx, int32_t rc, const char *fname) {
    if (rc == 0)
        return;
    std::string message;
    StoredError stored = read_stored_error(ctx);
    // The stored message is only trustworthy when the context is valid and
    // the stored code matches the code this call returned (PLAN.md §6.1).
    if (stored.valid && stored.code == rc)
        message = stored.message;
    throw QMCKLStatusError(
        format_error(code_name(rc), fname, message));
}

std::string last_error_message(int64_t ctx) {
    StoredError e = read_stored_error(ctx);
    if (!e.valid)
        return "Null context";
    return format_error(code_name(e.code), e.function, e.message);
}

}  // namespace qmckl
