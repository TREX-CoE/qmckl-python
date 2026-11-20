cmake_minimum_required(VERSION 3.15...3.31)

project(qmckl_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# 1. Find the QMCkl we will link against.
#
# The bindings are generated from *this* header, so the header and the library
# can never disagree.  A QMCkl that is too old (or absent) must fail loudly and
# tell the user what to do.
# ---------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)

pkg_check_modules(QMCKL QUIET IMPORTED_TARGET "qmckl>=1.1")

if(NOT QMCKL_FOUND)
  execute_process(
    COMMAND "${PKG_CONFIG_EXECUTABLE}" --modversion qmckl
    OUTPUT_VARIABLE _qmckl_version_seen
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_qmckl_version_seen)
    set(_why "found qmckl ${_qmckl_version_seen}, but qmckl-python requires >= 1.1")
  else()
    set(_why "qmckl.pc was not found")
  endif()
  message(FATAL_ERROR
    "qmckl-python: ${_why}.\n"
    "Install QMCkl 1.x (>= 1.1), then make it visible to pkg-config:\n"
    "    ./configure --prefix=<prefix> --without-openmp && make -j && make install\n"
    "    export PKG_CONFIG_PATH=<prefix>/lib/pkgconfig:$PKG_CONFIG_PATH\n"
    "pkg-config executable : ${PKG_CONFIG_EXECUTABLE}\n"
    "PKG_CONFIG_PATH       : '${PKG_CONFIG_PATH}'")
endif()

if(QMCKL_VERSION VERSION_GREATER_EQUAL 2.0)
  message(FATAL_ERROR
    "qmckl-python wraps the QMCkl 1.x API, but qmckl ${QMCKL_VERSION} was found.\n"
    "Point pkg-config at a QMCkl 1.x installation.")
endif()

message(STATUS "qmckl-python: QMCkl ${QMCKL_VERSION} (${QMCKL_PREFIX})")

find_path(QMCKL_INCLUDE_DIR NAMES qmckl.h
          PATHS ${QMCKL_INCLUDE_DIRS}
          NO_DEFAULT_PATH)
if(NOT QMCKL_INCLUDE_DIR)
  message(FATAL_ERROR
    "qmckl-python: qmckl.pc is present but qmckl.h is not in "
    "'${QMCKL_INCLUDE_DIRS}'.\nRebuild QMCkl with 'make install'.")
endif()
set(QMCKL_HEADER "${QMCKL_INCLUDE_DIR}/qmckl.h")

# ---------------------------------------------------------------------------
# 2. Does the linked library actually provide TREXIO I/O?
#
# qmckl.h declares qmckl_trexio_read unconditionally, but a library built with
# --without-trexio does not define it.  Bind only what the library exports.
# ---------------------------------------------------------------------------
set(QMCKL_HAS_TREXIO "AUTO" CACHE STRING
    "Whether the linked libqmckl exports qmckl_trexio_read (AUTO/ON/OFF)")
set_property(CACHE QMCKL_HAS_TREXIO PROPERTY STRINGS AUTO ON OFF)

if(QMCKL_HAS_TREXIO STREQUAL "AUTO")
  find_library(QMCKL_LIBRARY_FILE NAMES qmckl
               PATHS ${QMCKL_LIBDIR} ${QMCKL_LIBRARY_DIRS}
               NO_DEFAULT_PATH)
  if(NOT QMCKL_LIBRARY_FILE)
    message(FATAL_ERROR
      "qmckl-python: qmckl.pc is present but libqmckl was not found in "
      "'${QMCKL_LIBDIR}'. Pass -DQMCKL_HAS_TREXIO=ON or =OFF to skip this probe.")
  endif()

  # 'nm -D' reads the dynamic symbol table (Linux shared objects) and
  # 'nm -g' the regular one (macOS dylibs, static archives).  Neither works
  # everywhere, so keep whatever succeeds and search the union.
  set(_qmckl_nm_out "")
  set(_nm_ok FALSE)
  foreach(_nm_flag -D -g)
    execute_process(
      COMMAND "${CMAKE_NM}" ${_nm_flag} "${QMCKL_LIBRARY_FILE}"
      OUTPUT_VARIABLE _nm_out
      ERROR_QUIET
      RESULT_VARIABLE _nm_rc)
    if(_nm_rc EQUAL 0)
      string(APPEND _qmckl_nm_out "${_nm_out}")
      set(_nm_ok TRUE)
    endif()
  endforeach()

  if(NOT _nm_ok)
    message(FATAL_ERROR
      "qmckl-python: could not read the symbol table of '${QMCKL_LIBRARY_FILE}' "
      "with ${CMAKE_NM}.\n"
      "Pass -DQMCKL_HAS_TREXIO=ON or -DQMCKL_HAS_TREXIO=OFF to say whether your "
      "QMCkl was built with TREXIO support.")
  endif()

  string(FIND "${_qmckl_nm_out}" "qmckl_trexio_read" _qmckl_trexio_pos)
  if(_qmckl_trexio_pos EQUAL -1)
    set(QMCKL_HAS_TREXIO OFF)
  else()
    set(QMCKL_HAS_TREXIO ON)
  endif()
endif()

message(STATUS "qmckl-python: TREXIO I/O bound = ${QMCKL_HAS_TREXIO}")

# ---------------------------------------------------------------------------
# 3. Generate the bindings from the installed header, then compile them.
# ---------------------------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

if(QMCKL_HAS_TREXIO)
  set(GEN_TREXIO_FLAG --trexio)
else()
  set(GEN_TREXIO_FLAG --no-trexio)
endif()

set(GEN_CPP "${CMAKE_CURRENT_BINARY_DIR}/generated_bindings.cpp")
set(GEN_PYI "${CMAKE_CURRENT_BINARY_DIR}/_core.pyi")

add_custom_command(
  OUTPUT "${GEN_CPP}" "${GEN_PYI}"
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/csrc/generate.py"
          --header "${QMCKL_HEADER}"
          --cpp "${GEN_CPP}"
          --pyi "${GEN_PYI}"
          ${GEN_TREXIO_FLAG}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/generate.py" "${QMCKL_HEADER}"
  COMMENT "qmckl-python: generating bindings from ${QMCKL_HEADER}"
  VERBATIM)

nanobind_add_module(_core NB_STATIC
  csrc/module.cpp
  csrc/context.cpp
  "${GEN_CPP}"
)

target_include_directories(_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
target_link_libraries(_core PRIVATE PkgConfig::QMCKL)

# The stub must sit next to the extension in both wheel and editable installs.
install(TARGETS _core LIBRARY DESTINATION qmckl RUNTIME DESTINATION qmckl)
install(FILES "${GEN_PYI}" DESTINATION qmckl)
