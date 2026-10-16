# rkllm_api_demo CMakeLists - adapted for the Debian @PACKAGE@ package.
# Upstream original: examples/rkllm_api_demo/deploy/CMakeLists.txt (rknn-llm v@UPSTREAM_VERSION@)
# Change 1: the upstream file resolved the runtime through the relative path
#           ../../../rkllm-runtime/${CMAKE_SYSTEM_NAME}/..., which does not exist
#           once installed under /usr. It now uses the package config that is
#           installed at @LIBDIR_ABS@/cmake/rkllm.
# Change 2: dropped the Android branch and the install() rules that copied
#           librkllmrt.so next to the binary (the library now lives in the
#           system library path, so nothing has to be copied).

# Change 3: pinned the install prefix into the build tree and used a relative
#           DESTINATION. The earlier form `DESTINATION ${CMAKE_INSTALL_PREFIX}/bin`
#           is absolute, and CMake honours absolute destinations verbatim, so a
#           user running `make install` after configuring with the default prefix
#           would drop llm_demo into /usr/local/bin - a file dpkg does not own and
#           therefore cannot remove on purge. Examples live read-only under
#           /usr/share, so nothing should ever be installed out of the build tree.

cmake_minimum_required(VERSION 3.10)
project(rkllm_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Falls back to plain -lrkllmrt if the CMake package config is unavailable.
find_package(rkllm QUIET)
if(NOT TARGET rkllm::rkllmrt)
    find_library(RKLLM_RT_LIB rkllmrt REQUIRED)
    add_library(rkllm::rkllmrt SHARED IMPORTED)
    set_target_properties(rkllm::rkllmrt PROPERTIES IMPORTED_LOCATION "${RKLLM_RT_LIB}")
endif()

add_executable(llm_demo src/llm_demo.cpp)
target_link_libraries(llm_demo PRIVATE rkllm::rkllmrt)

set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "" FORCE)
install(TARGETS llm_demo RUNTIME DESTINATION bin)
