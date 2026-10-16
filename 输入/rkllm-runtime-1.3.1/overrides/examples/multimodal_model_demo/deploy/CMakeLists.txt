# multimodal_model_demo CMakeLists - adapted for the Debian @PACKAGE@ package.
# Upstream original: examples/multimodal_model_demo/deploy/CMakeLists.txt (rknn-llm v@UPSTREAM_VERSION@)
#
# Adaptations (everything else is untouched upstream logic):
#   1. OpenCV: prefers the system OpenCV. Falls back to the copy bundled by this
#      package under @PKGEXAMPLESDIR@/multimodal_model_demo/deploy/3rdparty/opencv.
#   2. RKNN runtime: taken from @PRIVLIBDIR_ABS@ (a private directory, so it can
#      never overwrite or conflict with a distro/vendor librknnrt.so) plus headers
#      in /usr/include/rknn.
#   3. RKLLM runtime: uses the installed CMake package config instead of the
#      relative path ../../../rkllm-runtime.
#   4. Android/uclibc branches removed - this target is Linux @UPSTREAM_ARCH@.
#   5. install() prefix moved to the build directory so that `make install` never
#      writes into the read-only /usr/share tree.

cmake_minimum_required(VERSION 3.15.1)

project(demo)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--allow-shlib-undefined")

if(CMAKE_VERSION VERSION_GREATER "3.15.0" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_link_options("-Wl,-Bsymbolic")
endif()

# ---------------------------------------------------------------- opencv ----
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(RKLLM_BUNDLED_OPENCV_DIR
        "@PKGEXAMPLESDIR@/multimodal_model_demo/deploy/3rdparty/opencv/opencv-linux-@UPSTREAM_ARCH@/share/OpenCV")
    if(NOT OpenCV_DIR AND EXISTS "${RKLLM_BUNDLED_OPENCV_DIR}")
        set(OpenCV_DIR "${RKLLM_BUNDLED_OPENCV_DIR}")
    endif()
endif()
find_package(OpenCV REQUIRED)

# ----------------------------------------------------------- rknn runtime ----
set(RKNN_RT_LIB "@PRIVLIBDIR_ABS@/librknnrt.so")
set(RKNN_RT_INCLUDES "/usr/include/rknn")
if(NOT EXISTS "${RKNN_RT_LIB}")
    message(FATAL_ERROR "librknnrt.so not found at ${RKNN_RT_LIB}; install the @PACKAGE@ package.")
endif()

# ---------------------------------------------------------- rkllm runtime ----
find_package(rkllm QUIET)
if(NOT TARGET rkllm::rkllmrt)
    find_library(RKLLM_RT_LIB rkllmrt REQUIRED)
    find_path(RKLLM_RT_INCLUDES rkllm.h PATH_SUFFIXES include/@PKGINCLUDE_SUBDIR@ REQUIRED)
    add_library(rkllm::rkllmrt SHARED IMPORTED)
    set_target_properties(rkllm::rkllmrt PROPERTIES IMPORTED_LOCATION "${RKLLM_RT_LIB}")
    include_directories("${RKLLM_RT_INCLUDES}")
endif()

# ------------------------------------------------------------- imgenc --------
include_directories(src ${RKNN_RT_INCLUDES})
add_executable(imgenc src/image_enc.cc src/img_encoder.cpp)
target_link_libraries(imgenc PRIVATE ${OpenCV_LIBS} ${RKNN_RT_LIB})

# ------------------------------------------------------------ audioenc -------
set(POCKETFFT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/pocketfft)
set(MINIAUDIO_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/miniaudio)
include_directories(src ${RKNN_RT_INCLUDES} ${POCKETFFT_PATH}/include ${MINIAUDIO_PATH})
add_executable(audioenc src/audio_enc.cc src/audio_encoder.cpp src/audio_utils.cc)
target_link_libraries(audioenc PRIVATE ${RKNN_RT_LIB} m dl pthread)

# ------------------------------------------------------------------ demo -----
add_executable(${PROJECT_NAME} src/image_enc.cc src/audio_enc.cc src/audio_utils.cc src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${RKNN_RT_LIB} rkllm::rkllmrt)

# Hide symbols to avoid possible symbol conflicts with RKLLM
set_target_properties(${PROJECT_NAME} PROPERTIES
    LINK_FLAGS "-Wl,--no-undefined,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/c_export.map")

# Install into the build tree - the source tree under /usr/share is read-only.
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "" FORCE)
install(TARGETS imgenc audioenc ${PROJECT_NAME} RUNTIME DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
