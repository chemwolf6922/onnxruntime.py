cmake_minimum_required(VERSION 3.20)
project(onnxruntime-py LANGUAGES CXX)

# Options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Force usage of vendored pybind11 submodule under extern/pybind11
set(PYBIND11_SUBMODULE_DIR "${CMAKE_SOURCE_DIR}/extern/pybind11")
if(EXISTS "${PYBIND11_SUBMODULE_DIR}/CMakeLists.txt")
  message(STATUS "Using vendored pybind11 from ${PYBIND11_SUBMODULE_DIR}")
  add_subdirectory("${PYBIND11_SUBMODULE_DIR}" EXCLUDE_FROM_ALL)
else()
  message(FATAL_ERROR "pybind11 submodule not found at ${PYBIND11_SUBMODULE_DIR}. Initialize with: git submodule update --init --recursive")
endif()

# ONNX Runtime root (extracted by the PowerShell script)
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime" CACHE PATH "Path to ONNX Runtime install root")

# Validate expected layout
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_c_api.h")
  message(FATAL_ERROR "Couldn't find ONNX Runtime headers at ${ONNXRUNTIME_ROOT}/include. Run GetOnnxruntime.ps1 first.")
endif()

# Library dir and names on Windows CPU x64
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")
set(ONNXRUNTIME_LIB_NAME "onnxruntime")

# Create extension
pybind11_add_module(_pyort src/pyort/bindings.cpp)

# Include and link
target_include_directories(_pyort PRIVATE "${ONNXRUNTIME_ROOT}/include")
if(EXISTS "${ONNXRUNTIME_LIB_DIR}/${ONNXRUNTIME_LIB_NAME}.lib")
  target_link_libraries(_pyort PRIVATE "${ONNXRUNTIME_LIB_DIR}/${ONNXRUNTIME_LIB_NAME}.lib")
else()
  message(FATAL_ERROR "Couldn't find ${ONNXRUNTIME_LIB_DIR}/${ONNXRUNTIME_LIB_NAME}.lib. Make sure CPU x64 package is downloaded.")
endif()

# Define version macro
set_target_properties(_pyort PROPERTIES
  CXX_STANDARD 20
)

# On Windows, ensure the DLLs next to the extension at runtime (copy them)
add_custom_command(TARGET _pyort POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${ONNXRUNTIME_LIB_DIR}/${ONNXRUNTIME_LIB_NAME}.dll"
          $<TARGET_FILE_DIR:_pyort>
)
