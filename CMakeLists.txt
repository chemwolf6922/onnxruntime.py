cmake_minimum_required(VERSION 3.20)
project(onnxruntime.py LANGUAGES CXX)

# This does not work for Windows...
# Please build with --config Release instead
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.8 REQUIRED Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/get_version.py"
  OUTPUT_VARIABLE ORTPY_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE ORTPY_VERSION_RESULT
)

if(NOT ORTPY_VERSION_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to get version: ${ORTPY_VERSION_RESULT}")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nanobind)

set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")

find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  PATHS "${ONNXRUNTIME_LIB_DIR}"
  NO_DEFAULT_PATH
  REQUIRED
)

nanobind_add_module(_ortpy MODULE
  src/cpp/Bindings.cpp
  src/cpp/Ortpy.cpp
)

target_include_directories(_ortpy PRIVATE
  "${ONNXRUNTIME_INCLUDE_DIR}"
)
target_link_libraries(_ortpy PRIVATE
  "${ONNXRUNTIME_LIBRARY}"
)

target_compile_definitions(_ortpy PRIVATE ORTPY_VERSION="${ORTPY_VERSION}")

nanobind_add_stub(
  _ortpy_stub
  MODULE _ortpy
  OUTPUT _ortpy.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:_ortpy>
  DEPENDS _ortpy
)

add_custom_command(TARGET _ortpy_stub POST_BUILD
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/pack_wheel.py" -b "$<CONFIG>"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Packing wheel after build"
)

