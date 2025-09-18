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
  OUTPUT_VARIABLE PYORT_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYORT_VERSION_RESULT
)

if(NOT PYORT_VERSION_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to get version: ${PYORT_VERSION_RESULT}")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nanobind)

# Determine target architecture
if(CMAKE_GENERATOR_PLATFORM)
  set(TARGET_ARCH "${CMAKE_GENERATOR_PLATFORM}")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR)
  set(TARGET_ARCH "${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif()

message(STATUS "Building for architecture: ${TARGET_ARCH}")

# Set ONNXRUNTIME_ROOT based on architecture
if(TARGET_ARCH MATCHES "AMD64|x64")
  set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime")
elseif(TARGET_ARCH MATCHES "ARM64")
  set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime-arm64")
else()
  message(FATAL_ERROR "Unsupported architecture: ${TARGET_ARCH}")
endif()
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")
set(ONNXRUNTIME_LIB_NAME "onnxruntime")

nanobind_add_module(_pyort MODULE
  src/cpp/Bindings.cpp
  src/cpp/Pyort.cpp
)

target_include_directories(_pyort PRIVATE
  "${ONNXRUNTIME_ROOT}/include"
)
target_link_libraries(_pyort PRIVATE 
  "${ONNXRUNTIME_LIB_DIR}/${ONNXRUNTIME_LIB_NAME}.lib"
)

target_compile_definitions(_pyort PRIVATE PYORT_VERSION="${PYORT_VERSION}")

nanobind_add_stub(
  _pyort_stub
  MODULE _pyort
  OUTPUT _pyort.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:_pyort>
  DEPENDS _pyort
)

add_custom_command(TARGET _pyort_stub POST_BUILD
  COMMAND python "${CMAKE_SOURCE_DIR}/pack_wheel.py" -b "$<CONFIG>"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Packing wheel after build"
)

