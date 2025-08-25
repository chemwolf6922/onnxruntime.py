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

execute_process(
  COMMAND python "${CMAKE_SOURCE_DIR}/get_version.py"
  OUTPUT_VARIABLE PYORT_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYORT_VERSION_RESULT
)

if(NOT PYORT_VERSION_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to get version: ${PYORT_VERSION_RESULT}")
endif()

set(PYBIND11_SUBMODULE_DIR "${CMAKE_SOURCE_DIR}/extern/pybind11")

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

add_library(_pyort MODULE
  src/cpp/Bindings.cpp
  src/cpp/Pyort.cpp
)

if(NOT PYTHON_INCLUDE_DIR OR NOT PYTHON_LIBRARY)
  message(STATUS "Python include directory or library not set, using the system python.")
  find_package(Python3 COMPONENTS Development REQUIRED)
  set(PYTHON_INCLUDE_DIR "${Python3_INCLUDE_DIRS}")
  set(PYTHON_LIBRARY "${Python3_LIBRARIES}")
  set(BUILD_FOR_HOST TRUE)
endif()

message(STATUS "Using Python include directory: ${PYTHON_INCLUDE_DIR}")
message(STATUS "Using Python library: ${PYTHON_LIBRARY}")

target_include_directories(_pyort PRIVATE
  "${PYTHON_INCLUDE_DIR}"
  "${PYBIND11_SUBMODULE_DIR}/include"
  "${ONNXRUNTIME_ROOT}/include"
)
target_link_libraries(_pyort PRIVATE 
  "${PYTHON_LIBRARY}"
  "${ONNXRUNTIME_LIB_DIR}/${ONNXRUNTIME_LIB_NAME}.lib"
)

target_compile_definitions(_pyort PRIVATE PYORT_VERSION="${PYORT_VERSION}")
set_target_properties(_pyort PROPERTIES
  PREFIX ""
  SUFFIX ".pyd"
  CXX_STANDARD 20
)

if(BUILD_FOR_HOST)
  add_custom_command(TARGET _pyort POST_BUILD
    COMMAND python -m pybind11_stubgen _pyort --output-dir "$<TARGET_FILE_DIR:_pyort>"
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:_pyort>"
    COMMENT "Generate .pyi file. This only works when building for the host."
  )
endif()

add_custom_command(TARGET _pyort POST_BUILD
  COMMAND python "${CMAKE_SOURCE_DIR}/pack_wheel.py" -b "$<CONFIG>" -p "${PYTHON_LIBRARY}" -a "${TARGET_ARCH}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Packing wheel after build"
)

