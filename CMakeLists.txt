cmake_minimum_required(VERSION 3.20)
project(onnxruntime.py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(default_build_type "Release")
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

execute_process(
  COMMAND python "${CMAKE_SOURCE_DIR}/get-version.py"
  OUTPUT_VARIABLE PYORT_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(PYBIND11_SUBMODULE_DIR "${CMAKE_SOURCE_DIR}/extern/pybind11")
add_subdirectory("${PYBIND11_SUBMODULE_DIR}" EXCLUDE_FROM_ALL)

set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime" CACHE PATH "Path to ONNX Runtime install root")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")
set(ONNXRUNTIME_LIB_NAME "onnxruntime")

pybind11_add_module(_pyort
  src/cpp/Bindings.cpp
  src/cpp/Pyort.cpp)

target_include_directories(_pyort PRIVATE "${ONNXRUNTIME_ROOT}/include")
target_compile_definitions(_pyort PRIVATE PYBIND11_NUMPY_DTYPE=1)
target_compile_definitions(_pyort PRIVATE PYORT_VERSION="${PYORT_VERSION}")
target_link_libraries(_pyort PRIVATE "${ONNXRUNTIME_LIB_DIR}/${ONNXRUNTIME_LIB_NAME}.lib")
set_target_properties(_pyort PROPERTIES
  CXX_STANDARD 20
)

add_custom_command(TARGET _pyort POST_BUILD
  COMMAND python "${CMAKE_SOURCE_DIR}/pack_wheel.py" -b "$<CONFIG>"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Packing wheel after build"
)

