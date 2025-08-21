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
  COMMAND python "${CMAKE_SOURCE_DIR}/get_version.py"
  OUTPUT_VARIABLE PYORT_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(PYBIND11_SUBMODULE_DIR "${CMAKE_SOURCE_DIR}/extern/pybind11")

set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime" CACHE PATH "Path to ONNX Runtime install root")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")
set(ONNXRUNTIME_LIB_NAME "onnxruntime")

add_library(_pyort MODULE
  src/cpp/Bindings.cpp
  src/cpp/Pyort.cpp
)

if(NOT PYTHON_INCLUDE_DIR OR NOT PYTHON_LIBRARY)
  message(STATUS "Python include directory or library not set, using the system python.")
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  set(PYTHON_INCLUDE_DIR "${Python3_INCLUDE_DIRS}")
  set(PYTHON_LIBRARY "${Python3_LIBRARIES}")
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

target_compile_definitions(_pyort PRIVATE PYBIND11_NUMPY_DTYPE=1)
target_compile_definitions(_pyort PRIVATE PYORT_VERSION="${PYORT_VERSION}")
set_target_properties(_pyort PROPERTIES
  PREFIX ""
  SUFFIX ".pyd"
  CXX_STANDARD 20
)

add_custom_command(TARGET _pyort POST_BUILD
  COMMAND python "${CMAKE_SOURCE_DIR}/pack_wheel.py" -b "$<CONFIG>" -p "${PYTHON_LIBRARY}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Packing wheel after build"
)

