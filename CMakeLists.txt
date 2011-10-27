cmake_minimum_required(VERSION 2.8)

set(lib_name autocalib)
project(${lib_name})

set(EIGEN3_DIR CACHE PATH "Path to Eigen3 sources")

find_package(OpenCV REQUIRED)

include_directories("." ${EIGEN3_DIR})

add_subdirectory(core)
add_subdirectory(core_tests)
add_subdirectory(core_evaluation)
add_subdirectory(evaluation)
