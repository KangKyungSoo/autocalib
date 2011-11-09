cmake_minimum_required(VERSION 2.8)

set(lib_name autocalib)
project(${lib_name})

include(FindEigen.cmake)
find_package(OpenCV REQUIRED)

set(GLFW_INCLUDE_DIR CACHE PATH "")
set(GLFW_LIBRARIES CACHE PATH "")

include_directories("." ${EIGEN_INCLUDE_DIRS} ${GLFW_INCLUDE_DIR})

add_subdirectory(core)
add_subdirectory(core_tests)
add_subdirectory(core_evaluation)
add_subdirectory(evaluation)
