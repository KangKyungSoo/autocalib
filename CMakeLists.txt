cmake_minimum_required(VERSION 2.8)

set(lib_name autocalib)
project(${lib_name})

include(FindEigen.cmake)
find_package(OpenCV REQUIRED)
set(GLFW_INCLUDE_DIR CACHE PATH "")
set(GLFW_LIBRARIES CACHE PATH "")
set(GLUT_INCLUDE_DIR CACHE PATH "")
set(GLUT_LIBRARIES CACHE PATH "")

include_directories("." ${EIGEN_INCLUDE_DIRS} ${GLFW_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})

#add_subdirectory(3rdparty/levmar-2.6)
add_subdirectory(core)
add_subdirectory(tests)
add_subdirectory(samples)
add_subdirectory(evaluation)
add_subdirectory(jps2jpg)
add_subdirectory(takeshots)
add_subdirectory(undistort)
add_subdirectory(crossvalidate)
