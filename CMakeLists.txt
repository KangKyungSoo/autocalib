cmake_minimum_required(VERSION 2.8)

set(lib_name autocalib)
project(${lib_name})

find_package(OpenCV REQUIRED)

add_subdirectory(core)
