cmake_minimum_required(VERSION 2.8)

set(target "${lib_name}_tests")
set(CMAKE_INCLUDE_CURRENT_DIR on)

find_package(GTest REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR} ${GTEST_INCLUDE_DIRS} ${CMAKE_CURRENT_DIRECTORY})

file(GLOB sources "src/*.h" "src/*.cpp")
add_executable(${target} ${includes} ${sources})

add_dependencies(${target} "${lib_name}_core" "${lib_name}_evaluation")
target_link_libraries(${target} ${OpenCV_LIBS} ${GTEST_BOTH_LIBRARIES} "${lib_name}_core" "${lib_name}_evaluation")

