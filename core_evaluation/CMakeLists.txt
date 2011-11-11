cmake_minimum_required(VERSION 2.8)

set(CMAKE_INCLUDE_CURRENT_DIR on)

include_directories(${OpenCV_INCLUDE_DIR} ${GTEST_INCLUDE_DIRS} ${CMAKE_CURRENT_DIRECTORY})

file(GLOB sources "src/*.cpp")
foreach (filename ${sources})
    get_filename_component(name ${filename} NAME_WE)
    set(target "${name}")
    add_executable(${target} ${filename})
    add_dependencies(${target} "${lib_name}_core" "${lib_name}_evaluation")
    target_link_libraries(${target} ${OpenCV_LIBS} "${lib_name}_core" "${lib_name}_evaluation")
endforeach (filename)



