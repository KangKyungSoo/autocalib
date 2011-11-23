cmake_minimum_required(VERSION 2.8)

set(target "${lib_name}_stereo_stream_splitter")
set(CMAKE_INCLUDE_CURRENT_DIR on)

include_directories(${OpenCV_INCLUDE_DIR} ${CMAKE_CURRENT_DIRECTORY})

file(GLOB sources "include/*.h" "src/*.h" "src/*.cpp")
add_library(${target} ${includes} ${sources})

target_link_libraries(${target} ${OpenCV_LIBS})

