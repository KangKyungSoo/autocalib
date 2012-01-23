cmake_minimum_required(VERSION 2.8)

set(target "${lib_name}_core")
set(CMAKE_INCLUDE_CURRENT_DIR on)

include_directories(${OpenCV_INCLUDE_DIR} ${CMAKE_CURRENT_DIRECTORY})

file(GLOB sources "include/*.h" "src/*.h" "src/*.cpp")

set(LEVMAR_LIBS levmar)

link_directories(${CMAKE_BINARY_DIR}) # location of the levmar library
link_directories(${LAPACKBLAS_DIR})

if(HAVE_PLASMA)
  link_directories(${PLASMA_DIR}/lib)
  set(LEVMAR_LIBS ${LEVMAR_LIBS} ${PLASMA_LIB_NAMES})
endif(HAVE_PLASMA)

if(HAVE_LAPACK)
  if(NEED_F2C)
    set(LEVMAR_LIBS ${LEVMAR_LIBS} ${LAPACKBLAS_LIB_NAMES} ${F2C_LIB_NAME})
  else(LEVMAR_LIBS)
    set(LIBS ${LEVMAR_LIBS} ${LAPACKBLAS_LIB_NAMES})
  endif(NEED_F2C)
endif(HAVE_LAPACK)

add_library(${target} ${includes} ${sources})
target_link_libraries(${target} ${OpenCV_LIBS} ${LEVMAR_LIBS})
message(STATUS "${target} will be linked against ${LEVMAR_LIBS}")
add_dependencies(${target} levmar)


