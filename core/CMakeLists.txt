cmake_minimum_required(VERSION 2.8)

set(target "${lib_name}_core")
set(CMAKE_INCLUDE_CURRENT_DIR on)

include_directories(${OpenCV_INCLUDE_DIR} ${CMAKE_CURRENT_DIRECTORY})

file(GLOB sources "include/*.h" "src/*.h" "src/*.cpp")

SET(LEVMAR_LIBS levmar)

LINK_DIRECTORIES(${CMAKE_BINARY_DIR}) # location of the levmar library
LINK_DIRECTORIES(${LAPACKBLAS_DIR})

# libraries the demo depends on
IF(HAVE_PLASMA)
  LINK_DIRECTORIES(${PLASMA_DIR}/lib)
  SET(LEVMAR_LIBS ${LEVMAR_LIBS} ${PLASMA_LIB_NAMES})
ENDIF(HAVE_PLASMA)

IF(HAVE_LAPACK)
  IF(NEED_F2C)
    SET(LEVMAR_LIBS ${LEVMAR_LIBS} ${LAPACKBLAS_LIB_NAMES} ${F2C_LIB_NAME})
  ELSE(LEVMAR_LIBS)
    SET(LIBS ${LEVMAR_LIBS} ${LAPACKBLAS_LIB_NAMES})
  ENDIF(NEED_F2C)
ENDIF(HAVE_LAPACK)

add_library(${target} ${includes} ${sources})
target_link_libraries(${target} ${OpenCV_LIBS} ${LEVMAR_LIBS})
MESSAGE(STATUS "${target} will be linked against ${LEVMAR_LIBS}")

# make sure that the library is built before the demo
ADD_DEPENDENCIES(${target} levmar)


