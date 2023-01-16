cmake_minimum_required (VERSION 3.5)

project (FFTWAPP)

message("Configuring MPR with MPI support")
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

SET(FFTW3_DIR "${CMAKE_SOURCE_DIR}/external/fftw-3.3.10")
SET(FFTW3_INCLUDE_DIR  "${FFTW3_DIR}/include")
include_directories(SYSTEM ${FFTW3_INCLUDE_DIR})

include_directories("${PROJECT_SOURCE_DIR}/examples")

add_subdirectory(examples)
