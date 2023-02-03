cmake_minimum_required (VERSION 3.5)

project (FFTWAPP)

message("Configuring MPR with MPI support")
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})


set(Caliper_INSTALL_DIR "${CMAKE_SOURCE_DIR}/external/caliper/build/")
include_directories("${Caliper_INSTALL_DIR}/usr/local/include/")
link_directories("${Caliper_INSTALL_DIR}/usr/local/lib")
set(caliper_INCLUDE_DIR "${Caliper_INSTALL_DIR}/usr/local/include")

message("Configuring caliper")
find_package(caliper EXACT REQUIRED PATHS ${Caliper_INSTALL_DIR})
include_directories(SYSTEM ${caliper_INCLUDE_DIR})

SET(FFTW3_DIR "${CMAKE_SOURCE_DIR}/external/fftw-3.3.10")
SET(FFTW3_INCLUDE_DIR  "${FFTW3_DIR}/include")
include_directories(SYSTEM ${FFTW3_INCLUDE_DIR})

SET(VIS_DIR "${CMAKE_SOURCE_DIR}/external/profiler")
SET(VIS_INCLUDE_DIR  "${VIS_DIR}/src")
include_directories(SYSTEM ${VIS_INCLUDE_DIR})

include_directories("${PROJECT_ScOURCE_DIR}/examples")
include_directories("${PROJECT_SOURCE_DIR}/external")

add_subdirectory(examples)
