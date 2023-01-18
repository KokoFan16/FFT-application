cmake_minimum_required (VERSION 3.5)

file(GLOB_RECURSE CFILES "${PROJECT_SOURCE_DIR}/examples/*.c")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE C COMPILE_FLAGS "-std=c99 -O2")
include_directories(SYSTEM ${FFTW3_INCLUDE_DIR})
link_directories("${FFTW3_DIR}/lib/")

add_executable(fftw_mpi_1d ./fftw_mpi.c)
#target_link_directories(fftw_mpi_1d PUBLIC "${FFTW3_DIR}/lib/")
target_link_libraries(fftw_mpi_1d ${MPI_C_LIBRARIES} fftw3_mpi fftw3)

#add_executable(fftw_mpi_2d ./fftw_2d.c)
#target_link_directories(fftw_mpi_2d PUBLIC "${FFTW3_DIR}/lib/")
#target_link_libraries(fftw_mpi_2d ${MPI_C_LIBRARIES} fftw3_mpi fftw3)
