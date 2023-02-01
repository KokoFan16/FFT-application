cmake_minimum_required (VERSION 3.9)

file(GLOB_RECURSE CFILES "${PROJECT_SOURCE_DIR}/src/core/*.cpp")

aux_source_directory(. MPR_CORE_SOURCES)

add_library(HVIS_src ${MPR_CORE_SOURCES})

target_link_libraries(HVIS_src ${MPI_C_LIBRARIES})