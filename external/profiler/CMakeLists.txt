cmake_minimum_required (VERSION 3.9)

project (HVIS)

add_compile_options(--std=c++17 -Wno-strict-aliasing -Werror=class-varargs -fpermissive)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++17 -Werror=class-varargs -fpermissive")

message("Configuring MPR with MPI support")
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/examples")

add_subdirectory(src)
add_subdirectory(examples)
