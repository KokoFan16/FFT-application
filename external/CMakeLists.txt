
#add_subdirectory(profiler)
SET(VIS_DIR "${CMAKE_SOURCE_DIR}/external/profiler")

INCLUDE_DIRECTORIES("${VIS_DIR}/src")
FILE(GLOB VIS_SRC ${VIS_DIR}/src/*.hpp ${VIS_DIR}/src/*.cpp)

ADD_LIBRARY(profiler "${VIS_SRC}")

INSTALL(DIRECTORY "${VIS_DIR}/src" DESTINATION ./)
INSTALL(TARGETS profiler DESTINATION lib)
