cmake_minimum_required(VERSION 3.0.0)
project(upmem-db-main-groupby VERSION 0.1.0)

include_directories(
    PUBLIC "${PROJECT_LIBRARY_DIR}/aggregate"
    PUBLIC "${CMAKE_CURRENT_LIST_DIR}/shared"
    PUBLIC "${PROJECT_LIBRARY_DIR}/hash"
)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/aggregate/hash )

if (NOT DEFINED NR_TASKLETS)
  set(NR_TASKLETS 16)
endif()

add_subdirectory(dpu)
add_subdirectory(host)