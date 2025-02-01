cmake_minimum_required(VERSION 3.0.0)
project(upmem-db-main-join VERSION 0.1.0)

include_directories(
    PUBLIC "${PROJECT_LIBRARY_DIR}/join"
    PUBLIC "${CMAKE_CURRENT_LIST_DIR}/shared"
    PUBLIC "${PROJECT_LIBRARY_DIR}/sort"
)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/join/bcast )

add_subdirectory(dpu)
add_subdirectory(host)