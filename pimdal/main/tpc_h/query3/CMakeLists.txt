cmake_minimum_required(VERSION 3.0.0)
project(upmem-db-main-query3 VERSION 0.1.0)

include_directories(
    PUBLIC "${PROJECT_LIBRARY_DIR}/select"
    PUBLIC "${PROJECT_LIBRARY_DIR}/join"
    PUBLIC "${PROJECT_LIBRARY_DIR}/general"
    PUBLIC "${CMAKE_CURRENT_LIST_DIR}/shared"
    PUBLIC "${PROJECT_LIBRARY_DIR}/aggregate"
    PUBLIC "${PROJECT_LIBRARY_DIR}/sort"
)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/query3 )

add_subdirectory(dpu)
add_subdirectory(host)