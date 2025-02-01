cmake_minimum_required(VERSION 3.0.0)
project(upmem-db-main-select VERSION 0.1.0)

include_directories(
    PUBLIC "${PROJECT_LIBRARY_DIR}/select"
    PUBLIC "${PROJECT_LIBRARY_DIR}/general"
    PUBLIC "${CMAKE_CURRENT_LIST_DIR}/shared"
)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/select)

if (NOT DEFINED NR_TASKLETS)
  set(NR_TASKLETS 16)
endif()

add_subdirectory(dpu)
add_subdirectory(host)