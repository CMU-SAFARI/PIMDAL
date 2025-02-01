cmake_minimum_required(VERSION 3.0.0)
project(pimdal-main VERSION 0.1.0)

set(PROJECT_LIBRARY_DIR "${CMAKE_SOURCE_DIR}")

include_directories(
    PUBLIC "${PROJECT_LIBRARY_DIR}/support"
)

# Set the relation sizes for the join micro benchmarks
set (INNER_SIZE 1024000)
set (OUTER_SIZE 1024000)

# Set the buffer size for the other micro benchmarks
set (BUFFER_SIZE 4000000)

# Activate performance measurements on the DPUs
# 0: off, 1: count cycles, 2: count instructions
set (PERF 0)

# Compile micro benchmarks
add_subdirectory(select)
add_subdirectory(aggregate/sort)
add_subdirectory(aggregate/hash)
add_subdirectory(sort)
add_subdirectory(join/sort)
add_subdirectory(join/hash)
add_subdirectory(join/broadcast)

# Compile TPC-H queries
add_subdirectory(tpc_h/query1)
add_subdirectory(tpc_h/query3)
add_subdirectory(tpc_h/query4)
add_subdirectory(tpc_h/query5)
add_subdirectory(tpc_h/query6)
