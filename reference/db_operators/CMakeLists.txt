cmake_minimum_required(VERSION 3.0.0)
project(db_operators)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

# Set the relation sizes for the join micro benchmarks
set (INNER_SIZE 256000000)
set (OUTER_SIZE 512000000)

# Set the buffer size for the other micro benchmarks
set (BUFFER_SIZE 500000000)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -Wno-unused-result -fopenmp")

if(CMAKE_BUILD_TYPE STREQUAL "Profile")
    set(CMAKE_CXX_FLAGS "-g -O3 -xHost -Wno-unused-result -qopenmp")
endif()

FIND_PACKAGE(Arrow REQUIRED)
FIND_PACKAGE(ArrowAcero REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)

add_subdirectory(abseil-cpp)

add_executable(select select.cpp)
target_include_directories(select PUBLIC PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_compile_definitions(select PUBLIC BUFFER_SIZE=${BUFFER_SIZE})
target_link_options(select PUBLIC -DBUFFER_SIZE=${BUFFER_SIZE})
target_link_libraries(select PUBLIC PRIVATE Arrow::arrow_shared ArrowAcero::arrow_acero_shared OpenMP::OpenMP_CXX)

add_executable(aggregate aggregate.cpp)
target_include_directories(aggregate PUBLIC PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_compile_definitions(aggregate PUBLIC BUFFER_SIZE=${BUFFER_SIZE})
target_link_options(aggregate PUBLIC -DBUFFER_SIZE=${BUFFER_SIZE})
target_link_libraries(aggregate PUBLIC PRIVATE Arrow::arrow_shared ArrowAcero::arrow_acero_shared OpenMP::OpenMP_CXX absl::flat_hash_map)

add_executable(sort sort.cpp)
target_include_directories(sort PUBLIC PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_compile_definitions(sort PUBLIC BUFFER_SIZE=${BUFFER_SIZE})
target_link_options(sort PUBLIC -DBUFFER_SIZE=${BUFFER_SIZE})
target_link_libraries(sort PUBLIC PRIVATE Arrow::arrow_shared ArrowAcero::arrow_acero_shared OpenMP::OpenMP_CXX)

add_executable(sm_join sort_merge_join.cpp)
target_include_directories(sm_join PUBLIC PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_compile_definitions(sm_join PUBLIC BUFFER_SIZE=${BUFFER_SIZE})
target_link_options(sm_join PUBLIC -DBUFFER_SIZE=${BUFFER_SIZE})
target_link_libraries(sm_join PUBLIC PRIVATE Arrow::arrow_shared ArrowAcero::arrow_acero_shared OpenMP::OpenMP_CXX)

add_executable(hash_join hash_join.cpp)
target_include_directories(hash_join PUBLIC PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_compile_definitions(hash_join PUBLIC BUFFER_SIZE=${BUFFER_SIZE})
target_link_options(hash_join PUBLIC -DBUFFER_SIZE=${BUFFER_SIZE})
target_link_libraries(hash_join PUBLIC PRIVATE Arrow::arrow_shared ArrowAcero::arrow_acero_shared OpenMP::OpenMP_CXX absl::flat_hash_set)