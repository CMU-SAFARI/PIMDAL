cmake_minimum_required(VERSION 3.0.0)
project(upmem-db-main-dpu VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include("${UPMEM_HOME}/share/upmem/cmake/dpu.cmake")

set(CMAKE_C_FLAGS_DEBUG "-Wall -Wextra -g -Og")
set(CMAKE_C_FLAGS_RELEASE "-Wall -Wextra -g0 -O2")

if (NOT DEFINED NR_TASKLETS)
  set(NR_TASKLETS 16)
endif()

set (DPU_SOURCES
  kernel_aggregate.c
  ${PROJECT_LIBRARY_DIR}/aggregate/aggregate_hash.c
)

add_executable(kernel_haggregate ${DPU_SOURCES})
target_compile_definitions(kernel_haggregate PUBLIC NR_TASKLETS=${NR_TASKLETS} PERF=${PERF})
target_link_options(kernel_haggregate PUBLIC -DNR_TASKLETS=${NR_TASKLETS} -DPERF=${PERF})