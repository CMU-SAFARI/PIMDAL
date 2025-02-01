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
  kernel_join.c
  ${PROJECT_LIBRARY_DIR}/sort/sort_keyval.c
  ${PROJECT_LIBRARY_DIR}/join/sort_merge.c
)

add_executable(kernel_bjoin ${DPU_SOURCES})
target_compile_definitions(kernel_bjoin PUBLIC NR_TASKLETS=${NR_TASKLETS} INNER_SIZE=${INNER_SIZE} OUTER_SIZE=${OUTER_SIZE} STACK_SIZE_DEFAULT=2600 PERF=${PERF})
target_link_options(kernel_bjoin PUBLIC -DNR_TASKLETS=${NR_TASKLETS} -DINNER_SIZE=${INNER_SIZE} -DOUTER_SIZE=${OUTER_SIZE} -DSTACK_SIZE_DEFAULT=2600 -DPERF=${PERF})