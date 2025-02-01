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
  kernel_select.c
  ${PROJECT_LIBRARY_DIR}/select/sel.c
)

add_executable(kernel_select ${DPU_SOURCES})
target_compile_definitions(kernel_select PUBLIC NR_TASKLETS=${NR_TASKLETS} BUFFER_SIZE=${BUFFER_SIZE} PERF=${PERF})
target_link_options(kernel_select PUBLIC -DNR_TASKLETS=${NR_TASKLETS} -DBUFFER_SIZE=${BUFFER_SIZE} -DPERF=${PERF})