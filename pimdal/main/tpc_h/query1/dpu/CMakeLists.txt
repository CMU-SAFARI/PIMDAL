cmake_minimum_required(VERSION 3.0.0)
project(upmem-db-main-dpu VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include("${UPMEM_HOME}/share/upmem/cmake/dpu.cmake")

set(CMAKE_C_FLAGS_DEBUG "-Wall -Wextra -g -Og")
set(CMAKE_C_FLAGS_RELEASE "-Wall -Wextra -g0 -O2")

if (NOT DEFINED NR_TASKLETS)
  set(NR_TASKLETS 16)
endif()

set (DPU_SOURCES_1
  kernel_q1_1.c
  ${PROJECT_LIBRARY_DIR}/select/sel.c
)

set (DPU_SOURCES_2
  kernel_q1_2.c
  ${PROJECT_LIBRARY_DIR}/aggregate/aggregate_hash.c
)

add_executable(kernel_q1_1 ${DPU_SOURCES_1})
target_compile_definitions(kernel_q1_1 PUBLIC NR_TASKLETS=${NR_TASKLETS} NR_DPU=${NR_DPU} PTR_TYPE=key_ptr32)
target_link_options(kernel_q1_1 PUBLIC -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPU=${NR_DPU} -DPTR_TYPE=key_ptr32)

add_executable(kernel_q1_2 ${DPU_SOURCES_2})
target_compile_definitions(kernel_q1_2 PUBLIC NR_TASKLETS=${NR_TASKLETS} NR_DPU=${NR_DPU} PTR_TYPE=key_ptrout)
target_link_options(kernel_q1_2 PUBLIC -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPU=${NR_DPU} -DPTR_TYPE=key_ptrout)