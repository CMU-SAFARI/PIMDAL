cmake_minimum_required(VERSION 3.0.0)
project(upmem-db-main-dpu VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include("${UPMEM_HOME}/share/upmem/cmake/dpu.cmake")

set(CMAKE_C_FLAGS_DEBUG "-Wall -Wextra -g -Og")
set(CMAKE_C_FLAGS_RELEASE "-Wall -Wextra -g0 -O2")

if (NOT DEFINED NR_TASKLETS)
  set(NR_TASKLETS 12)
endif()

set (DPU_SOURCES_1
  kernel_q4_1.c
  ${PROJECT_LIBRARY_DIR}/select/sel.c
  ${PROJECT_LIBRARY_DIR}/join/hash_join.c
)

set (DPU_SOURCES_2
  kernel_q4_2.c
  ${PROJECT_LIBRARY_DIR}/select/sel.c
  ${PROJECT_LIBRARY_DIR}/join/hash_join.c
)

set (DPU_SOURCES_3
  kernel_q4_3.c
  ${PROJECT_LIBRARY_DIR}/join/hash_join.c
  ${PROJECT_LIBRARY_DIR}/aggregate/aggregate_hash.c
)

set (DPU_SOURCES_4
  kernel_q4_4.c
  ${PROJECT_LIBRARY_DIR}/join/hash_join.c
  ${PROJECT_LIBRARY_DIR}/aggregate/aggregate_hash.c
)

add_executable(kernel_q4_1 ${DPU_SOURCES_1})
target_compile_definitions(kernel_q4_1 PUBLIC NR_TASKLETS=${NR_TASKLETS} NR_DPU=${NR_DPU} TYPE=key_ptr32 VAL_SIZE=16)
target_link_options(kernel_q4_1 PUBLIC -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPU=${NR_DPU} -DTYPE=key_ptr32 -DVAL_SIZE=16)

add_executable(kernel_q4_2 ${DPU_SOURCES_2})
target_compile_definitions(kernel_q4_2 PUBLIC NR_TASKLETS=${NR_TASKLETS} NR_DPU=${NR_DPU} TYPE=keyval_ptr32)
target_link_options(kernel_q4_2 PUBLIC -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPU=${NR_DPU} -DTYPE=keyval_ptr32)

add_executable(kernel_q4_3 ${DPU_SOURCES_3})
target_compile_definitions(kernel_q4_3 PUBLIC NR_TASKLETS=${NR_TASKLETS} NR_DPU=${NR_DPU} TYPE=key_ptr32 UNIQUE)
target_link_options(kernel_q4_3 PUBLIC -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPU=${NR_DPU} -DTYPE=key_ptr32 -DUNIQUE)

add_executable(kernel_q4_4 ${DPU_SOURCES_4})
target_compile_definitions(kernel_q4_4 PUBLIC NR_TASKLETS=${NR_TASKLETS} NR_DPU=${NR_DPU} TYPE=key_ptrtext)
target_link_options(kernel_q4_4 PUBLIC -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPU=${NR_DPU} -DTYPE=key_ptrtext)