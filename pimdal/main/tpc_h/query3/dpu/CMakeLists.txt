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
  kernel_q3_1.c
  ${PROJECT_LIBRARY_DIR}/select/sel.c
  ${PROJECT_LIBRARY_DIR}/join/hash_join.c
)

set (DPU_SOURCES_2
  kernel_q3_2.c
  ${PROJECT_LIBRARY_DIR}/select/sel.c
  ${PROJECT_LIBRARY_DIR}/join/hash_join.c
)

set (DPU_SOURCES_3
  kernel_q3_3.c
  ${PROJECT_LIBRARY_DIR}/join/hash_join.c
)

set (DPU_SOURCES_4
  kernel_q3_4.c
  ${PROJECT_LIBRARY_DIR}/select/sel.c
  ${PROJECT_LIBRARY_DIR}/join/hash_join.c
)

set (DPU_SOURCES_5
  kernel_q3_5.c
  ${PROJECT_LIBRARY_DIR}/join/hash_join.c
  ${PROJECT_LIBRARY_DIR}/aggregate/aggregate_hash.c
  ${PROJECT_LIBRARY_DIR}/sort/sort_keyval.c
)

add_executable(kernel_q3_1 ${DPU_SOURCES_1})
target_compile_definitions(kernel_q3_1 PUBLIC NR_TASKLETS=${NR_TASKLETS} NR_DPU=${NR_DPU} PTR_TYPE=key_ptrtext)
target_link_options(kernel_q3_1 PUBLIC -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPU=${NR_DPU} -DPTR_TYPE=key_ptrtext)

add_executable(kernel_q3_2 ${DPU_SOURCES_2})
target_compile_definitions(kernel_q3_2 PUBLIC NR_TASKLETS=${NR_TASKLETS} NR_DPU=${NR_DPU} PTR_TYPE=key_ptr32)
target_link_options(kernel_q3_2 PUBLIC -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPU=${NR_DPU} -DPTR_TYPE=key_ptr32)

add_executable(kernel_q3_3 ${DPU_SOURCES_3})
target_compile_definitions(kernel_q3_3 PUBLIC NR_TASKLETS=${NR_TASKLETS} NR_DPU=${NR_DPU} PTR_TYPE=key_ptr32)
target_link_options(kernel_q3_3 PUBLIC -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPU=${NR_DPU} -DPTR_TYPE=key_ptr32)

add_executable(kernel_q3_4 ${DPU_SOURCES_4})
target_compile_definitions(kernel_q3_4 PUBLIC NR_TASKLETS=${NR_TASKLETS} NR_DPU=${NR_DPU} PTR_TYPE=key_ptr32)
target_link_options(kernel_q3_4 PUBLIC -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPU=${NR_DPU} -DPTR_TYPE=key_ptr32)

add_executable(kernel_q3_5 ${DPU_SOURCES_5})
target_compile_definitions(kernel_q3_5 PUBLIC NR_TASKLETS=${NR_TASKLETS} NR_DPU=${NR_DPU} PTR_TYPE=keyptr_out SORT_BLOCK_SIZE=32)
target_link_options(kernel_q3_5 PUBLIC -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPU=${NR_DPU} -DPTR_TYPE=keyptr_out -DSORT_BLOCK_SIZE=32)