cmake_minimum_required(VERSION 3.0.0)
project(upmem-aggregate-host VERSION 0.1.0)

FIND_PACKAGE(Arrow REQUIRED)
FIND_PACKAGE(ArrowAcero REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)

include("${UPMEM_HOME}/share/upmem/cmake/include/host/DpuHost.cmake")

set(CMAKE_CXX_FLAGS "--std=c++14 -O3 -Wno-unused-result -g3 -fopenmp")
link_directories("${DPU_HOST_LINK_DIRECTORIES}")

# Change executable for synchronous execution
add_executable(host_saggregate host_aggregate_sync.cpp)
target_include_directories(host_saggregate PUBLIC "${DPU_HOST_INCLUDE_DIRECTORIES}" PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_compile_definitions(host_saggregate PUBLIC NR_DPU=${NR_DPU} BUFFER_SIZE=${BUFFER_SIZE} PERF=${PERF} NR_TASKLETS=${NR_TASKLETS})
target_link_options(host_saggregate PUBLIC -DNR_DPU=${NR_DPU} -DBUFFER_SIZE=${BUFFER_SIZE} -DPERF=${PERF} -DNR_TASKLETS=${NR_TASKLETS})
target_link_libraries(host_saggregate PUBLIC ${DPU_HOST_LIBRARIES} PRIVATE Arrow::arrow_shared ArrowAcero::arrow_acero_shared OpenMP::OpenMP_CXX)