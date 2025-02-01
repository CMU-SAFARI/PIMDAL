cmake_minimum_required(VERSION 3.0.0)
project(upmem-sort-host VERSION 0.1.0)

FIND_PACKAGE(Arrow REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)

include("${UPMEM_HOME}/share/upmem/cmake/include/host/DpuHost.cmake")

set(CMAKE_CXX_FLAGS "--std=c++14 -O3 -Wno-unused-result -g3 -fopenmp")
link_directories("${DPU_HOST_LINK_DIRECTORIES}")

add_executable(host_bjoin host_join_sync.cpp)
target_include_directories(host_bjoin PUBLIC "${DPU_HOST_INCLUDE_DIRECTORIES}" PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_compile_definitions(host_bjoin PUBLIC NR_DPU=${NR_DPU} INNER_SIZE=${INNER_SIZE} OUTER_SIZE=${OUTER_SIZE})
target_link_options(host_bjoin PUBLIC -DNR_DPU=${NR_DPU} -DINNER_SIZE=${INNER_SIZE} -DOUTER_SIZE=${OUTER_SIZE})
target_link_libraries(host_bjoin PUBLIC ${DPU_HOST_LIBRARIES} PRIVATE Arrow::arrow_shared OpenMP::OpenMP_CXX)