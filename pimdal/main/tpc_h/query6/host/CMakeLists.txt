cmake_minimum_required(VERSION 3.0.0)
project(upmem-sort-host VERSION 0.1.0)

FIND_PACKAGE(Arrow REQUIRED)
FIND_PACKAGE(ArrowAcero REQUIRED)
find_package(Parquet REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)

include("${UPMEM_HOME}/share/upmem/cmake/include/host/DpuHost.cmake")

set(CMAKE_CXX_FLAGS "--std=c++14 -O3 -Wno-unused-result -g3 -fopenmp")
link_directories("${DPU_HOST_LINK_DIRECTORIES}")

add_executable(host_q6 host_q6.cpp)
target_include_directories(host_q6 PUBLIC "${DPU_HOST_INCLUDE_DIRECTORIES}" PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_compile_definitions(host_q6 PUBLIC NR_DPU=${NR_DPU})
target_link_options(host_q6 PUBLIC -DNR_DPU=${NR_DPU})
target_link_libraries(host_q6 PUBLIC ${DPU_HOST_LIBRARIES} PRIVATE Arrow::arrow_shared ArrowAcero::arrow_acero_shared Parquet::parquet_shared OpenMP::OpenMP_CXX)