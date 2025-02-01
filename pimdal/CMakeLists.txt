cmake_minimum_required(VERSION 3.0.0)
project(pimdal)

if(NOT DEFINED UPMEM_HOME)
  if("$ENV{UPMEM_HOME}" STREQUAL "")
    set(UPMEM_HOME "/usr")
  else()
    set(UPMEM_HOME $ENV{UPMEM_HOME})
  endif()
endif()

set(NR_DPU 16UL)
set(NR_RANKS 1)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

include_directories(
    PUBLIC "${CMAKE_SOURCE_DIR}/support"
)

add_subdirectory(main)
