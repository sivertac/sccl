cmake_minimum_required(VERSION 3.25)

add_library(sccl
    ${CMAKE_CURRENT_SOURCE_DIR}/environment_variables.c
    ${CMAKE_CURRENT_SOURCE_DIR}/instance.c
    ${CMAKE_CURRENT_SOURCE_DIR}/alloc.c
    ${CMAKE_CURRENT_SOURCE_DIR}/device.c
    ${CMAKE_CURRENT_SOURCE_DIR}/buffer.c
    ${CMAKE_CURRENT_SOURCE_DIR}/stream.c
)
target_compile_features(sccl PRIVATE c_std_17)
target_compile_options(sccl PRIVATE -Wall -Wextra -Wswitch)
target_include_directories(sccl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sccl PRIVATE Vulkan::Vulkan)

set_target_properties(sccl PROPERTIES PUBLIC_HEADER
    "${CMAKE_CURRENT_SOURCE_DIR}/sccl.h"
)
