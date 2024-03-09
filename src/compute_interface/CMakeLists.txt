cmake_minimum_required(VERSION 3.25)

add_library(compute_interface
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_interface.c
)
target_compile_features(compute_interface PRIVATE c_std_17)
target_compile_options(compute_interface PRIVATE -Wall -Wextra -Wswitch)
target_include_directories(compute_interface PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
