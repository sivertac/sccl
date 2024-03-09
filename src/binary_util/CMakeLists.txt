cmake_minimum_required(VERSION 3.25)

add_library(binary_util
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_util.cpp
)
target_compile_features(binary_util PRIVATE cxx_std_20)
target_compile_options(binary_util PRIVATE -Wall -Wextra -Wswitch)
target_include_directories(binary_util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

