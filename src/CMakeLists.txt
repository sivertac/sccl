cmake_minimum_required(VERSION 3.25)

target_sources(compute-shader-test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_interface.cpp
)

target_include_directories(compute-shader-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})