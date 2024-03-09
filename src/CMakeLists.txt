cmake_minimum_required(VERSION 3.25)

add_subdirectory(compute_interface)

target_sources(compute_shader_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_util.cpp
)
target_include_directories(compute_shader_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(compute_shader_test PRIVATE compute_interface)
