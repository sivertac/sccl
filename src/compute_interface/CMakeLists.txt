cmake_minimum_required(VERSION 3.25)

target_sources(compute_interface PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_interface.c
)
target_include_directories(compute_interface PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
