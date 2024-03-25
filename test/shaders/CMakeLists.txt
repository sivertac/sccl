cmake_minimum_required(VERSION 3.25)

compile_shader(
    noop_shader
    ${CMAKE_CURRENT_SOURCE_DIR}/noop_shader.comp
    ${CMAKE_CURRENT_BINARY_DIR}/noop_shader.spv
)
