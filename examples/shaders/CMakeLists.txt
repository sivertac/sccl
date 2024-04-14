cmake_minimum_required(VERSION 3.25)

compile_shader(
    compute_basic_shader
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_basic_shader.comp
    ${CMAKE_CURRENT_BINARY_DIR}/compute_basic_shader.spv
)

compile_shader(
    compute_reduce_shader
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_reduce_shader.comp
    ${CMAKE_CURRENT_BINARY_DIR}/compute_reduce_shader.spv
)

compile_shader(
    compute_reduce_shader_batch
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_reduce_shader_batch.comp
    ${CMAKE_CURRENT_BINARY_DIR}/compute_reduce_shader_batch.spv
)
