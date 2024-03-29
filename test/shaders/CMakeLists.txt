cmake_minimum_required(VERSION 3.25)

compile_shader(
    noop_shader
    ${CMAKE_CURRENT_SOURCE_DIR}/noop_shader.comp
    ${CMAKE_CURRENT_BINARY_DIR}/noop_shader.spv
)

compile_shader(
    buffer_layout_shader
    ${CMAKE_CURRENT_SOURCE_DIR}/buffer_layout_shader.comp
    ${CMAKE_CURRENT_BINARY_DIR}/buffer_layout_shader.spv
)

compile_shader(
    specialization_constants_shader
    ${CMAKE_CURRENT_SOURCE_DIR}/specialization_constants_shader.comp
    ${CMAKE_CURRENT_BINARY_DIR}/specialization_constants_shader.spv
)

compile_shader(
    push_constants_shader
    ${CMAKE_CURRENT_SOURCE_DIR}/push_constants_shader.comp
    ${CMAKE_CURRENT_BINARY_DIR}/push_constants_shader.spv
)

compile_shader(
    copy_buffer_shader
    ${CMAKE_CURRENT_SOURCE_DIR}/copy_buffer_shader.comp
    ${CMAKE_CURRENT_BINARY_DIR}/copy_buffer_shader.spv
)
