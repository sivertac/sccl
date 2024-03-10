cmake_minimum_required(VERSION 3.25)

function(compile_shader target source output)
    add_custom_command(
        OUTPUT ${output}
        DEPENDS ${source}
        DEPFILE ${output}.d
        COMMAND
            ${glslc_executable}
            -MD -MF ${output}.d
            -o ${output}
            ${source}
    )
    add_custom_target(${target} DEPENDS ${output})
endfunction()

compile_shader(
    compute_basic_shader
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_basic_shader.comp
    ${CMAKE_CURRENT_BINARY_DIR}/compute_basic_shader.spv
)

