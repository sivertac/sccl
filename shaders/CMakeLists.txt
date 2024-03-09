cmake_minimum_required(VERSION 3.25)

# Set your source directory and shader source
set(SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/test_compute_shader.comp)
set(SHADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_compute_shader.spv)

# Use glslangValidator to compile the shader
#add_custom_target(
#    compile-compute-shader
#    BYPRODUCTS ${SHADER_OUTPUT}
#    COMMAND glslc -std=460 -o ${SHADER_OUTPUT} ${SHADER_SOURCE}
#    DEPENDS ${SHADER_SOURCE}
#    COMMENT "Compiling compute shader"
#)

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
    add_custom_target(compile-compute-shader DEPENDS ${output})
endfunction()

compile_shader(
    compile-compute-shader
    ${CMAKE_CURRENT_SOURCE_DIR}/test_compute_shader.comp
    ${CMAKE_CURRENT_BINARY_DIR}/test_compute_shader.spv
)

# Add the shader to the target
add_dependencies(compute-shader-test compile-compute-shader)
