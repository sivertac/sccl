cmake_minimum_required(VERSION 3.25)

# Set your source directory and shader source
set(SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/test_compute_shader.comp)
set(SHADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_compute_shader.spv)

# Use glslangValidator to compile the shader
add_custom_target(
    compile-compute-shader
    BYPRODUCTS ${SHADER_OUTPUT}
    COMMAND glslangValidator -V ${SHADER_SOURCE} -o ${SHADER_OUTPUT}
    DEPENDS ${SHADER_SOURCE}
    COMMENT "Compiling compute shader"
)

# Add the shader to the target
add_dependencies(compute-shader-test compile-compute-shader)


