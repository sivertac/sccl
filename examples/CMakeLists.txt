cmake_minimum_required(VERSION 3.25)

add_executable(compute_shader_test
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_link_libraries(compute_shader_test PRIVATE compute_interface binary_util)

target_compile_features(compute_shader_test PRIVATE cxx_std_20)
target_compile_options(compute_shader_test PRIVATE -Wall -Wextra -Wswitch)

target_link_libraries(compute_shader_test PUBLIC Vulkan::Vulkan)
target_link_libraries(compute_shader_test PUBLIC glfw)
target_link_libraries(compute_shader_test PUBLIC glm::glm)

add_subdirectory(shaders)

# Add the shader to the target
add_dependencies(compute_shader_test compile_compute_shader)
