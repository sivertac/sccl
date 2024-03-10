cmake_minimum_required(VERSION 3.25)

add_subdirectory(shaders)

add_executable(compute_basic
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_basic.cpp
)

target_link_libraries(compute_basic PRIVATE compute_interface binary_util)

target_compile_features(compute_basic PRIVATE cxx_std_20)
target_compile_options(compute_basic PRIVATE -Wall -Wextra -Wswitch)
target_link_libraries(compute_basic PUBLIC Vulkan::Vulkan)
target_link_libraries(compute_basic PUBLIC glfw)
target_link_libraries(compute_basic PUBLIC glm::glm)
# Add the shader to the target
add_dependencies(compute_basic compute_basic_shader)
