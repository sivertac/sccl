cmake_minimum_required(VERSION 3.25)

project(vulkan-compute-shader-test VERSION 0.0.1 LANGUAGES CXX C)

find_package(Vulkan REQUIRED)
find_package(Vulkan COMPONENTS glslc)
find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)
find_package(glm REQUIRED)
find_package(glfw3 3.3.2 REQUIRED)

# clang-tidy
set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-config-file=${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy")

add_library(compute_interface)
target_compile_features(compute_interface PRIVATE c_std_17)
target_compile_options(compute_interface PRIVATE -Wall -Wextra -Wswitch)

add_executable(compute_shader_test)

target_link_libraries(compute_shader_test PRIVATE compute_interface)

target_compile_features(compute_shader_test PRIVATE cxx_std_20)
target_compile_options(compute_shader_test PRIVATE -Wall -Wextra -Wswitch)

target_link_libraries(compute_shader_test PUBLIC Vulkan::Vulkan)
target_link_libraries(compute_shader_test PUBLIC glfw)
target_link_libraries(compute_shader_test PUBLIC glm::glm)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/shaders)
