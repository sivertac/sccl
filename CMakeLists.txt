cmake_minimum_required(VERSION 3.25)

project(vulkan-compute-meme VERSION 0.0.1 LANGUAGES CXX C)

find_package(Vulkan REQUIRED)
find_package(Vulkan COMPONENTS glslc)
find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)
find_package(glm REQUIRED)
find_package(glfw3 3.3.2 REQUIRED)

# clang-tidy
set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-config-file=${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy")

add_subdirectory(src)
add_subdirectory(examples)