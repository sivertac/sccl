cmake_minimum_required(VERSION 3.25)

project(vulkan-compute-shader-test VERSION 0.0.1 LANGUAGES CXX)

find_package(Vulkan REQUIRED)

add_executable(compute-shader-test)

target_compile_features(compute-shader-test PRIVATE cxx_std_20)
target_compile_options(compute-shader-test PRIVATE -Wall -Wextra -Wswitch)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
