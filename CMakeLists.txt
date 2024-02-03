cmake_minimum_required(VERSION 3.25)

project(vulkan-compute-shader-test VERSION 0.0.1 LANGUAGES CXX)

find_package(Vulkan REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 3.3.2 REQUIRED)


add_executable(compute-shader-test)

target_compile_features(compute-shader-test PRIVATE cxx_std_20)
target_compile_options(compute-shader-test PRIVATE -Wall -Wextra -Wswitch)

target_link_libraries(compute-shader-test PUBLIC Vulkan::Vulkan)
target_link_libraries(compute-shader-test PUBLIC glfw)
target_link_libraries(compute-shader-test PUBLIC glm::glm)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/shaders)
