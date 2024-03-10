cmake_minimum_required(VERSION 3.25)

add_subdirectory(shaders)

add_executable(compute_basic
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_basic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/examples_common.cpp
)
target_include_directories(compute_basic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(compute_basic PRIVATE compute_interface binary_util)
target_compile_features(compute_basic PRIVATE cxx_std_20)
target_compile_options(compute_basic PRIVATE -Wall -Wextra -Wswitch)
target_link_libraries(compute_basic PUBLIC Vulkan::Vulkan)
target_link_libraries(compute_basic PUBLIC glfw)
target_link_libraries(compute_basic PUBLIC glm::glm)
add_dependencies(compute_basic compute_basic_shader)

add_executable(compute_reduce
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/examples_common.cpp
)
target_include_directories(compute_reduce PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(compute_reduce PRIVATE compute_interface binary_util)
target_compile_features(compute_reduce PRIVATE cxx_std_20)
target_compile_options(compute_reduce PRIVATE -Wall -Wextra -Wswitch)
target_link_libraries(compute_reduce PUBLIC Vulkan::Vulkan)
target_link_libraries(compute_reduce PUBLIC glfw)
target_link_libraries(compute_reduce PUBLIC glm::glm)
add_dependencies(compute_reduce compute_reduce_shader)
