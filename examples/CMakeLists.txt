cmake_minimum_required(VERSION 3.25)

add_subdirectory(shaders)

add_library(examples_common
    ${CMAKE_CURRENT_SOURCE_DIR}/examples_common.cpp
)
target_include_directories(examples_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(examples_common PRIVATE cxx_std_20)
target_compile_options(examples_common PRIVATE -Wall -Wextra -Wswitch)
target_link_libraries(examples_common PUBLIC sccl binary_util Vulkan::Vulkan)

add_executable(compute_basic
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_basic.cpp
)
target_include_directories(compute_basic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(compute_basic PRIVATE examples_common)
target_compile_features(compute_basic PRIVATE cxx_std_20)
target_compile_options(compute_basic PRIVATE -Wall -Wextra -Wswitch)
add_dependencies(compute_basic compute_basic_shader)

add_executable(compute_reduce
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_reduce.cpp
)
target_include_directories(compute_reduce PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(compute_reduce PRIVATE examples_common)
target_compile_features(compute_reduce PRIVATE cxx_std_20)
target_compile_options(compute_reduce PRIVATE -Wall -Wextra -Wswitch)
add_dependencies(compute_reduce compute_reduce_shader)

add_executable(compute_reduce_batch
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_reduce_batch.cpp
)
target_include_directories(compute_reduce_batch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(compute_reduce_batch PRIVATE examples_common)
target_compile_features(compute_reduce_batch PRIVATE cxx_std_20)
target_compile_options(compute_reduce_batch PRIVATE -Wall -Wextra -Wswitch)
add_dependencies(compute_reduce_batch compute_reduce_shader_batch)

add_executable(compute_matrix_multiply
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_matrix_multiply.cpp
)
target_include_directories(compute_matrix_multiply PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(compute_matrix_multiply PRIVATE examples_common)
target_compile_features(compute_matrix_multiply PRIVATE cxx_std_20)
target_compile_options(compute_matrix_multiply PRIVATE -Wall -Wextra -Wswitch)
add_dependencies(compute_matrix_multiply compute_matrix_multiply_shader)

add_executable(multi_gpu_dmabuf
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_dmabuf.cpp
)
target_include_directories(multi_gpu_dmabuf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(multi_gpu_dmabuf PRIVATE examples_common)
target_compile_features(multi_gpu_dmabuf PRIVATE cxx_std_20)
target_compile_options(multi_gpu_dmabuf PRIVATE -Wall -Wextra -Wswitch)
