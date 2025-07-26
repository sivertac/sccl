cmake_minimum_required(VERSION 3.25)

project(vulkan-compute-meme VERSION 0.0.1 LANGUAGES CXX C)

find_package(Vulkan REQUIRED)
find_package(Vulkan COMPONENTS glslc)
find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(cmake)

# clang-tidy
set(CMAKE_CXX_CLANG_TIDY "clang-tidy-19;-config-file=${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy")

add_compile_options(-Wuninitialized -Werror -Wall)
#add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
#add_link_options(-fsanitize=address)

add_subdirectory(src)
add_subdirectory(examples)

if (NOT NO_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
