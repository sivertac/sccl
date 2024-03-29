cmake_minimum_required(VERSION 3.25)

add_subdirectory(shaders)

add_custom_target(test_verbose COMMAND ${CMAKE_CTEST_COMMAND} "--output-on-failure")

# So that GoogleTest does not get installed with this project
set(INSTALL_GTEST OFF)

# Download and unpack googletest at configure time
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.12.1
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

function(create_test target)
    # parse arguments
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SOURCES DEPENDS)
    cmake_parse_arguments(CREATE_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # create test
    add_executable(${target} ${CREATE_TEST_SOURCES})
    target_link_libraries(${target} PUBLIC gtest_main)
    target_link_libraries(${target} PRIVATE sccl)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wswitch)
    if (CREATE_TEST_DEPENDS) # check if not empty
        add_dependencies(${target} ${CREATE_TEST_DEPENDS})
    endif()
    gtest_discover_tests(${target})
endfunction()

# add tests
create_test(test_sccl_instance SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_sccl_instance.cpp)
create_test(test_sccl_device SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_sccl_device.cpp)
create_test(test_sccl_buffer SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_sccl_buffer.cpp)
create_test(test_sccl_stream SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_sccl_stream.cpp)
create_test(test_sccl_copy_buffer SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_sccl_copy_buffer.cpp)
create_test(test_sccl_shader SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_sccl_shader.cpp DEPENDS noop_shader specialization_constants_shader push_constants_shader buffer_layout_shader)

