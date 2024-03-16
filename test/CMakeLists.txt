cmake_minimum_required(VERSION 3.25)

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

function(create_test test_name test_sources)
    add_executable(${test_name} ${test_sources})
    target_link_libraries(${test_name} PUBLIC gtest_main)
    target_link_libraries(${test_name} PRIVATE sccl)
    gtest_discover_tests(${test_name})
endfunction()

# add tests
create_test(test_sccl_instance ${CMAKE_CURRENT_SOURCE_DIR}/test_sccl_instance.cpp)
