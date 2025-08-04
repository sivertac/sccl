cmake_minimum_required(VERSION 3.25)

include(FetchContent)
FetchContent_Declare(
    bison
    URL https://ftp.gnu.org/gnu/bison/bison-3.8.2.tar.xz
    URL_HASH SHA256=9bba0214ccf7f1079c5d59210045227bcf619519840ebfa80cd3849cff5a5bf2

    DOWNLOAD_NO_EXTRACT true
)
#FetchContent_MakeAvailable(bison)
FetchContent_Populate(
    bison   
)

# Unpack and configure
execute_process(
    COMMAND tar -xf bison-3.8.2.tar.xz 
    WORKING_DIRECTORY ${bison_SOURCE_DIR}
)

set(bison_INSTALL_DIR ${CMAKE_BINARY_DIR}/bison)

execute_process(
    COMMAND ${bison_SOURCE_DIR}/bison-3.8.2/configure --prefix=${bison_INSTALL_DIR} 
    WORKING_DIRECTORY ${bison_BINARY_DIR}
)

# Build
add_custom_target(
    bison_build
    COMMAND make
    COMMAND make install
    WORKING_DIRECTORY ${bison_BINARY_DIR}
)

set(BISON_EXECUTABLE ${bison_INSTALL_DIR}/bin/bison)
set(YACC_EXECUTABLE ${bison_INSTALL_DIR}/bin/yacc)

