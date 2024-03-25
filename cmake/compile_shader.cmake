cmake_minimum_required(VERSION 3.25)

macro(compile_shader target source output)
    add_custom_command(
        OUTPUT ${output}
        DEPENDS ${source}
        DEPFILE ${output}.d
        COMMAND
            ${glslc_executable}
            -MD -MF ${output}.d
            -o ${output}
            ${source}
    )
    add_custom_target(${target} DEPENDS ${output})
endmacro()