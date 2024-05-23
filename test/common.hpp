#pragma once
#ifndef COMMON_HEADER
#define COMMON_HEADER

#include <inttypes.h>
#include <optional>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#define SCCL_TEST_ASSERT(cmd)                                                  \
    do {                                                                       \
        sccl_error_t SCCL_TEST_ASSERT_error = cmd;                             \
        ASSERT_EQ(SCCL_TEST_ASSERT_error, sccl_success);                       \
    } while (0)

inline uint32_t get_environment_gpu_index()
{
    const char *str = getenv("SCCL_TEST_GPU_INDEX");
    if (str == NULL) {
        return 0;
    }
    uint32_t gpu_index = static_cast<uint32_t>(std::atoi(str));
    printf("environment - SCCL_TEST_GPU_INDEX = %" PRIu32 "\n", gpu_index);
    return gpu_index;
}

inline const char *get_environment_shaders_dir()
{
    const char *str = getenv("SCCL_TEST_SHADERS_DIR");
    if (str == NULL) {
        return "shaders"; /* default relative path */
    }
    printf("environment - SCCL_TEST_SHADERS_DIR = %s\n", str);
    return str;
}

inline std::optional<std::string> read_file(const char *filepath)
{
    printf("filepath = %s\n", filepath);
    // Read the shader code from the file
    FILE *file = fopen(filepath, "rb");
    if (!file) {
        return std::nullopt;
    }
    fseek(file, 0, SEEK_END);
    size_t size = ftell(file);
    rewind(file);
    char *data = (char *)malloc(size);
    size_t read_size = fread(data, 1, size, file);
    if (read_size < size) {
        free(data);
        fclose(file);
        return std::nullopt;
    }
    fclose(file);

    std::string ret(data, size);
    free(data);

    return ret;
}

inline std::optional<std::string> read_test_shader(const char *shader_file_name)
{
    return read_file((std::string(get_environment_shaders_dir()) + "/" +
                      std::string(shader_file_name))
                         .c_str());
}

#endif // COMMON_HEADER