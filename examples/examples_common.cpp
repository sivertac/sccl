
#include "examples_common.hpp"

#include <binary_util.hpp>
#include <cerrno>
#include <iostream>

void print_data_buffer(const sccl_buffer_t buffer, size_t size)
{
    void *data;
    UNWRAP_SCCL_ERROR(sccl_host_map_buffer(buffer, &data, 0, size));

    print_buffer_binary_xxd((const char *)data, size);

    sccl_host_unmap_buffer(buffer);
}

// NOLINTBEGIN
std::optional<std::string> read_file(const char *filepath)
{
    // Read the shader code from the file
    FILE *file = fopen(filepath, "rb");
    if (file == NULL) {
        return std::nullopt;
    }
    fseek(file, 0, SEEK_END);
    size_t size = ftell(file);

    rewind(file);

    char *data = (char *)malloc(size);
    if (data == nullptr) {
        fclose(file);
        return std::nullopt;
    }

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
// NOLINTEND
