
#include "examples_common.hpp"

#include <binary_util.hpp>
#include <iostream>

void print_data_buffer(const sccl_buffer_t buffer, size_t size)
{
    void *data;
    UNWRAP_SCCL_ERROR(sccl_host_map_buffer(buffer, &data, 0, size));

    print_buffer_binary_xxd((const char *)data, size);

    sccl_host_unmap_buffer(buffer);
}

/*
void print_data_buffers(const ComputeDevice *compute_device,
                        size_t num_elements, VkDeviceMemory input_buffer_memory,
                        VkDeviceMemory output_buffer_memory)
{
    VkDeviceSize buffer_size = sizeof(int) * num_elements;

    // Map and fill the buffers
    void *inputDataPtr;
    void *outputDataPtr;

    vkMapMemory(compute_device->m_device, input_buffer_memory, 0, buffer_size,
                0, &inputDataPtr);
    vkMapMemory(compute_device->m_device, output_buffer_memory, 0, buffer_size,
                0, &outputDataPtr);

    // Print inputDataPtr and outputDataPtr
    std::cout << "Input buffer:" << std::endl;
    print_buffer_binary_xxd((const char *)inputDataPtr, buffer_size);

    std::cout << "Output buffer:" << std::endl;
    print_buffer_binary_xxd((const char *)outputDataPtr, buffer_size);

    vkUnmapMemory(compute_device->m_device, input_buffer_memory);
    vkUnmapMemory(compute_device->m_device, output_buffer_memory);
}
*/

std::optional<std::string> read_file(const char *filepath)
{
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
