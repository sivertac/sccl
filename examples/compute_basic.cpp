#include <iostream>
#include <optional>
#include <string>

#include <binary_util.hpp>
#include <compute_interface.h>
#include <vulkan/vk_enum_string_helper.h>

#define UNWRAP_VKRESULT(result)                                                \
    do {                                                                       \
        if (result != VK_SUCCESS) {                                            \
            fprintf(stderr, "Vulkan error: %s\n", string_VkResult(result));    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

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
    fread(data, 1, size, file);
    fclose(file);

    std::string ret(data, size);
    free(data);

    return ret;
}

const char *COMPUTE_SHADER_PATH = "shaders/compute_basic_shader.spv";

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    ComputeDevice compute_device = {};
    UNWRAP_VKRESULT(create_compute_device(true, &compute_device));

    // read shader
    auto shader_source = read_file(COMPUTE_SHADER_PATH);
    if (!shader_source.has_value()) {
        fprintf(stderr, "Failed to open shader file: %s\n",
                COMPUTE_SHADER_PATH);
        exit(EXIT_FAILURE);
    }

    // create buffers
    ComputeBuffer input_buffer;
    UNWRAP_VKRESULT(
        create_compute_buffer(&compute_device, 1000, &input_buffer));
    ComputeBuffer output_buffer;
    UNWRAP_VKRESULT(
        create_compute_buffer(&compute_device, 1000, &output_buffer));

    // create compute pipeline
    ComputePipeline compute_pipeline;
    UNWRAP_VKRESULT(create_compute_pipeline(
        &compute_device, shader_source.value().data(),
        shader_source.value().size(), 1, 1, &compute_pipeline));

    // create descriptor set
    ComputeDescriptorSet compute_descriptor_set;
    UNWRAP_VKRESULT(create_compute_descriptor_set(
        &compute_device, &compute_pipeline, &compute_descriptor_set));

    // update descriptor set
    UNWRAP_VKRESULT(update_compute_descriptor_set(
        &compute_device, &input_buffer, 1, &output_buffer, 1,
        &compute_descriptor_set));

    // run compute
    uint32_t size = 10;
    print_data_buffers(&compute_device, size, input_buffer.m_buffer_memory,
                       output_buffer.m_buffer_memory);
    run_compute_pipeline_sync(&compute_device, &compute_pipeline,
                              &compute_descriptor_set, size, 1, 1);
    print_data_buffers(&compute_device, size, input_buffer.m_buffer_memory,
                       output_buffer.m_buffer_memory);

    // cleanup
    destroy_compute_buffer(&compute_device, &input_buffer);
    destroy_compute_buffer(&compute_device, &output_buffer);
    destroy_compute_pipeline(&compute_device, &compute_pipeline);
    destroy_compute_device(&compute_device);

    return EXIT_SUCCESS;
}
