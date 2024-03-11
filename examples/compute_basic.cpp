
#include "examples_common.hpp"

#include <iostream>
#include <string>

#include <compute_interface.h>

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
    UNWRAP_VKRESULT(create_compute_buffer(&compute_device, 1000, &input_buffer,
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    ComputeBuffer output_buffer;
    UNWRAP_VKRESULT(create_compute_buffer(&compute_device, 1000, &output_buffer,
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));

    // set up SpecializationInfo, this is used to set constants in shader, for
    // example work group size
    int32_t constant_1 = 2;
    VkSpecializationMapEntry specialization_map_entry = {};
    specialization_map_entry.constantID = 0;
    specialization_map_entry.offset = 0;
    specialization_map_entry.size = sizeof(constant_1);
    VkSpecializationInfo specialization_info = {};
    specialization_info.mapEntryCount = 1;
    specialization_info.pMapEntries = &specialization_map_entry;
    specialization_info.dataSize = sizeof(constant_1);
    specialization_info.pData = &constant_1;

    // create compute pipeline
    ComputePipeline compute_pipeline;
    UNWRAP_VKRESULT(
        create_compute_pipeline(&compute_device, shader_source.value().data(),
                                shader_source.value().size(), 1, 1, 0,
                                &specialization_info, &compute_pipeline));

    // create descriptor sets
    ComputeDescriptorSets compute_descriptor_sets;
    UNWRAP_VKRESULT(create_compute_descriptor_sets(
        &compute_device, &compute_pipeline, &compute_descriptor_sets));

    // update descriptor sets
    UNWRAP_VKRESULT(update_compute_descriptor_sets(
        &compute_device, &compute_pipeline, &input_buffer, &output_buffer, NULL,
        &compute_descriptor_sets));

    // run compute
    uint32_t size = 10;
    print_data_buffers(&compute_device, size, input_buffer.m_buffer_memory,
                       output_buffer.m_buffer_memory);
    run_compute_pipeline_sync(&compute_device, &compute_pipeline,
                              &compute_descriptor_sets, size, 1, 1);
    print_data_buffers(&compute_device, size, input_buffer.m_buffer_memory,
                       output_buffer.m_buffer_memory);

    // cleanup
    destroy_compute_buffer(&compute_device, &input_buffer);
    destroy_compute_buffer(&compute_device, &output_buffer);
    destroy_compute_pipeline(&compute_device, &compute_pipeline);
    destroy_compute_device(&compute_device);

    return EXIT_SUCCESS;
}
