
#include "examples_common.hpp"

#include <inttypes.h>
#include <iostream>
#include <string>
#include <vector>

#include <compute_interface.h>

const char *COMPUTE_SHADER_PATH = "shaders/compute_reduce_shader.spv";

struct UniformBufferObject {
    uint numberOfRanks;
    uint rankSize;
};

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    ComputeDevice compute_device = {};
    UNWRAP_VKRESULT(create_compute_device(true, &compute_device));

    // query device limits
    VkPhysicalDeviceProperties physical_device_properties;
    vkGetPhysicalDeviceProperties(compute_device.m_physical_device,
                                  &physical_device_properties);
    printf("physical_device_properties.limits.maxComputeWorkGroupSize[0] = "
           "%" PRIu32 "\n",
           physical_device_properties.limits.maxComputeWorkGroupSize[0]);
    printf("physical_device_properties.limits.maxComputeWorkGroupSize[1] = "
           "%" PRIu32 "\n",
           physical_device_properties.limits.maxComputeWorkGroupSize[1]);
    printf("physical_device_properties.limits.maxComputeWorkGroupSize[2] = "
           "%" PRIu32 "\n",
           physical_device_properties.limits.maxComputeWorkGroupSize[2]);

    // set rank sizes
    unsigned int number_of_ranks = 4;
    unsigned int rank_size =
        physical_device_properties.limits.maxComputeWorkGroupCount[0];
    size_t rank_size_bytes = rank_size * sizeof(int);
    printf("number_of_ranks = %d\n", number_of_ranks);
    printf("rank_size = %d\n", rank_size);
    printf("rank_size_bytes = %lu\n", rank_size_bytes);

    UniformBufferObject ubo = {};
    ubo.numberOfRanks = number_of_ranks;
    ubo.rankSize = rank_size;

    // read shader
    auto shader_source = read_file(COMPUTE_SHADER_PATH);
    if (!shader_source.has_value()) {
        fprintf(stderr, "Failed to open shader file: %s\n",
                COMPUTE_SHADER_PATH);
        exit(EXIT_FAILURE);
    }

    // create input buffer
    ComputeBuffer input_buffer;
    UNWRAP_VKRESULT(create_compute_buffer(
        &compute_device, rank_size_bytes * number_of_ranks, &input_buffer,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    std::vector<int> input_data(rank_size * number_of_ranks);

    // fill input buffer
    for (size_t i = 0; i < rank_size * number_of_ranks; ++i) {
        input_data[i] = static_cast<int>(i);
    }
    UNWRAP_VKRESULT(write_to_compute_buffer(
        &compute_device, &input_buffer, 0, sizeof(int) * input_data.size(),
        static_cast<const void *>(input_data.data())));

    // create output buffer
    ComputeBuffer output_buffer;
    UNWRAP_VKRESULT(create_compute_buffer(&compute_device, rank_size_bytes,
                                          &output_buffer,
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));

    // create uniform buffer
    ComputeBuffer uniform_buffer_object;
    UNWRAP_VKRESULT(create_compute_buffer(
        &compute_device, sizeof(UniformBufferObject), &uniform_buffer_object,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));

    // fill uniform buffer
    UNWRAP_VKRESULT(write_to_compute_buffer(
        &compute_device, &uniform_buffer_object, 0, sizeof(UniformBufferObject),
        static_cast<const void *>(&ubo)));

    // create compute pipeline
    ComputePipeline compute_pipeline;
    UNWRAP_VKRESULT(create_compute_pipeline(
        &compute_device, shader_source.value().data(),
        shader_source.value().size(), 1, 1, 1, NULL, &compute_pipeline));

    // create descriptor set
    ComputeDescriptorSets compute_descriptor_sets;
    UNWRAP_VKRESULT(create_compute_descriptor_sets(
        &compute_device, &compute_pipeline, &compute_descriptor_sets));

    // update descriptor set
    UNWRAP_VKRESULT(update_compute_descriptor_sets(
        &compute_device, &compute_pipeline, &input_buffer, &output_buffer,
        &uniform_buffer_object, &compute_descriptor_sets));

    // run compute
    uint32_t size = rank_size;
    UNWRAP_VKRESULT(
        run_compute_pipeline_sync(&compute_device, &compute_pipeline,
                                  &compute_descriptor_sets, size, 1, 1));

    // verify output data
    std::vector<int> output_data(rank_size);
    UNWRAP_VKRESULT(read_from_compute_buffer(&compute_device, &output_buffer, 0,
                                             output_buffer.m_size,
                                             output_data.data()));
    for (size_t i = 0; i < rank_size; ++i) {
        // compute expected data
        int expected_value = 0;
        for (size_t rank = 0; rank < number_of_ranks; ++rank) {
            expected_value += input_data[rank * rank_size + i];
        }
        if (output_data[i] != expected_value) {
            printf("Unexpected value at i = %lu: output_data[i] = %d, "
                   "expected_value = %d\n",
                   i, output_data[i], expected_value);
        }
    }
    printf("Success, all values expected!\n");

    // cleanup
    destroy_compute_buffer(&compute_device, &input_buffer);
    destroy_compute_buffer(&compute_device, &output_buffer);
    destroy_compute_buffer(&compute_device, &uniform_buffer_object);
    destroy_compute_pipeline(&compute_device, &compute_pipeline);
    destroy_compute_device(&compute_device);

    return EXIT_SUCCESS;
}
