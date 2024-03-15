
#include "examples_common.hpp"

#include <chrono>
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

    // query device properties
    VkPhysicalDeviceSubgroupProperties physical_device_subgroup_properties = {};
    physical_device_subgroup_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 physical_device_properties = {};
    physical_device_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    physical_device_properties.pNext = &physical_device_subgroup_properties;
    vkGetPhysicalDeviceProperties2(compute_device.m_physical_device,
                                   &physical_device_properties);
    VkPhysicalDeviceFeatures2 physical_device_features = {};
    physical_device_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    vkGetPhysicalDeviceFeatures2(compute_device.m_physical_device,
                                 &physical_device_features);

    printf("physical_device_properties.properties.limits."
           "maxComputeWorkGroupSize[0] = "
           "%" PRIu32 "\n",
           physical_device_properties.properties.limits
               .maxComputeWorkGroupSize[0]);
    printf("physical_device_properties.properties.limits."
           "maxComputeWorkGroupSize[1] = "
           "%" PRIu32 "\n",
           physical_device_properties.properties.limits
               .maxComputeWorkGroupSize[1]);
    printf("physical_device_properties.properties.limits."
           "maxComputeWorkGroupSize[2] = "
           "%" PRIu32 "\n",
           physical_device_properties.properties.limits
               .maxComputeWorkGroupSize[2]);
    printf("physical_device_properties.properties.limits."
           "maxComputeWorkGroupCount[0] = %" PRIu32 "\n",
           physical_device_properties.properties.limits
               .maxComputeWorkGroupCount[0]);
    printf("physical_device_properties.properties.limits."
           "maxComputeWorkGroupCount[1] = %" PRIu32 "\n",
           physical_device_properties.properties.limits
               .maxComputeWorkGroupCount[1]);
    printf("physical_device_properties.properties.limits."
           "maxComputeWorkGroupCount[2] = %" PRIu32 "\n",
           physical_device_properties.properties.limits
               .maxComputeWorkGroupCount[2]);
    printf("physical_device_properties.properties.limits."
           "maxComputeWorkGroupInvocations = "
           "%" PRIu32 "\n",
           physical_device_properties.properties.limits
               .maxComputeWorkGroupInvocations);
    printf("physical_device_subgroup_properties.subgroupSize = %" PRIu32 "\n",
           physical_device_subgroup_properties.subgroupSize);
    printf("physical_device_properties.properties.limits.maxStorageBufferRange "
           "= %" PRIu32 "\n",
           physical_device_properties.properties.limits.maxStorageBufferRange);
    printf("physical_device_features.features.shaderInt64 = %d\n",
           physical_device_features.features.shaderInt64);

    // set rank sizes
    uint32_t shader_workgroup_size[3];
    shader_workgroup_size[0] = physical_device_subgroup_properties.subgroupSize;
    shader_workgroup_size[1] = 1;
    shader_workgroup_size[2] = 1;

    // distribute
    // physical_device_properties.properties.limits.maxStorageBufferRange
    // accross dimentions
    unsigned int number_of_ranks = 10;
    size_t allocated_size =
        physical_device_properties.properties.limits.maxStorageBufferRange /
        sizeof(int) / number_of_ranks / shader_workgroup_size[0];

    uint32_t shader_workgroup_count[3];
    shader_workgroup_count[0] =
        (allocated_size > physical_device_properties.properties.limits
                              .maxComputeWorkGroupCount[0])
            ? physical_device_properties.properties.limits
                  .maxComputeWorkGroupCount[0]
            : allocated_size;
    allocated_size /= shader_workgroup_count[0];
    shader_workgroup_count[1] =
        (allocated_size > physical_device_properties.properties.limits
                              .maxComputeWorkGroupCount[1])
            ? physical_device_properties.properties.limits
                  .maxComputeWorkGroupCount[1]
            : allocated_size;
    allocated_size /= shader_workgroup_count[1];
    shader_workgroup_count[2] =
        (allocated_size > physical_device_properties.properties.limits
                              .maxComputeWorkGroupCount[2])
            ? physical_device_properties.properties.limits
                  .maxComputeWorkGroupCount[2]
            : allocated_size;
    ;

    for (size_t i = 0; i < 3; ++i) {
        printf("shader_workgroup_count[%lu] = %" PRIu32 "\n", i,
               shader_workgroup_count[i]);
    }

    unsigned int rank_size =
        shader_workgroup_count[0] * shader_workgroup_count[1] *
        shader_workgroup_count[2] * shader_workgroup_size[0] *
        shader_workgroup_size[1] * shader_workgroup_size[2];
    size_t rank_size_bytes = rank_size * sizeof(int);
    printf("number_of_ranks = %d\n", number_of_ranks);
    printf("rank_size = %u\n", rank_size);
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

    // set up SpecializationInfo, this is used to set constants in shader, for
    VkSpecializationMapEntry specialization_map_entry[3] = {};
    specialization_map_entry[0].constantID = 0;
    specialization_map_entry[0].offset = sizeof(uint32_t) * 0;
    specialization_map_entry[0].size = sizeof(uint32_t);
    specialization_map_entry[1].constantID = 1;
    specialization_map_entry[1].offset = sizeof(uint32_t) * 1;
    specialization_map_entry[1].size = sizeof(uint32_t);
    specialization_map_entry[2].constantID = 2;
    specialization_map_entry[2].offset = sizeof(uint32_t) * 2;
    specialization_map_entry[2].size = sizeof(uint32_t);
    VkSpecializationInfo specialization_info = {};
    specialization_info.mapEntryCount = 3;
    specialization_info.pMapEntries = specialization_map_entry;
    specialization_info.dataSize = sizeof(shader_workgroup_size);
    specialization_info.pData = &shader_workgroup_size;

    // create compute pipeline
    ComputePipeline compute_pipeline;
    UNWRAP_VKRESULT(
        create_compute_pipeline(&compute_device, shader_source.value().data(),
                                shader_source.value().size(), 1, 1, 1,
                                &specialization_info, &compute_pipeline));

    // create descriptor set
    ComputeDescriptorSets compute_descriptor_sets;
    UNWRAP_VKRESULT(create_compute_descriptor_sets(
        &compute_device, &compute_pipeline, &compute_descriptor_sets));

    // update descriptor set
    UNWRAP_VKRESULT(update_compute_descriptor_sets(
        &compute_device, &compute_pipeline, &input_buffer, &output_buffer,
        &uniform_buffer_object, &compute_descriptor_sets));

    // run compute
    std::chrono::system_clock::time_point time_point;
    std::chrono::system_clock::duration duration;
    time_point = std::chrono::high_resolution_clock::now();
    UNWRAP_VKRESULT(run_compute_pipeline_sync(
        &compute_device, &compute_pipeline, &compute_descriptor_sets,
        shader_workgroup_count[0], shader_workgroup_count[1],
        shader_workgroup_count[2]));
    duration = std::chrono::high_resolution_clock::now() - time_point;
    printf("Shader time: %" PRIi64 " ns\n", duration.count());

    // verify output data
    std::vector<int> output_data(rank_size);
    UNWRAP_VKRESULT(read_from_compute_buffer(&compute_device, &output_buffer, 0,
                                             output_buffer.m_size,
                                             output_data.data()));

    time_point = std::chrono::high_resolution_clock::now();
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
            return EXIT_FAILURE;
        }
    }
    duration = std::chrono::high_resolution_clock::now() - time_point;
    printf("Verify time: %" PRIi64 " ns\n", duration.count());
    printf("Success, all values expected!\n");

    // cleanup
    destroy_compute_buffer(&compute_device, &input_buffer);
    destroy_compute_buffer(&compute_device, &output_buffer);
    destroy_compute_buffer(&compute_device, &uniform_buffer_object);
    destroy_compute_pipeline(&compute_device, &compute_pipeline);
    destroy_compute_device(&compute_device);

    return EXIT_SUCCESS;
}
