#pragma once
#ifndef COMPUTE_INTERFACE_HEADER
#define COMPUTE_INTERFACE_HEADER

#include <vulkan/vulkan.h>

struct ComputeDevice {
    VkInstance m_instance;
    VkPhysicalDevice m_physical_device;
    VkDevice m_device;
    bool m_validation_layers_enabled;
    VkDebugUtilsMessengerEXT m_debug_messenger;
};

struct ComputePipeline {
    VkQueue m_queue;
    VkDescriptorSetLayout m_descriptor_set_layout;
    VkDescriptorPool m_descriptor_pool;
    VkShaderModule m_shader_module;
    VkPipelineLayout m_pipeline_layout;
    VkPipeline m_compute_pipeline;
    VkCommandPool m_command_pool;
    VkCommandBuffer m_command_buffer;
};

struct ComputeDescriptorSet {
    VkDescriptorSet m_descriptor_set;
};

struct ComputeBuffer {
    VkBuffer m_buffer;
    VkDeviceMemory m_buffer_memory;
    VkDeviceSize m_size;
};

VkResult create_compute_device(bool enable_validation_layers,
                             ComputeDevice *compute_device);

void destroy_compute_device(ComputeDevice *compute_device);

VkResult create_compute_pipeline(const ComputeDevice *compute_device,
                               const char *shader_source,
                               const size_t shader_source_size,
                               const int num_input_buffers,
                               const int num_output_buffers,
                               ComputePipeline *compute_pipeline);

VkResult update_compute_descriptor_set(
    const ComputeDevice *compute_device, const ComputeBuffer *input_buffers,
    const int num_input_buffers, const ComputeBuffer *output_buffers,
    const int num_output_buffers, ComputeDescriptorSet *compute_descriptor_set);

VkResult
run_compute_pipeline_sync(const ComputeDevice *compute_device,
                       const ComputePipeline *compute_pipeline,
                       const ComputeDescriptorSet *compute_descriptor_set,
                       const uint32_t group_count_x,
                       const uint32_t group_count_y,
                       const uint32_t group_count_z);

void destroy_compute_pipeline(const ComputeDevice *compute_device,
                            ComputePipeline *compute_pipeline);

/**
 * Doesn't need to be destroyed as descriptor sets are maintained by the
 * descriptor set pool.
 */
VkResult
create_compute_descriptor_set(const ComputeDevice *compute_device,
                           const ComputePipeline *compute_pipeline,
                           ComputeDescriptorSet *compute_descriptor_set);

void destroy_compute_descriptor_set(const ComputeDevice *compute_device,
                                 const ComputePipeline *compute_pipeline,
                                 ComputeDescriptorSet *compute_descriptor_set);

VkResult create_compute_buffer(const ComputeDevice *compute_device,
                             VkDeviceSize size, ComputeBuffer *compute_buffer);

void destroy_compute_buffer(const ComputeDevice *compute_device,
                          ComputeBuffer *compute_buffer);

#endif
