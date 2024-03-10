#pragma once
#ifndef COMPUTE_INTERFACE_HEADER
#define COMPUTE_INTERFACE_HEADER

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <vulkan/vulkan.h>

typedef struct ComputeDevice {
    VkInstance m_instance;
    VkPhysicalDevice m_physical_device;
    VkDevice m_device;
    bool m_validation_layers_enabled;
    VkDebugUtilsMessengerEXT m_debug_messenger;
} ComputeDevice;

typedef struct ComputePipeline {
    VkQueue m_queue;
    VkDescriptorSetLayout m_input_descriptor_set_layout;
    uint32_t m_num_input_bindings;
    VkDescriptorSetLayout m_output_descriptor_set_layout;
    uint32_t m_num_output_bindings;
    VkDescriptorSetLayout m_uniform_descriptor_set_layout;
    uint32_t m_num_uniform_bindings;
    VkDescriptorPool m_descriptor_pool;
    VkShaderModule m_shader_module;
    VkPipelineLayout m_pipeline_layout;
    VkPipeline m_compute_pipeline;
    VkCommandPool m_command_pool;
    VkCommandBuffer m_command_buffer;
} ComputePipeline;

typedef struct ComputeDescriptorSets {
    VkDescriptorSet m_input_descriptor_set;
    VkDescriptorSet m_output_descriptor_set;
    VkDescriptorSet m_uniform_descriptor_set;
} ComputeDescriptorSets;

typedef struct ComputeBuffer {
    VkBuffer m_buffer;
    VkDeviceMemory m_buffer_memory;
    VkDeviceSize m_size;
} ComputeBuffer;

/**
 * @brief Init vulkan structures and store in ComputeDevice.
 * @param enable_validation_layers Enable validation layers.
 * @param compute_device
 * @return `VK_SUCCESS` if ok.
 */
VkResult create_compute_device(bool enable_validation_layers,
                               ComputeDevice *compute_device);

void destroy_compute_device(ComputeDevice *compute_device);

VkResult create_compute_pipeline(const ComputeDevice *compute_device,
                                 const char *shader_source,
                                 const size_t shader_source_size,
                                 const int num_input_buffers,
                                 const int num_output_buffers,
                                 const int num_uniform_buffers,
                                 ComputePipeline *compute_pipeline);

VkResult update_compute_descriptor_sets(
    const ComputeDevice *compute_device,
    const ComputePipeline *compute_pipeline, const ComputeBuffer *input_buffers,
    const ComputeBuffer *output_buffers, const ComputeBuffer *uniform_buffers,
    ComputeDescriptorSets *compute_descriptor_sets);

VkResult
run_compute_pipeline_sync(const ComputeDevice *compute_device,
                          const ComputePipeline *compute_pipeline,
                          const ComputeDescriptorSets *compute_descriptor_sets,
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
create_compute_descriptor_sets(const ComputeDevice *compute_device,
                               const ComputePipeline *compute_pipeline,
                               ComputeDescriptorSets *compute_descriptor_sets);

VkResult create_compute_buffer(const ComputeDevice *compute_device,
                               VkDeviceSize size, ComputeBuffer *compute_buffer,
                               VkBufferUsageFlagBits usage);

void destroy_compute_buffer(const ComputeDevice *compute_device,
                            ComputeBuffer *compute_buffer);

VkResult write_to_compute_buffer(const ComputeDevice *compute_device,
                                 const ComputeBuffer *compute_buffer,
                                 VkDeviceSize offset, VkDeviceSize size,
                                 const void *data);

#ifdef __cplusplus
}
#endif

#endif
