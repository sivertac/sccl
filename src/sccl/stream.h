#pragma once
#ifndef STREAM_HEADER
#define STREAM_HEADER

#include "sccl.h"
#include "vector.h"
#include <vulkan/vulkan.h>

struct sccl_stream {
    sccl_device_t device;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    vector_t descriptor_sets; /* contains descriptor_set_entry_t to free when
                                 command buffer is done executing */
    VkFence fence;
};

sccl_error_t add_descriptor_set_to_stream(const sccl_stream_t stream,
                                          VkDescriptorPool descriptor_pool,
                                          VkDescriptorSet descriptor_set);

#endif // STREAM_HEADER