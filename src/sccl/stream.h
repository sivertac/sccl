#pragma once
#ifndef STREAM_HEADER
#define STREAM_HEADER

#include "sccl.h"
#include "vector.h"
#include <vulkan/vulkan.h>

typedef enum {
    command_buffer_type_compute,
    command_buffer_type_transfer
} command_buffer_type_t;

typedef struct {
    command_buffer_type_t type;
    VkCommandBuffer command_buffer;
} command_buffer_entry_t;

struct sccl_stream {
    sccl_device_t device;

    VkCommandPool compute_command_pool;
    /* this is set to `compute_command_pool' if `has_seperate_transfer_queue()
     * == false`*/
    VkCommandPool transfer_command_pool;

    /* contains descriptor_set_entry_t to free when command buffer is done
     * executing */
    vector_t descriptor_sets;
    VkFence fence;

    /* contains command buffers of recorded commands */
    vector_t command_buffers;
    /* timeline semaphore for syncing between transfer and compute queue */
    VkSemaphore timeline_semaphore;
};

sccl_error_t add_descriptor_set_to_stream(const sccl_stream_t stream,
                                          VkDescriptorPool descriptor_pool,
                                          VkDescriptorSet descriptor_set);

sccl_error_t
determine_next_command_buffer(const sccl_stream_t stream,
                              command_buffer_type_t next_command_buffer_type,
                              VkCommandBuffer *next_command_buffer_entry);

#endif // STREAM_HEADER