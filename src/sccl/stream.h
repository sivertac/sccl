#pragma once
#ifndef STREAM_HEADER
#define STREAM_HEADER

#include "sccl.h"
#include <vulkan/vulkan.h>

struct sccl_stream {
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    uint32_t queue_index;
    VkQueue queue;
    sccl_device_t device;
};

#endif // STREAM_HEADER