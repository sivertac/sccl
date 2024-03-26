#pragma once
#ifndef STREAM_HEADER
#define STREAM_HEADER

#include "sccl.h"
#include <vulkan/vulkan.h>

struct sccl_stream {
    sccl_device_t device;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
};

#endif // STREAM_HEADER