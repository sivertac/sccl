#pragma once
#ifndef STREAM_HEADER
#define STREAM_HEADER

#include "sccl.h"
#include <vulkan/vulkan.h>

struct sccl_stream {
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkDevice device;
};

#endif // STREAM_HEADER