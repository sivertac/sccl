#pragma once
#ifndef BUFFER_HEADER
#define BUFFER_HEADER

#include "sccl.h"
#include <vulkan/vulkan.h>

struct sccl_buffer {
    sccl_device_t device;
    sccl_buffer_type_t type;
    VkBuffer buffer;
    VkDeviceMemory device_memory;
};

#endif // BUFFER_HEADER