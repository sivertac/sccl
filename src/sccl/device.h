#pragma once
#ifndef DEVICE_HEADER
#define DEVICE_HEADER

#include <vulkan/vulkan.h>

/* always select queue at index 0 */
#define SCCL_QUEUE_INDEX 0

struct sccl_device {
    VkPhysicalDevice physical_device;
    VkDevice device;
    uint32_t queue_family_index;
};

#endif // DEVICE_HEADER