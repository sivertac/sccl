#pragma once
#ifndef DEVICE_HEADER
#define DEVICE_HEADER

#include <stdbool.h>
#include <vulkan/vulkan.h>

struct sccl_device {
    VkDevice device;
    VkPhysicalDevice physical_device;
    uint32_t queue_family_index;
    uint32_t queue_count;
    bool *queue_usage;
};

#endif // DEVICE_HEADER