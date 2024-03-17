#pragma once
#ifndef DEVICE_HEADER
#define DEVICE_HEADER

#include <vulkan/vulkan.h>

struct sccl_device {
    VkDevice device;
    VkPhysicalDevice physical_device;
    uint32_t queue_family_index;
};

#endif // DEVICE_HEADER