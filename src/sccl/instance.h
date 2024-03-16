#pragma once
#ifndef INSTANCE_HEADER
#define INSTANCE_HEADER

#include <vulkan/vulkan.h>

struct sccl_instance {
    VkInstance instance;
    uint32_t physical_device_count;
    VkPhysicalDevice *physical_devices;
};

#endif // INSTANCE_HEADER