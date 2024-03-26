#pragma once
#ifndef INSTANCE_HEADER
#define INSTANCE_HEADER

#include <vulkan/vulkan.h>

struct sccl_instance {
    VkInstance instance;
    VkPhysicalDevice *physical_devices;
    VkDebugUtilsMessengerEXT debug_messenger;
    uint32_t physical_device_count;
};

#endif // INSTANCE_HEADER