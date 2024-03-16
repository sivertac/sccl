#pragma once
#ifndef DEVICE_HEADER
#define DEVICE_HEADER

#include <vulkan/vulkan.h>

struct sccl_device {
    VkDevice device;
};

#endif // DEVICE_HEADER