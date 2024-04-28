#pragma once
#ifndef DEVICE_HEADER
#define DEVICE_HEADER

#include "sccl.h"
#include <stdbool.h>
#include <vulkan/vulkan.h>

/* always select queue at index 0 */
#define SCCL_QUEUE_INDEX 0

struct sccl_device {
    VkPhysicalDevice physical_device;
    VkDevice device;
    uint32_t compute_queue_family_index;
    uint32_t transfer_queue_family_index;
};

bool has_seperate_transfer_queue(const sccl_device_t device);

#endif // DEVICE_HEADER