#pragma once
#ifndef BUFFER_HEADER
#define BUFFER_HEADER

#include "sccl.h"
#include <stdbool.h>
#include <vulkan/vulkan.h>

struct sccl_buffer {
    sccl_device_t device;
    sccl_buffer_type_t type;
    VkBuffer buffer;
    VkDeviceMemory device_memory;
};

bool is_buffer_type_storage(sccl_buffer_type_t type);

bool is_buffer_type_uniform(sccl_buffer_type_t type);

bool is_buffer_type_host(sccl_buffer_type_t type);

bool is_buffer_type_device(sccl_buffer_type_t type);

bool is_buffer_type_shared(sccl_buffer_type_t type);

bool is_buffer_type_external(sccl_buffer_type_t type);

bool is_buffer_type_regular(sccl_buffer_type_t type);

bool is_buffer_type_host_pointer(sccl_buffer_type_t type);

bool is_buffer_type_dmabuf(sccl_buffer_type_t type);

#endif // BUFFER_HEADER