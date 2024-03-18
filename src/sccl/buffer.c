
#include "buffer.h"
#include "alloc.h"
#include "device.h"
#include "error.h"

static sccl_error_t find_memory_type(VkPhysicalDevice physical_device,
                                     uint32_t type_filter,
                                     VkMemoryPropertyFlags properties,
                                     uint32_t *output_index)
{
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & properties) ==
                properties) {
            *output_index = i;
            return sccl_success;
        }
    }

    return sccl_unsupported_error;
}

sccl_error_t sccl_create_buffer(const sccl_device_t device,
                                sccl_buffer_t *buffer, sccl_buffer_type_t type,
                                size_t size)
{

    struct sccl_buffer *buffer_internal;
    CHECK_SCCL_ERROR_RET(
        sccl_calloc((void **)&buffer_internal, 1, sizeof(struct sccl_buffer)));

    buffer_internal->type = type;
    buffer_internal->device = device->device;

    /* determine buffer usage flags */
    VkBufferUsageFlags buffer_usage_flags;
    buffer_usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    /* create buffer */
    VkBufferCreateInfo buffer_info = {0};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = buffer_usage_flags;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK_VKRESULT_RET(vkCreateBuffer(buffer_internal->device, &buffer_info,
                                      NULL, &buffer_internal->buffer));

    /* get memory requirements */
    VkMemoryRequirements mem_requirements = {0};
    vkGetBufferMemoryRequirements(buffer_internal->device,
                                  buffer_internal->buffer, &mem_requirements);

    /* determine memory property flags */
    VkMemoryPropertyFlags memory_property_flags;
    if (type == sccl_buffer_type_host) {
        memory_property_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    } else if (type == sccl_buffer_type_device) {
        memory_property_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    } else if (type == sccl_buffer_type_shared) {
        memory_property_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    } else {
        return sccl_invalid_argument;
    }

    /* find memory type */
    uint32_t memory_type_index;
    CHECK_SCCL_ERROR_RET(find_memory_type(
        device->physical_device, mem_requirements.memoryTypeBits,
        memory_property_flags, &memory_type_index));

    /* allocate memory */
    VkMemoryAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = memory_type_index;
    CHECK_VKRESULT_RET(vkAllocateMemory(device->device, &alloc_info, NULL,
                                        &buffer_internal->device_memory));

    /* bind */
    CHECK_VKRESULT_RET(vkBindBufferMemory(buffer_internal->device,
                                          buffer_internal->buffer,
                                          buffer_internal->device_memory, 0));

    /* set public handle */
    *buffer = (sccl_buffer_t)buffer_internal;

    return sccl_success;
}

void sccl_destroy_buffer(sccl_buffer_t buffer)
{
    vkFreeMemory(buffer->device, buffer->device_memory, NULL);
    vkDestroyBuffer(buffer->device, buffer->buffer, NULL);
    sccl_free(buffer);
}

sccl_error_t sccl_host_map_buffer(const sccl_buffer_t buffer, void **data,
                                  size_t offset, size_t size)
{
    if (buffer->type == sccl_buffer_type_device) {
        return sccl_invalid_argument;
    }

    CHECK_VKRESULT_RET(vkMapMemory(buffer->device, buffer->device_memory,
                                   offset, size, 0, data));

    return sccl_success;
}

void sccl_host_unmap_buffer(const sccl_buffer_t buffer)
{
    vkUnmapMemory(buffer->device, buffer->device_memory);
}
