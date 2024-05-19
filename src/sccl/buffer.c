
#include "buffer.h"
#include "alloc.h"
#include "device.h"
#include "error.h"
#include <stdbool.h>

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

static bool is_buffer_type_storage(sccl_buffer_type_t type)
{
    switch (type) {
    case sccl_buffer_type_host_storage:
    case sccl_buffer_type_device_storage:
    case sccl_buffer_type_shared_storage:
        return true;
    case sccl_buffer_type_host_uniform:
    case sccl_buffer_type_device_uniform:
    case sccl_buffer_type_shared_uniform:
    default:
        return false;
    }
}

static bool is_buffer_type_uniform(sccl_buffer_type_t type)
{
    switch (type) {
    case sccl_buffer_type_host_uniform:
    case sccl_buffer_type_device_uniform:
    case sccl_buffer_type_shared_uniform:
        return true;
    case sccl_buffer_type_host_storage:
    case sccl_buffer_type_device_storage:
    case sccl_buffer_type_shared_storage:
    default:
        return false;
    }
}

static bool is_buffer_type_external(sccl_buffer_type_t type)
{
    switch (type) {
    case sccl_buffer_type_external:
        return true;
    default:
        return false;
    }
}

static sccl_error_t
allocate_buffer_internal(const sccl_device_t device,
                         struct sccl_buffer **buffer_internal,
                         sccl_buffer_type_t type)
{
    CHECK_SCCL_ERROR_RET(
        sccl_calloc((void **)buffer_internal, 1, sizeof(struct sccl_buffer)));

    (*buffer_internal)->type = type;
    (*buffer_internal)->device = device;
    return sccl_success;
}

sccl_error_t sccl_create_buffer(const sccl_device_t device,
                                sccl_buffer_t *buffer, sccl_buffer_type_t type,
                                size_t size)
{
    sccl_error_t error = sccl_success;

    struct sccl_buffer *buffer_internal = NULL;
    CHECK_SCCL_ERROR_GOTO(
        allocate_buffer_internal(device, &buffer_internal, type), error_return,
        error);

    /* determine buffer usage flags */
    VkBufferUsageFlags buffer_usage_flags = 0;
    /* check if buffer is storage or uniform */
    if (is_buffer_type_storage(type)) {
        buffer_usage_flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    } else if (is_buffer_type_uniform(type)) {
        buffer_usage_flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    } else {
        return sccl_invalid_argument;
    }
    buffer_usage_flags |=
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    /* create buffer */
    VkBufferCreateInfo buffer_info = {0};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = buffer_usage_flags;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK_VKRESULT_GOTO(vkCreateBuffer(buffer_internal->device->device,
                                       &buffer_info, NULL,
                                       &buffer_internal->buffer),
                        error_return, error);

    /* get memory requirements */
    VkMemoryRequirements mem_requirements = {0};
    vkGetBufferMemoryRequirements(buffer_internal->device->device,
                                  buffer_internal->buffer, &mem_requirements);

    /* determine memory property flags */
    VkMemoryPropertyFlags memory_property_flags;
    switch (type) {
    case sccl_buffer_type_host_storage:
    case sccl_buffer_type_host_uniform:
        memory_property_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        break;
    case sccl_buffer_type_device_storage:
    case sccl_buffer_type_device_uniform:
        memory_property_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    case sccl_buffer_type_shared_storage:
    case sccl_buffer_type_shared_uniform:
        memory_property_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        /* `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` makes shared buffer turbo slow
         * to read back to CPU */
        break;
    default:
        return sccl_invalid_argument;
    }

    /* find memory type */
    uint32_t memory_type_index;
    CHECK_SCCL_ERROR_GOTO(find_memory_type(device->physical_device,
                                           mem_requirements.memoryTypeBits,
                                           memory_property_flags,
                                           &memory_type_index),
                          error_return, error);

    /* allocate memory */
    VkMemoryAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = memory_type_index;
    CHECK_VKRESULT_GOTO(vkAllocateMemory(buffer_internal->device->device,
                                         &alloc_info, NULL,
                                         &buffer_internal->device_memory),
                        error_return, error);

    /* bind */
    CHECK_VKRESULT_GOTO(vkBindBufferMemory(buffer_internal->device->device,
                                           buffer_internal->buffer,
                                           buffer_internal->device_memory, 0),
                        error_return, error);

    /* set public handle */
    *buffer = (sccl_buffer_t)buffer_internal;

    return sccl_success;

error_return:
    if (buffer_internal != NULL) {
        if (buffer_internal->device_memory != VK_NULL_HANDLE) {
            vkFreeMemory(device->device, buffer_internal->device_memory, NULL);
        }
        if (buffer_internal->buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device->device, buffer_internal->buffer, NULL);
        }
        sccl_free(buffer_internal);
    }

    return error;
}

sccl_error_t sccl_register_host_pointer_buffer(const sccl_device_t device,
                                               sccl_buffer_t *buffer,
                                               void *host_pointer, size_t size)
{
    sccl_error_t error = sccl_success;

    struct sccl_buffer *buffer_internal = NULL;
    CHECK_SCCL_ERROR_GOTO(allocate_buffer_internal(device, &buffer_internal,
                                                   sccl_buffer_type_external),
                          error_return, error);

    /* create buffer for external memory */
    VkExternalMemoryBufferCreateInfo external_memory_buffer_create_info = {0};
    external_memory_buffer_create_info.sType =
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    external_memory_buffer_create_info.handleTypes =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;

    VkBufferCreateInfo buffer_info = {0};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.pNext = &external_memory_buffer_create_info;
    buffer_info.size = size;
    buffer_info.usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    // buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK_VKRESULT_GOTO(vkCreateBuffer(buffer_internal->device->device,
                                       &buffer_info, NULL,
                                       &buffer_internal->buffer),
                        error_return, error);

    /* get memory requirements */
    VkMemoryRequirements mem_requirements = {0};
    vkGetBufferMemoryRequirements(buffer_internal->device->device,
                                  buffer_internal->buffer, &mem_requirements);

    VkMemoryPropertyFlags memory_property_flags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    /* find memory type */
    uint32_t memory_type_index;
    CHECK_SCCL_ERROR_GOTO(find_memory_type(device->physical_device,
                                           mem_requirements.memoryTypeBits,
                                           memory_property_flags,
                                           &memory_type_index),
                          error_return, error);

    /* allocate memory */
    VkImportMemoryHostPointerInfoEXT import_memory_host_pointer_info = {0};
    import_memory_host_pointer_info.sType =
        VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    import_memory_host_pointer_info.handleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    import_memory_host_pointer_info.pHostPointer = host_pointer;

    VkMemoryAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = &import_memory_host_pointer_info;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = memory_type_index;
    CHECK_VKRESULT_GOTO(vkAllocateMemory(buffer_internal->device->device,
                                         &alloc_info, NULL,
                                         &buffer_internal->device_memory),
                        error_return, error);

    /* bind */
    CHECK_VKRESULT_GOTO(vkBindBufferMemory(buffer_internal->device->device,
                                           buffer_internal->buffer,
                                           buffer_internal->device_memory, 0),
                        error_return, error);

    /* set public handle */
    *buffer = (sccl_buffer_t)buffer_internal;

    return sccl_success;

error_return:
    if (buffer_internal != NULL) {
        if (buffer_internal->device_memory != VK_NULL_HANDLE) {
            vkFreeMemory(device->device, buffer_internal->device_memory, NULL);
        }
        if (buffer_internal->buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device->device, buffer_internal->buffer, NULL);
        }
        sccl_free(buffer_internal);
    }

    return error;
}

void sccl_destroy_buffer(sccl_buffer_t buffer)
{
    vkFreeMemory(buffer->device->device, buffer->device_memory, NULL);
    vkDestroyBuffer(buffer->device->device, buffer->buffer, NULL);
    sccl_free(buffer);
}

sccl_buffer_type_t sccl_get_buffer_type(const sccl_buffer_t buffer)
{
    return buffer->type;
}

size_t sccl_get_buffer_min_offset_alignment(const sccl_buffer_t buffer)
{
    sccl_device_properties_t device_properties = {0};
    sccl_get_device_properties(buffer->device, &device_properties);
    if (is_buffer_type_storage(buffer->type)) {
        return device_properties.min_storage_buffer_offset_alignment;
    } else if (is_buffer_type_uniform(buffer->type)) {
        return device_properties.min_uniform_buffer_offset_alignment;
    } else if (is_buffer_type_external(buffer->type)) {
        return device_properties.min_external_buffer_host_pointer_alignment;
    }
    assert(false);
    return 0;
}

sccl_error_t sccl_host_map_buffer(const sccl_buffer_t buffer, void **data,
                                  size_t offset, size_t size)
{
    if (buffer->type == sccl_buffer_type_device) {
        return sccl_invalid_argument;
    }

    CHECK_VKRESULT_RET(vkMapMemory(
        buffer->device->device, buffer->device_memory, offset, size, 0, data));

    return sccl_success;
}

void sccl_host_unmap_buffer(const sccl_buffer_t buffer)
{
    vkUnmapMemory(buffer->device->device, buffer->device_memory);
}
