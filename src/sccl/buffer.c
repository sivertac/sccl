
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

static VkBufferUsageFlags get_buffer_usage_flags(sccl_buffer_type_t type)
{
    /* determine buffer usage flags */
    VkBufferUsageFlags buffer_usage_flags = 0;

    if (is_buffer_type_storage(type)) {
        buffer_usage_flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    } else if (is_buffer_type_uniform(type)) {
        buffer_usage_flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    } else {
        assert(false);
    }
    buffer_usage_flags |=
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    return buffer_usage_flags;
}

static VkMemoryPropertyFlags get_memory_property_flags(sccl_buffer_type_t type)
{
    /* determine memory property flags */
    VkMemoryPropertyFlags memory_property_flags;

    if (is_buffer_type_host(type)) {
        memory_property_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    } else if (is_buffer_type_device(type)) {
        memory_property_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    } else if (is_buffer_type_shared(type)) {
        memory_property_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    } else if (is_buffer_type_external(type)) {
        memory_property_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    } else {
        assert(false);
    }

    return memory_property_flags;
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

static sccl_error_t create_buffer_internal(const sccl_device_t device,
                                           struct sccl_buffer **buffer_internal,
                                           sccl_buffer_type_t type, size_t size,
                                           void *buffer_create_info_pnext,
                                           void *memory_allocate_info_pnext)
{
    sccl_error_t error = sccl_success;

    CHECK_SCCL_ERROR_GOTO(
        allocate_buffer_internal(device, buffer_internal, type), error_return,
        error);

    VkBufferUsageFlags buffer_usage_flags = get_buffer_usage_flags(type);
    VkMemoryPropertyFlags memory_property_flags =
        get_memory_property_flags(type);

    /* create buffer */
    VkBufferCreateInfo buffer_create_info = {0};
    buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_create_info.pNext = buffer_create_info_pnext;
    buffer_create_info.size = size;
    buffer_create_info.usage = buffer_usage_flags;
    buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK_VKRESULT_GOTO(vkCreateBuffer((*buffer_internal)->device->device,
                                       &buffer_create_info, NULL,
                                       &(*buffer_internal)->buffer),
                        error_return, error);

    /* get memory requirements */
    VkMemoryRequirements mem_requirements = {0};
    vkGetBufferMemoryRequirements((*buffer_internal)->device->device,
                                  (*buffer_internal)->buffer,
                                  &mem_requirements);

    /* find memory type */
    uint32_t memory_type_index;
    CHECK_SCCL_ERROR_GOTO(find_memory_type(device->physical_device,
                                           mem_requirements.memoryTypeBits,
                                           memory_property_flags,
                                           &memory_type_index),
                          error_return, error);

    /* allocate memory */
    VkMemoryAllocateInfo memory_allocate_info = {0};
    memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memory_allocate_info.pNext = memory_allocate_info_pnext;
    memory_allocate_info.allocationSize = mem_requirements.size;
    memory_allocate_info.memoryTypeIndex = memory_type_index;
    CHECK_VKRESULT_GOTO(vkAllocateMemory((*buffer_internal)->device->device,
                                         &memory_allocate_info, NULL,
                                         &(*buffer_internal)->device_memory),
                        error_return, error);

    /* bind */
    CHECK_VKRESULT_GOTO(vkBindBufferMemory((*buffer_internal)->device->device,
                                           (*buffer_internal)->buffer,
                                           (*buffer_internal)->device_memory,
                                           0),
                        error_return, error);

    return sccl_success;

error_return:
    if (*buffer_internal != NULL) {
        if ((*buffer_internal)->device_memory != VK_NULL_HANDLE) {
            vkFreeMemory(device->device, (*buffer_internal)->device_memory,
                         NULL);
        }
        if ((*buffer_internal)->buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device->device, (*buffer_internal)->buffer, NULL);
        }
        sccl_free(*buffer_internal);
    }

    return error;
}

bool is_buffer_type_storage(sccl_buffer_type_t type)
{
    switch (type) {
    case sccl_buffer_type_host_storage:
    case sccl_buffer_type_device_storage:
    case sccl_buffer_type_shared_storage:
    case sccl_buffer_type_external_host_pointer_storage:
    case sccl_buffer_type_host_dmabuf_storage:
    case sccl_buffer_type_device_dmabuf_storage:
    case sccl_buffer_type_shared_dmabuf_storage:
    case sccl_buffer_type_external_dmabuf_storage:
        return true;
    default:
        return false;
    }
}

bool is_buffer_type_uniform(sccl_buffer_type_t type)
{
    switch (type) {
    case sccl_buffer_type_host_uniform:
    case sccl_buffer_type_device_uniform:
    case sccl_buffer_type_shared_uniform:
    case sccl_buffer_type_external_host_pointer_uniform:
    case sccl_buffer_type_host_dmabuf_uniform:
    case sccl_buffer_type_device_dmabuf_uniform:
    case sccl_buffer_type_shared_dmabuf_uniform:
    case sccl_buffer_type_external_dmabuf_uniform:
        return true;
    default:
        return false;
    }
}

bool is_buffer_type_host(sccl_buffer_type_t type)
{
    switch (type) {
    case sccl_buffer_type_host_storage:
    case sccl_buffer_type_host_uniform:
    case sccl_buffer_type_host_dmabuf_storage:
    case sccl_buffer_type_host_dmabuf_uniform:
        return true;
    default:
        return false;
    }
}

bool is_buffer_type_device(sccl_buffer_type_t type)
{
    switch (type) {
    case sccl_buffer_type_device_storage:
    case sccl_buffer_type_device_uniform:
    case sccl_buffer_type_device_dmabuf_storage:
    case sccl_buffer_type_device_dmabuf_uniform:
        return true;
    default:
        return false;
    }
}

bool is_buffer_type_shared(sccl_buffer_type_t type)
{
    switch (type) {
    case sccl_buffer_type_shared_storage:
    case sccl_buffer_type_shared_uniform:
    case sccl_buffer_type_shared_dmabuf_storage:
    case sccl_buffer_type_shared_dmabuf_uniform:
        return true;
    default:
        return false;
    }
}

bool is_buffer_type_external(sccl_buffer_type_t type)
{
    switch (type) {
    case sccl_buffer_type_external_host_pointer_storage:
    case sccl_buffer_type_external_host_pointer_uniform:
    case sccl_buffer_type_external_dmabuf_storage:
    case sccl_buffer_type_external_dmabuf_uniform:
        return true;
    default:
        return false;
    }
}

bool is_buffer_type_regular(sccl_buffer_type_t type)
{
    switch (type) {
    case sccl_buffer_type_host_storage:
    case sccl_buffer_type_device_storage:
    case sccl_buffer_type_shared_storage:
    case sccl_buffer_type_host_uniform:
    case sccl_buffer_type_device_uniform:
    case sccl_buffer_type_shared_uniform:
        return true;
    default:
        return false;
    }
}

bool is_buffer_type_host_pointer(sccl_buffer_type_t type)
{
    switch (type) {
    case sccl_buffer_type_external_host_pointer_storage:
    case sccl_buffer_type_external_host_pointer_uniform:
        return true;
    default:
        return false;
    }
}

bool is_buffer_type_dmabuf(sccl_buffer_type_t type)
{
    switch (type) {
    case sccl_buffer_type_host_dmabuf_storage:
    case sccl_buffer_type_device_dmabuf_storage:
    case sccl_buffer_type_shared_dmabuf_storage:
    case sccl_buffer_type_host_dmabuf_uniform:
    case sccl_buffer_type_device_dmabuf_uniform:
    case sccl_buffer_type_shared_dmabuf_uniform:
    case sccl_buffer_type_external_dmabuf_storage:
    case sccl_buffer_type_external_dmabuf_uniform:
        return true;
    default:
        return false;
    }
}

sccl_error_t sccl_create_buffer(const sccl_device_t device,
                                sccl_buffer_t *buffer, sccl_buffer_type_t type,
                                size_t size)
{

    struct sccl_buffer *buffer_internal = NULL;

    /* check buffer type is regular */
    if (!is_buffer_type_regular(type)) {
        return sccl_invalid_argument;
    }

    CHECK_SCCL_ERROR_RET(create_buffer_internal(device, &buffer_internal, type,
                                                size, NULL, NULL));

    /* set public handle */
    *buffer = (sccl_buffer_t)buffer_internal;

    return sccl_success;
}

sccl_error_t sccl_create_external_host_pointer_buffer(
    const sccl_device_t device, sccl_buffer_t *buffer, sccl_buffer_type_t type,
    void *host_pointer, size_t size)
{
    struct sccl_buffer *buffer_internal = NULL;

    /* check if supported */
    if (!device->host_pointer_supported) {
        return sccl_unsupported_error;
    }

    /* check type is host pointer */
    if (!is_buffer_type_host_pointer(type)) {
        return sccl_invalid_argument;
    }

    /* create buffer for external memory */
    VkExternalMemoryBufferCreateInfo external_memory_buffer_create_info = {0};
    external_memory_buffer_create_info.sType =
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    external_memory_buffer_create_info.handleTypes =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;

    /* use host pointer ext */
    VkImportMemoryHostPointerInfoEXT import_memory_host_pointer_info = {0};
    import_memory_host_pointer_info.sType =
        VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    import_memory_host_pointer_info.handleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    import_memory_host_pointer_info.pHostPointer = host_pointer;

    CHECK_SCCL_ERROR_RET(create_buffer_internal(
        device, &buffer_internal, type, size,
        &external_memory_buffer_create_info, &import_memory_host_pointer_info));

    /* set public handle */
    *buffer = (sccl_buffer_t)buffer_internal;

    return sccl_success;
}

sccl_error_t sccl_create_dmabuf_buffer(const sccl_device_t device,
                                       sccl_buffer_t *buffer,
                                       sccl_buffer_type_t type, size_t size)
{
    struct sccl_buffer *buffer_internal = NULL;

    /* check if supported */
    if (!device->dmabuf_buffer_supported) {
        return sccl_unsupported_error;
    }

    /* check if type is dmabuf */
    if (!is_buffer_type_dmabuf(type)) {
        return sccl_invalid_argument;
    }

    VkExternalMemoryBufferCreateInfo external_memory_buffer_create_info = {0};
    external_memory_buffer_create_info.sType =
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    external_memory_buffer_create_info.handleTypes =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

    VkExportMemoryAllocateInfo export_memory_allocate_info = {0};
    export_memory_allocate_info.sType =
        VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    export_memory_allocate_info.handleTypes =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

    CHECK_SCCL_ERROR_RET(create_buffer_internal(
        device, &buffer_internal, type, size,
        &external_memory_buffer_create_info, &export_memory_allocate_info));

    /* set public handle */
    *buffer = (sccl_buffer_t)buffer_internal;

    return sccl_success;
}

sccl_error_t sccl_export_dmabuf_buffer(const sccl_buffer_t buffer, int *out_fd)
{
    /* check if supported */
    if (!buffer->device->dmabuf_buffer_supported) {
        return sccl_unsupported_error;
    }

    /* check if type is dmabuf */
    if (!is_buffer_type_dmabuf(buffer->type)) {
        return sccl_invalid_argument;
    }

    VkMemoryGetFdInfoKHR memory_get_fd_info = {0};
    memory_get_fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    memory_get_fd_info.memory = buffer->device_memory;
    memory_get_fd_info.handleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    CHECK_VKRESULT_RET(buffer->device->pfn_vk_get_memory_fd_khr(
        buffer->device->device, &memory_get_fd_info, out_fd));

    return sccl_success;
}

sccl_error_t sccl_import_dmabuf_buffer(const sccl_device_t device,
                                       sccl_buffer_t *buffer, int in_fd,
                                       sccl_buffer_type_t type, size_t size)
{
    struct sccl_buffer *buffer_internal = NULL;

    /* check if supported */
    if (!device->dmabuf_buffer_supported) {
        return sccl_unsupported_error;
    }

    /* check if type is dmabuf */
    if (!is_buffer_type_dmabuf(type)) {
        return sccl_invalid_argument;
    }

    /* query memory import info */
    VkMemoryFdPropertiesKHR memory_fd_properties = {0};
    memory_fd_properties.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR;
    CHECK_VKRESULT_RET(device->pfn_vk_get_memory_fd_properties_khr(
        device->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, in_fd,
        &memory_fd_properties));
    // printf("memory_fd_properties.memoryTypeBits = %d\n",
    // memory_fd_properties.memoryTypeBits);

    /* create buffer for external memory */
    VkExternalMemoryBufferCreateInfo external_memory_buffer_create_info = {0};
    external_memory_buffer_create_info.sType =
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    external_memory_buffer_create_info.handleTypes =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

    /* use VK_KHR_external_memory_fd */
    VkImportMemoryFdInfoKHR import_memory_fd_info = {0};
    import_memory_fd_info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
    import_memory_fd_info.handleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    import_memory_fd_info.fd = in_fd;

    CHECK_SCCL_ERROR_RET(create_buffer_internal(
        device, &buffer_internal, type, size,
        &external_memory_buffer_create_info, &import_memory_fd_info));

    /* set public handle */
    *buffer = (sccl_buffer_t)buffer_internal;

    return sccl_success;
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
