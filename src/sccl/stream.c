
#include "stream.h"
#include "alloc.h"
#include "device.h"
#include "error.h"

sccl_error_t sccl_create_stream(const sccl_device_t device,
                                sccl_stream_t *stream)
{

    struct sccl_stream *stream_internal;
    CHECK_SCCL_ERROR_RET(
        sccl_calloc((void **)&stream_internal, 1, sizeof(struct sccl_stream)));

    stream_internal->device = device->device;

    VkCommandPoolCreateInfo command_pool_create_info = {0};
    command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_create_info.queueFamilyIndex = device->queue_family_index;
    command_pool_create_info.flags =
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    CHECK_VKRESULT_RET(vkCreateCommandPool(device->device,
                                           &command_pool_create_info, NULL,
                                           &stream_internal->command_pool));

    VkCommandBufferAllocateInfo command_buffer_allocate_info = {0};
    command_buffer_allocate_info.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_allocate_info.commandPool = stream_internal->command_pool;
    command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_allocate_info.commandBufferCount = 1;

    CHECK_VKRESULT_RET(
        vkAllocateCommandBuffers(device->device, &command_buffer_allocate_info,
                                 &stream_internal->command_buffer));

    /* set public handle */
    *stream = (sccl_stream_t)stream_internal;

    return sccl_success;
}

void sccl_destroy_stream(sccl_stream_t stream)
{
    vkFreeCommandBuffers(stream->device, stream->command_pool, 1,
                         &stream->command_buffer);
    vkDestroyCommandPool(stream->device, stream->command_pool, NULL);
    sccl_free(stream);
}
