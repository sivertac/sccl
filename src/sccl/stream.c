
#include "stream.h"
#include "alloc.h"
#include "device.h"
#include "error.h"

static sccl_error_t reset_command_buffer(const sccl_stream_t stream)
{
    CHECK_VKRESULT_RET(vkResetCommandBuffer(stream->command_buffer, 0));

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    CHECK_VKRESULT_RET(
        vkBeginCommandBuffer(stream->command_buffer, &begin_info));

    return sccl_success;
}

static sccl_error_t acquire_next_queue(const sccl_device_t device,
                                       uint32_t *queue_index, VkQueue *queue)
{

    for (uint32_t i = 0; i < device->queue_count; ++i) {
        if (!device->queue_usage[i]) {
            *queue_index = i;
            vkGetDeviceQueue(device->device, device->queue_family_index,
                             *queue_index, queue);
            device->queue_usage[i] = true;
            return sccl_success;
        }
    }

    return sccl_out_of_resources_error;
}

static void free_queue_index(const sccl_device_t device, uint32_t queue_index)
{
    device->queue_usage[queue_index] = false;
}

sccl_error_t sccl_create_stream(const sccl_device_t device,
                                sccl_stream_t *stream)
{

    struct sccl_stream *stream_internal;
    CHECK_SCCL_ERROR_RET(
        sccl_calloc((void **)&stream_internal, 1, sizeof(struct sccl_stream)));

    stream_internal->device = device;

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

    /* select queue */
    CHECK_SCCL_ERROR_RET(acquire_next_queue(
        device, &stream_internal->queue_index, &stream_internal->queue));

    /* reset stream initially so command buffer starts recording */
    CHECK_SCCL_ERROR_RET(reset_command_buffer(stream_internal));

    /* set public handle */
    *stream = (sccl_stream_t)stream_internal;

    return sccl_success;
}

void sccl_destroy_stream(sccl_stream_t stream)
{
    free_queue_index(stream->device, stream->queue_index);
    vkFreeCommandBuffers(stream->device->device, stream->command_pool, 1,
                         &stream->command_buffer);
    vkDestroyCommandPool(stream->device->device, stream->command_pool, NULL);
    sccl_free(stream);
}

sccl_error_t sccl_dispatch_stream(const sccl_stream_t stream)
{

    CHECK_VKRESULT_RET(vkEndCommandBuffer(stream->command_buffer));

    VkSubmitInfo submit_info = {0};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &stream->command_buffer;

    CHECK_VKRESULT_RET(
        vkQueueSubmit(stream->queue, 1, &submit_info, VK_NULL_HANDLE));

    /* Can't reset command buffer here as buffer is in pending state until
     * execution completes */

    return sccl_success;
}

sccl_error_t sccl_join_stream(const sccl_stream_t stream)
{

    /* block until queue is done */
    CHECK_VKRESULT_RET(vkQueueWaitIdle(stream->queue));

    /* Reset command buffer here so we can record for next dispatch */
    CHECK_SCCL_ERROR_RET(reset_command_buffer(stream));

    return sccl_success;
}

// sccl_error_t sccl_copy_buffer(const sccl_stream_t stream, const sccl_buffer_t
// src, size_t src_offset, const sccl_buffer_t dst, size_t dst_offset, size_t
// size) {
// }
