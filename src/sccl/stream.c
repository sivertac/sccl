
#include "stream.h"
#include "alloc.h"
#include "buffer.h"
#include "device.h"
#include "error.h"
#include <stdbool.h>

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

    /* reset stream initially so command buffer starts recording */
    CHECK_SCCL_ERROR_RET(reset_command_buffer(stream_internal));

    /* create fence */
    VkFenceCreateInfo fence_create_info = {0};
    fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    CHECK_VKRESULT_RET(vkCreateFence(device->device, &fence_create_info, NULL,
                                     &stream_internal->fence));

    /* set public handle */
    *stream = (sccl_stream_t)stream_internal;

    return sccl_success;
}

void sccl_destroy_stream(sccl_stream_t stream)
{
    vkDestroyFence(stream->device->device, stream->fence, NULL);
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

    VkQueue queue;
    vkGetDeviceQueue(stream->device->device, stream->device->queue_family_index,
                     SCCL_QUEUE_INDEX, &queue);

    CHECK_VKRESULT_RET(vkQueueSubmit(queue, 1, &submit_info, stream->fence));

    /* Can't reset command buffer here as buffer is in pending state until
     * execution completes */

    return sccl_success;
}

sccl_error_t sccl_join_stream(const sccl_stream_t stream)
{

    /* block until command buffer is done
     * 1 fence per stream
     * 1 minute timeout */
    VkResult res = VK_SUCCESS;
    do {
        res = vkWaitForFences(stream->device->device, 1, &stream->fence, false,
                              60000000000);
    } while (res == VK_TIMEOUT);
    CHECK_VKRESULT_RET(res);

    /* reset fence */
    CHECK_VKRESULT_RET(
        vkResetFences(stream->device->device, 1, &stream->fence));

    /* reset command buffer here so we can record for next dispatch */
    CHECK_SCCL_ERROR_RET(reset_command_buffer(stream));

    return sccl_success;
}

sccl_error_t sccl_copy_buffer(const sccl_stream_t stream,
                              const sccl_buffer_t src, size_t src_offset,
                              const sccl_buffer_t dst, size_t dst_offset,
                              size_t size)
{
    VkBufferCopy buffer_copy = {0};
    buffer_copy.srcOffset = src_offset;
    buffer_copy.dstOffset = dst_offset;
    buffer_copy.size = size;
    vkCmdCopyBuffer(stream->command_buffer, src->buffer, dst->buffer, 1,
                    &buffer_copy);

    /* create barrier so next command will wait until this is finished */
    VkMemoryBarrier memory_barrier = {};
    memory_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memory_barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    memory_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    vkCmdPipelineBarrier(stream->command_buffer,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                         &memory_barrier, 0, NULL, 0, NULL);

    return sccl_success;
}
