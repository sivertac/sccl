
#include "stream.h"
#include "alloc.h"
#include "buffer.h"
#include "device.h"
#include "error.h"
#include <stdbool.h>

typedef struct {
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;
} descriptor_set_entry_t;

// static sccl_error_t reset_command_buffer(const sccl_stream_t stream)
//{
//     CHECK_VKRESULT_RET(vkResetCommandBuffer(stream->command_buffer, 0));
//
//     VkCommandBufferBeginInfo begin_info = {0};
//     begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
//     begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
//
//     CHECK_VKRESULT_RET(
//         vkBeginCommandBuffer(stream->command_buffer, &begin_info));
//
//     return sccl_success;
// }

/**
 * Get current command buffer.
 * Returns NULL if empty.
 */
static command_buffer_entry_t *
get_current_command_buffer(vector_t *command_buffers)
{
    if (vector_get_size(command_buffers) <= 0) {
        return NULL;
    }
    return vector_get_last_element(command_buffers);
}

static VkCommandPool get_command_pool(const sccl_stream_t stream,
                                      command_buffer_type_t type)
{
    switch (type) {
    case command_buffer_type_compute:
        return stream->compute_command_pool;
        break;
    case command_buffer_type_transfer:
        return stream->transfer_command_pool;
        break;
    default:
        assert(false);
    }
    return VK_NULL_HANDLE;
}

/**
 * Allocate and begin command buffer.
 */
static sccl_error_t allocate_command_buffer(VkDevice device,
                                            VkCommandPool command_pool,
                                            VkCommandBuffer *command_buffer)
{
    VkCommandBufferAllocateInfo command_buffer_allocate_info = {0};
    command_buffer_allocate_info.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_allocate_info.commandPool = command_pool;
    command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_allocate_info.commandBufferCount = 1;

    CHECK_VKRESULT_RET(vkAllocateCommandBuffers(
        device, &command_buffer_allocate_info, command_buffer));

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VKRESULT_RET(vkBeginCommandBuffer(*command_buffer, &begin_info));

    return sccl_success;
}

static void free_command_buffers(sccl_stream_t stream)
{
    for (size_t i = 0; i < vector_get_size(&stream->command_buffers); ++i) {
        command_buffer_entry_t *e =
            vector_get_element(&stream->command_buffers, i);
        VkCommandPool command_pool = get_command_pool(stream, e->type);
        vkFreeCommandBuffers(stream->device->device, command_pool, 1,
                             &e->command_buffer);
    }
    vector_clear(&stream->command_buffers);
}

static uint32_t get_queue_family_index(const sccl_device_t device,
                                       command_buffer_type_t type)
{
    switch (type) {
    case command_buffer_type_compute:
        return device->compute_queue_family_index;
    case command_buffer_type_transfer:
        return device->transfer_queue_family_index;
    default:
        assert(false);
    }
    return ~0;
}

static sccl_error_t free_descriptor_sets(VkDevice device,
                                         vector_t *descriptor_sets)
{
    for (size_t i = 0; i < vector_get_size(descriptor_sets); ++i) {
        descriptor_set_entry_t *e = vector_get_element(descriptor_sets, i);
        CHECK_VKRESULT_RET(vkFreeDescriptorSets(device, e->descriptor_pool, 1,
                                                &e->descriptor_set));
    }
    vector_clear(descriptor_sets);
    return sccl_success;
}

static sccl_error_t wait_streams(const sccl_device_t device,
                                 const sccl_stream_t *streams,
                                 size_t streams_count,
                                 uint8_t *completed_streams, bool wait_all)
{
    sccl_error_t error = sccl_success;

    VkFence *fences = NULL;

    CHECK_SCCL_ERROR_GOTO(
        sccl_calloc((void **)&fences, streams_count, sizeof(VkFence)),
        error_return, error);
    /* copy fences */
    for (size_t i = 0; i < streams_count; ++i) {
        fences[i] = streams[i]->fence;
    }

    /* block until command buffer is done
     * 1 fence per stream
     * 1 minute timeout */
    VkResult res = VK_SUCCESS;
    do {
        /* TODO: use VK_KHR_external_fence_fd
         * (https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_fence_fd.html)
         * instead */
        res = vkWaitForFences(device->device, streams_count, fences, wait_all,
                              60000000000);
    } while (res == VK_TIMEOUT);
    CHECK_VKRESULT_GOTO(res, error_return, error);

    /* check signaled streams*/
    for (size_t i = 0; i < streams_count; ++i) {
        res = vkGetFenceStatus(device->device, fences[i]);
        if (res == VK_SUCCESS) {
            if (completed_streams != NULL) {
                completed_streams[i] = 1;
            }
        } else if (res == VK_NOT_READY) {
            if (completed_streams != NULL) {
                completed_streams[i] = 0;
            }
        } else {
            CHECK_VKRESULT_GOTO(res, error_return, error);
        }
    }

    sccl_free(fences);
    return sccl_success;
error_return:
    if (fences != NULL && completed_streams != NULL) {
        sccl_free(fences);
    }
    return error;
}

static sccl_error_t create_timeline_semaphore(const VkDevice device,
                                              VkSemaphore *timeline_semaphore)
{
    VkSemaphoreTypeCreateInfo timeline_semapore_create_info;
    timeline_semapore_create_info.sType =
        VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timeline_semapore_create_info.pNext = NULL;
    timeline_semapore_create_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timeline_semapore_create_info.initialValue = 0;
    VkSemaphoreCreateInfo semaphore_create_info;
    semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphore_create_info.pNext = &timeline_semapore_create_info;
    semaphore_create_info.flags = 0;
    CHECK_VKRESULT_RET(vkCreateSemaphore(device, &semaphore_create_info, NULL,
                                         timeline_semaphore));

    return sccl_success;
}

sccl_error_t add_descriptor_set_to_stream(const sccl_stream_t stream,
                                          VkDescriptorPool descriptor_pool,
                                          VkDescriptorSet descriptor_set)
{
    descriptor_set_entry_t entry = {0};
    entry.descriptor_pool = descriptor_pool;
    entry.descriptor_set = descriptor_set;
    return vector_add_element(&stream->descriptor_sets, &entry);
}

sccl_error_t
determine_next_command_buffer(const sccl_stream_t stream,
                              command_buffer_type_t next_command_buffer_type,
                              VkCommandBuffer *next_command_buffer_entry)
{
    command_buffer_entry_t *current_command_buffer_entry =
        get_current_command_buffer(&stream->command_buffers);
    if (current_command_buffer_entry == NULL ||
        current_command_buffer_entry->type != next_command_buffer_type) {
        if (current_command_buffer_entry != NULL) {
            CHECK_VKRESULT_RET(vkEndCommandBuffer(
                current_command_buffer_entry->command_buffer));
        }

        /* alloc new command buffer */
        command_buffer_entry_t e = {0};
        e.type = next_command_buffer_type;
        CHECK_SCCL_ERROR_RET(allocate_command_buffer(
            stream->device->device, get_command_pool(stream, e.type),
            &e.command_buffer));

        /* append to command buffers */
        CHECK_SCCL_ERROR_RET(vector_add_element(&stream->command_buffers, &e));
    }

    current_command_buffer_entry =
        get_current_command_buffer(&stream->command_buffers);
    assert(current_command_buffer_entry != NULL);
    *next_command_buffer_entry = current_command_buffer_entry->command_buffer;

    return sccl_success;
}

sccl_error_t sccl_create_stream(const sccl_device_t device,
                                sccl_stream_t *stream)
{
    sccl_error_t error = sccl_success;

    struct sccl_stream *stream_internal = NULL;
    CHECK_SCCL_ERROR_GOTO(
        sccl_calloc((void **)&stream_internal, 1, sizeof(struct sccl_stream)),
        error_return, error);

    stream_internal->device = device;

    /* create compute command pool */
    {
        VkCommandPoolCreateInfo command_pool_create_info = {0};
        command_pool_create_info.sType =
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        command_pool_create_info.queueFamilyIndex =
            device->compute_queue_family_index;
        /* VK_COMMAND_POOL_CREATE_TRANSIENT_BIT because these command buffers
         * will be used once */
        command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        CHECK_VKRESULT_GOTO(
            vkCreateCommandPool(device->device, &command_pool_create_info, NULL,
                                &stream_internal->compute_command_pool),
            error_return, error);
    }

    if (has_seperate_transfer_queue(device)) {
        /* create transfer command pool */
        VkCommandPoolCreateInfo command_pool_create_info = {0};
        command_pool_create_info.sType =
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        command_pool_create_info.queueFamilyIndex =
            device->transfer_queue_family_index;
        /* VK_COMMAND_POOL_CREATE_TRANSIENT_BIT because these command buffers
         * will be used once */
        command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        CHECK_VKRESULT_GOTO(
            vkCreateCommandPool(device->device, &command_pool_create_info, NULL,
                                &stream_internal->transfer_command_pool),
            error_return, error);
    } else {
        /* use compute pool as transfer */
        stream_internal->transfer_command_pool =
            stream_internal->compute_command_pool;
    }

    // VkCommandBufferAllocateInfo command_buffer_allocate_info = {0};
    // command_buffer_allocate_info.sType =
    //     VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    // command_buffer_allocate_info.commandPool = stream_internal->command_pool;
    // command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    // command_buffer_allocate_info.commandBufferCount = 1;
    //
    // CHECK_VKRESULT_GOTO(
    //     vkAllocateCommandBuffers(device->device,
    //     &command_buffer_allocate_info,
    //                              &stream_internal->command_buffer),
    //     error_return, error);

    /* reset stream initially so command buffer starts recording */
    // CHECK_SCCL_ERROR_GOTO(reset_command_buffer(stream_internal),
    // error_return,
    //                       error);

    /* create descriptor set container */
    CHECK_SCCL_ERROR_GOTO(vector_init(&stream_internal->descriptor_sets,
                                      sizeof(descriptor_set_entry_t)),
                          error_return, error);

    /* create fence */
    VkFenceCreateInfo fence_create_info = {0};
    fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    CHECK_VKRESULT_GOTO(vkCreateFence(device->device, &fence_create_info, NULL,
                                      &stream_internal->fence),
                        error_return, error);

    /* create command_buffer container */
    CHECK_SCCL_ERROR_GOTO(vector_init(&stream_internal->command_buffers,
                                      sizeof(command_buffer_entry_t)),
                          error_return, error);

    /* create timeline semaphore */
    CHECK_SCCL_ERROR_GOTO(
        create_timeline_semaphore(device->device,
                                  &stream_internal->timeline_semaphore),
        error_return, error);

    /* set public handle */
    *stream = (sccl_stream_t)stream_internal;

    return sccl_success;

error_return:
    if (stream_internal != NULL) {
        if (stream_internal->timeline_semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(device->device,
                               stream_internal->timeline_semaphore, NULL);
        }
        if (vector_is_initilized(&stream_internal->command_buffers)) {
            /* command buffers should be empty at this stage */
            /* free_command_buffers(device->device,
             * stream_internal->compute_command_pool,
             * stream_internal->transfer_command_pool,
             * &stream_internal->command_buffers); */
            vector_destroy(&stream_internal->command_buffers);
        }
        if (stream_internal->fence != VK_NULL_HANDLE) {
            vkDestroyFence(device->device, stream_internal->fence, NULL);
        }
        if (vector_is_initilized(&stream_internal->descriptor_sets)) {
            free_descriptor_sets(device->device,
                                 &stream_internal->descriptor_sets);
            vector_destroy(&stream_internal->descriptor_sets);
        }
        // if (stream_internal->command_pool != VK_NULL_HANDLE) {
        //     if (stream_internal->command_buffer != VK_NULL_HANDLE) {
        //         vkFreeCommandBuffers(device->device,
        //                              stream_internal->command_pool, 1,
        //                              &stream_internal->command_buffer);
        //     }
        //     vkDestroyCommandPool(device->device,
        //     stream_internal->command_pool,
        //                          NULL);
        // }
        if (has_seperate_transfer_queue(device) &&
            stream_internal->transfer_command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device->device,
                                 stream_internal->transfer_command_pool, NULL);
        }
        if (stream_internal->compute_command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device->device,
                                 stream_internal->compute_command_pool, NULL);
        }
        sccl_free(stream_internal);
    }

    return error;
}

void sccl_destroy_stream(sccl_stream_t stream)
{
    vkDestroySemaphore(stream->device->device, stream->timeline_semaphore,
                       NULL);

    free_command_buffers(stream);
    vector_destroy(&stream->command_buffers);

    vkDestroyFence(stream->device->device, stream->fence, NULL);

    free_descriptor_sets(stream->device->device, &stream->descriptor_sets);
    vector_destroy(&stream->descriptor_sets);

    // vkFreeCommandBuffers(stream->device->device, stream->command_pool, 1,
    //                      &stream->command_buffer);
    // vkDestroyCommandPool(stream->device->device, stream->command_pool, NULL);

    /* only set if seperate transfer queue */
    if (has_seperate_transfer_queue(stream->device)) {
        vkDestroyCommandPool(stream->device->device,
                             stream->transfer_command_pool, NULL);
    }
    vkDestroyCommandPool(stream->device->device, stream->compute_command_pool,
                         NULL);

    sccl_free(stream);
}

sccl_error_t sccl_dispatch_stream(const sccl_stream_t stream)
{
    if (vector_get_size(&stream->command_buffers) <= 0) {
        /* if no command buffers, submit without command buffer to fence is
         * signaled */
        VkSubmitInfo submit_info = {0};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 0;
        submit_info.pCommandBuffers = VK_NULL_HANDLE;

        VkQueue queue;
        vkGetDeviceQueue(
            stream->device->device,
            get_queue_family_index(stream->device, command_buffer_type_compute),
            SCCL_QUEUE_INDEX, &queue);
        CHECK_VKRESULT_RET(
            vkQueueSubmit(queue, 1, &submit_info, stream->fence));

        return sccl_success;
    }

    /* end current command buffer */
    command_buffer_entry_t *last_command_buffer_entry =
        get_current_command_buffer(&stream->command_buffers);
    assert(last_command_buffer_entry != NULL);
    CHECK_VKRESULT_RET(
        vkEndCommandBuffer(last_command_buffer_entry->command_buffer));

    /* Submit command buffers in order. For each command buffer, add semaphore
     * dependencies */
    for (uint64_t i = 0; i < vector_get_size(&stream->command_buffers); ++i) {
        command_buffer_entry_t *command_buffer_entry =
            vector_get_element(&stream->command_buffers, i);

        const uint64_t wait_value = i;
        /* last command buffer will reset timeline semaphore back to 0 */
        const uint64_t signal_value = i + 1;

        VkTimelineSemaphoreSubmitInfo timeline_semaphore_submit_info = {0};
        timeline_semaphore_submit_info.sType =
            VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_semaphore_submit_info.waitSemaphoreValueCount = 1;
        timeline_semaphore_submit_info.pWaitSemaphoreValues = &wait_value;
        timeline_semaphore_submit_info.signalSemaphoreValueCount = 1;
        timeline_semaphore_submit_info.pSignalSemaphoreValues = &signal_value;

        const VkPipelineStageFlags wait_stages =
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

        VkSubmitInfo submit_info = {0};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.pNext = &timeline_semaphore_submit_info;
        submit_info.waitSemaphoreCount = (i == 0) ? 0 : 1;
        submit_info.pWaitSemaphores = &stream->timeline_semaphore;
        submit_info.pWaitDstStageMask = &wait_stages;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &stream->timeline_semaphore;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer_entry->command_buffer;

        VkQueue queue;
        vkGetDeviceQueue(
            stream->device->device,
            get_queue_family_index(stream->device, command_buffer_entry->type),
            SCCL_QUEUE_INDEX, &queue);
        VkFence fence = (i >= vector_get_size(&stream->command_buffers) - 1)
                            ? stream->fence
                            : VK_NULL_HANDLE;
        CHECK_VKRESULT_RET(vkQueueSubmit(queue, 1, &submit_info, fence));
    }

    return sccl_success;
}

sccl_error_t sccl_join_stream(const sccl_stream_t stream)
{
    /* wait */
    CHECK_SCCL_ERROR_RET(sccl_wait_streams(stream->device, &stream, 1, NULL));

    /* reset */
    CHECK_SCCL_ERROR_RET(sccl_reset_stream(stream));

    return sccl_success;
}

sccl_error_t sccl_reset_stream(const sccl_stream_t stream)
{
    /* free descriptor sets */
    CHECK_SCCL_ERROR_RET(
        free_descriptor_sets(stream->device->device, &stream->descriptor_sets));

    /* reset command buffer here so we can record for next dispatch */
    // CHECK_SCCL_ERROR_RET(reset_command_buffer(stream));

    /* free command buffers */
    free_command_buffers(stream);

    /* reset timeline semaphore */
    vkDestroySemaphore(stream->device->device, stream->timeline_semaphore,
                       NULL);
    CHECK_SCCL_ERROR_RET(create_timeline_semaphore(
        stream->device->device, &stream->timeline_semaphore));

    /* reset fence */
    CHECK_VKRESULT_RET(
        vkResetFences(stream->device->device, 1, &stream->fence));

    return sccl_success;
}

sccl_error_t sccl_wait_streams(const sccl_device_t device,
                               const sccl_stream_t *streams,
                               size_t streams_count, uint8_t *completed_streams)
{
    return wait_streams(device, streams, streams_count, completed_streams,
                        false);
}

sccl_error_t sccl_wait_streams_all(const sccl_device_t device,
                                   const sccl_stream_t *streams,
                                   size_t streams_count)
{
    return wait_streams(device, streams, streams_count, NULL, true);
}

sccl_error_t sccl_copy_buffer(const sccl_stream_t stream,
                              const sccl_buffer_t src, size_t src_offset,
                              const sccl_buffer_t dst, size_t dst_offset,
                              size_t size)
{
    /* determine command buffer to record to */
    VkCommandBuffer command_buffer;
    CHECK_SCCL_ERROR_RET(determine_next_command_buffer(
        stream, command_buffer_type_transfer, &command_buffer));

    /* record command */
    VkBufferCopy buffer_copy = {0};
    buffer_copy.srcOffset = src_offset;
    buffer_copy.dstOffset = dst_offset;
    buffer_copy.size = size;
    vkCmdCopyBuffer(command_buffer, src->buffer, dst->buffer, 1, &buffer_copy);

    /* create barrier so next command will wait until this is finished */
    VkMemoryBarrier memory_barrier = {};
    memory_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memory_barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    memory_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                         &memory_barrier, 0, NULL, 0, NULL);

    return sccl_success;
}
