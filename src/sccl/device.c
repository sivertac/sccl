
#include "device.h"
#include "alloc.h"
#include "error.h"
#include "instance.h"
#include <stdbool.h>
#include <string.h>

static sccl_error_t
get_physical_device_at_index(const sccl_instance_t instance,
                             VkPhysicalDevice *physical_device,
                             uint32_t device_index)
{
    assert(instance->physical_devices != NULL);
    if (device_index >= instance->physical_device_count) {
        return sccl_invalid_argument;
    }

    *physical_device = instance->physical_devices[device_index];

    return sccl_success;
}

static sccl_error_t find_queue_family_index(VkPhysicalDevice physical_device,
                                            uint32_t *queue_family_index)
{
    uint32_t queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties2(physical_device,
                                              &queue_family_count, NULL);
    if (queue_family_count == 0) {
        return sccl_unsupported_error;
    }

    VkQueueFamilyProperties2 *queue_family_properties = NULL;
    CHECK_SCCL_ERROR_RET(sccl_calloc((void **)&queue_family_properties,
                                     queue_family_count,
                                     sizeof(VkQueueFamilyProperties2)));
    for (size_t i = 0; i < queue_family_count; ++i) {
        queue_family_properties[i].sType =
            VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
    }

    vkGetPhysicalDeviceQueueFamilyProperties2(
        physical_device, &queue_family_count, queue_family_properties);

    bool found = false;
    for (size_t i = 0; i < queue_family_count; ++i) {
        VkQueueFlagBits queue_flags =
            queue_family_properties[i].queueFamilyProperties.queueFlags;
        if ((queue_flags & VK_QUEUE_COMPUTE_BIT) &&
            (queue_flags & VK_QUEUE_TRANSFER_BIT)) {
            found = true;
            *queue_family_index = i;
            break;
        }
    }

    if (!found) {
        sccl_free(queue_family_properties);
        return sccl_unsupported_error;
    }

    sccl_free(queue_family_properties);

    return sccl_success;
}

sccl_error_t sccl_create_device(const sccl_instance_t instance,
                                sccl_device_t *device, uint32_t device_index)
{
    sccl_error_t error = sccl_success;

    struct sccl_device *device_internal = NULL;
    CHECK_SCCL_ERROR_GOTO(
        sccl_calloc((void **)&device_internal, 1, sizeof(struct sccl_device)),
        error_return, error);

    /* find device at index */
    VkPhysicalDevice physical_device;
    CHECK_SCCL_ERROR_GOTO(
        get_physical_device_at_index(instance, &physical_device, device_index),
        error_return, error);
    device_internal->physical_device = physical_device;

    CHECK_SCCL_ERROR_GOTO(
        find_queue_family_index(physical_device,
                                &device_internal->queue_family_index),
        error_return, error);

    float queue_priority = 1.0;

    VkDeviceQueueCreateInfo queue_create_info = {0};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = device_internal->queue_family_index;
    /* only allocate 1 queue */
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceFeatures physical_device_features = {0};
    physical_device_features.shaderInt64 = true;

    VkDeviceCreateInfo device_create_info = {0};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_create_info;
    device_create_info.pEnabledFeatures = &physical_device_features;

    CHECK_VKRESULT_GOTO(vkCreateDevice(physical_device, &device_create_info,
                                       NULL, &device_internal->device),
                        error_return, error);

    /* set public handle */
    *device = (sccl_device_t)device_internal;

    return sccl_success;

error_return:
    if (device_internal != NULL) {
        if (device_internal->device != VK_NULL_HANDLE) {
            vkDestroyDevice(device_internal->device, NULL);
        }
        sccl_free(device_internal);
    }

    return error;
}

void sccl_destroy_device(sccl_device_t device)
{
    vkDestroyDevice(device->device, NULL);

    sccl_free(device);
}

void sccl_get_device_properties(const sccl_device_t device,
                                sccl_device_properties_t *device_properties)
{
    memset(device_properties, 0, sizeof(sccl_device_properties_t));

    VkPhysicalDeviceSubgroupProperties physical_device_subgroup_properties = {};
    physical_device_subgroup_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 physical_device_properties = {};
    physical_device_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    physical_device_properties.pNext = &physical_device_subgroup_properties;
    vkGetPhysicalDeviceProperties2(device->physical_device,
                                   &physical_device_properties);
    for (size_t i = 0; i < 3; ++i) {
        device_properties->max_work_group_count[i] =
            physical_device_properties.properties.limits
                .maxComputeWorkGroupCount[i];
    }
    for (size_t i = 0; i < 3; ++i) {
        device_properties->max_work_group_size[i] =
            physical_device_properties.properties.limits
                .maxComputeWorkGroupSize[i];
    }
    device_properties->max_work_group_invocations =
        physical_device_properties.properties.limits
            .maxComputeWorkGroupInvocations;
    device_properties->native_work_group_size =
        physical_device_subgroup_properties.subgroupSize;
    device_properties->max_storage_buffer_size =
        physical_device_properties.properties.limits.maxStorageBufferRange;
    device_properties->max_uniform_buffer_size =
        physical_device_properties.properties.limits.maxUniformBufferRange;
    device_properties->max_push_constant_size =
        physical_device_properties.properties.limits.maxPushConstantsSize;
    device_properties->min_storage_buffer_offset_alignment =
        physical_device_properties.properties.limits
            .minStorageBufferOffsetAlignment;
    device_properties->min_uniform_buffer_offset_alignment =
        physical_device_properties.properties.limits
            .minUniformBufferOffsetAlignment;
}
