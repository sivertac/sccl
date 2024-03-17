
#include "device.h"
#include "alloc.h"
#include "error.h"
#include "instance.h"
#include <stdbool.h>

static sccl_error_t
get_physical_device_at_index(const sccl_instance_t instance,
                             VkPhysicalDevice *physical_device,
                             uint32_t device_index)
{
    assert(instance->physical_devices != SCCL_NULL);
    if (device_index >= instance->physical_device_count) {
        return sccl_invalid_argument;
    }

    *physical_device = instance->physical_devices[device_index];

    return sccl_success;
}

static sccl_error_t find_queue_family_index(VkPhysicalDevice physical_device,
                                            uint32_t *queue_family_index,
                                            uint32_t *queue_count)
{
    uint32_t queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties2(physical_device,
                                              &queue_family_count, NULL);
    if (queue_family_count == 0) {
        return sccl_unsupported_error;
    }

    VkQueueFamilyProperties2 *queue_family_properties;
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
            *queue_count =
                queue_family_properties[i].queueFamilyProperties.queueCount;
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
    struct sccl_device *device_internal;
    CHECK_SCCL_ERROR_RET(
        sccl_calloc((void **)&device_internal, 1, sizeof(struct sccl_device)));

    /* find device at index */
    VkPhysicalDevice physical_device;
    CHECK_SCCL_ERROR_RET(
        get_physical_device_at_index(instance, &physical_device, device_index));
    device_internal->physical_device = physical_device;

    CHECK_SCCL_ERROR_RET(find_queue_family_index(
        physical_device, &device_internal->queue_family_index,
        &device_internal->queue_count));

    /* allocate space to store what queues are in use */
    CHECK_SCCL_ERROR_RET(sccl_calloc((void **)&device_internal->queue_usage,
                                     device_internal->queue_count,
                                     sizeof(bool)));

    float *queue_priorities;
    CHECK_SCCL_ERROR_RET(sccl_calloc((void **)&queue_priorities,
                                     device_internal->queue_count,
                                     sizeof(float)));
    for (size_t i = 0; i < device_internal->queue_count; ++i) {
        queue_priorities[i] = 1.0f;
    }

    VkDeviceQueueCreateInfo queue_create_info = {0};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = device_internal->queue_family_index;
    queue_create_info.queueCount = device_internal->queue_count;
    queue_create_info.pQueuePriorities = queue_priorities;

    VkPhysicalDeviceFeatures physical_device_features = {0};
    physical_device_features.shaderInt64 = true;

    VkDeviceCreateInfo device_create_info = {0};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_create_info;
    device_create_info.pEnabledFeatures = &physical_device_features;

    CHECK_VKRESULT_RET(vkCreateDevice(physical_device, &device_create_info,
                                      NULL, &device_internal->device));

    sccl_free(queue_priorities);

    /* set public handle */
    *device = (sccl_device_t)device_internal;

    return sccl_success;
}

void sccl_destroy_device(sccl_device_t device)
{
    vkDestroyDevice(device->device, NULL);

    sccl_free(device->queue_usage);
    sccl_free(device);
}
