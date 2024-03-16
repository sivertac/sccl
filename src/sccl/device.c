
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
                                            uint32_t *queue_family_index)
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

    uint32_t queue_family_index;
    CHECK_SCCL_ERROR_RET(
        find_queue_family_index(physical_device, &queue_family_index));

    /* create logical device
       priority of the compute queue (0.0 to 1.0)
    */
    float queue_priority = 1.0f;

    VkDeviceQueueCreateInfo queue_create_info = {0};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = queue_family_index;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceFeatures physical_device_features = {0};
    physical_device_features.shaderInt64 = true;

    VkDeviceCreateInfo device_create_info = {0};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_create_info;
    device_create_info.pEnabledFeatures = &physical_device_features;

    CHECK_VKRESULT_RET(vkCreateDevice(physical_device, &device_create_info,
                                      NULL, &device_internal->device));

    /* set public handle */
    *device = (sccl_device_t)device_internal;

    return sccl_success;
}

void sccl_destroy_device(sccl_device_t device)
{
    vkDestroyDevice(device->device, NULL);

    sccl_free(device);
}
