
#include "device.h"
#include "alloc.h"
#include "error.h"
#include "instance.h"
#include <string.h>

#define QUEUE_COUNT 2

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
                                            VkQueueFlags queue_flags,
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

    /* find queue that is closest to queue_flags */
    bool found = false;
    VkQueueFlags last_queue_flags =
        VK_QUEUE_FLAG_BITS_MAX_ENUM; /* init all bits set */
    for (size_t i = 0; i < queue_family_count; ++i) {
        VkQueueFlags check_queue_flags =
            queue_family_properties[i].queueFamilyProperties.queueFlags;
        /* if queue flags match, and less bits are set than last selected queue
         * index */
        if (((check_queue_flags & queue_flags) == queue_flags) &&
            __builtin_popcount(check_queue_flags) <
                __builtin_popcount(last_queue_flags)) {
            found = true;
            last_queue_flags = check_queue_flags;
            *queue_family_index = i;
        }
    }

    if (!found) {
        sccl_free(queue_family_properties);
        return sccl_unsupported_error;
    }

    sccl_free(queue_family_properties);

    return sccl_success;
}

static sccl_error_t check_device_extension_support(
    VkPhysicalDevice physical_device, const char **check_extension_names,
    size_t check_extension_names_count, bool *supported)
{
    sccl_error_t error = sccl_success;

    uint32_t available_extension_properties_list_count = 0;
    VkExtensionProperties *available_extension_properties_list = NULL;

    CHECK_VKRESULT_GOTO(vkEnumerateDeviceExtensionProperties(
                            physical_device, NULL,
                            &available_extension_properties_list_count, NULL),
                        error_return, error);

    CHECK_SCCL_ERROR_GOTO(
        sccl_calloc((void **)&available_extension_properties_list,
                    available_extension_properties_list_count,
                    sizeof(VkExtensionProperties)),
        error_return, error);

    CHECK_VKRESULT_GOTO(vkEnumerateDeviceExtensionProperties(
                            physical_device, NULL,
                            &available_extension_properties_list_count,
                            available_extension_properties_list),
                        error_return, error);

    size_t found_count = 0;
    for (size_t available_extension_index = 0;
         available_extension_index < available_extension_properties_list_count;
         ++available_extension_index) {
        VkExtensionProperties *extension_properties =
            &available_extension_properties_list[available_extension_index];
        for (size_t i = 0; i < check_extension_names_count; ++i) {
            if (strncmp(check_extension_names[i],
                        extension_properties->extensionName,
                        VK_MAX_EXTENSION_NAME_SIZE) == 0) {
                ++found_count;
            }
        }
    }

    sccl_free(available_extension_properties_list);

    *supported = !(found_count < check_extension_names_count);

    return sccl_success;

error_return:
    if (available_extension_properties_list != NULL) {
        sccl_free(available_extension_properties_list);
    }

    return error;
}

static const char *all_wanted_device_extension_names[] = {
    VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME};
static const uint32_t all_wanted_device_extension_names_count = 3;

/**
 * device_extension_names must of of size
 * all_wanted_device_extension_names_count. device_extension_names_count is set
 * by this function.
 */
static sccl_error_t determine_device_extensions(
    VkPhysicalDevice physical_device, char **device_extension_names,
    size_t *device_extension_names_count, bool *host_pointer_supported,
    bool *dmabuf_buffer_supported)
{
    (void)all_wanted_device_extension_names;
    assert(all_wanted_device_extension_names_count ==
           sizeof(all_wanted_device_extension_names) / sizeof(const char *));

    *device_extension_names_count = 0;

    /* check host pointer support */
    const char *host_pointer_ext_names[] = {
        VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME};
    const size_t host_pointer_ext_names_count =
        sizeof(host_pointer_ext_names) / sizeof(char *);
    CHECK_SCCL_ERROR_RET(check_device_extension_support(
        physical_device, host_pointer_ext_names, host_pointer_ext_names_count,
        host_pointer_supported));
    if (*host_pointer_supported) {
        memcpy((void *)(device_extension_names + *device_extension_names_count),
               host_pointer_ext_names,
               host_pointer_ext_names_count * sizeof(const char *));
        *device_extension_names_count += host_pointer_ext_names_count;
    }

    /* check dmabuf support */
    const char *dmabuf_ext_names[] = {
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME};
    const size_t dmabuf_ext_names_count =
        sizeof(dmabuf_ext_names) / sizeof(char *);
    CHECK_SCCL_ERROR_RET(check_device_extension_support(
        physical_device, dmabuf_ext_names, dmabuf_ext_names_count,
        dmabuf_buffer_supported));
    if (*dmabuf_buffer_supported) {
        memcpy((void *)(device_extension_names + *device_extension_names_count),
               dmabuf_ext_names, dmabuf_ext_names_count * sizeof(const char *));
        *device_extension_names_count += dmabuf_ext_names_count;
    }

    return sccl_success;
}

bool has_seperate_transfer_queue(const sccl_device_t device)
{
    return device->compute_queue_family_index !=
           device->transfer_queue_family_index;
}

sccl_error_t sccl_create_device(const sccl_instance_t instance,
                                sccl_device_t *device, uint32_t device_index)
{
    sccl_error_t error = sccl_success;

    struct sccl_device *device_internal = NULL;

    char **device_extensions = NULL;
    size_t device_extensions_count = 0;

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
        find_queue_family_index(physical_device, VK_QUEUE_COMPUTE_BIT,
                                &device_internal->compute_queue_family_index),
        error_return, error);

    CHECK_SCCL_ERROR_GOTO(
        find_queue_family_index(physical_device, VK_QUEUE_TRANSFER_BIT,
                                &device_internal->transfer_queue_family_index),
        error_return, error);

    const bool different_queue_family =
        has_seperate_transfer_queue(device_internal);

    float queue_priority[] = {1.0, 1.0};

    VkDeviceQueueCreateInfo queue_create_infos[QUEUE_COUNT];
    memset(queue_create_infos, 0, sizeof(queue_create_infos));
    queue_create_infos[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_infos[0].queueFamilyIndex =
        device_internal->compute_queue_family_index;
    queue_create_infos[0].queueCount = 1;
    queue_create_infos[0].pQueuePriorities = queue_priority;
    if (different_queue_family) {
        queue_create_infos[1].sType =
            VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_infos[1].queueFamilyIndex =
            device_internal->transfer_queue_family_index;
        queue_create_infos[1].queueCount = 1;
        queue_create_infos[1].pQueuePriorities = queue_priority;
    }

    /* enable required features */
    VkPhysicalDeviceFeatures physical_device_features = {0};
    physical_device_features.shaderInt64 = true;
    VkPhysicalDeviceVulkan12Features physical_device_vulkan_1_2_features = {0};
    physical_device_vulkan_1_2_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    physical_device_vulkan_1_2_features.timelineSemaphore = true;

    /* enable extentions if available */
    CHECK_SCCL_ERROR_GOTO(sccl_calloc((void **)&device_extensions,
                                      all_wanted_device_extension_names_count,
                                      sizeof(char *)),
                          error_return, error);
    CHECK_SCCL_ERROR_GOTO(
        determine_device_extensions(physical_device, device_extensions,
                                    &device_extensions_count,
                                    &device_internal->host_pointer_supported,
                                    &device_internal->dmabuf_buffer_supported),
        error_return, error);

    VkDeviceCreateInfo device_create_info = {0};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.pNext = &physical_device_vulkan_1_2_features;
    device_create_info.queueCreateInfoCount =
        (different_queue_family) ? QUEUE_COUNT : 1;
    device_create_info.pQueueCreateInfos = queue_create_infos;
    device_create_info.pEnabledFeatures = &physical_device_features;

    device_create_info.ppEnabledExtensionNames =
        (const char *const *)device_extensions;
    device_create_info.enabledExtensionCount = device_extensions_count;

    CHECK_VKRESULT_GOTO(vkCreateDevice(physical_device, &device_create_info,
                                       NULL, &device_internal->device),
                        error_return, error);

    /* dynamically load extension API calls */
    if (device_internal->dmabuf_buffer_supported) {
        device_internal->pfn_vk_get_memory_fd_khr =
            (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device_internal->device,
                                                      "vkGetMemoryFdKHR");
        device_internal->pfn_vk_get_memory_fd_properties_khr =
            (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(
                device_internal->device, "vkGetMemoryFdPropertiesKHR");
    }

    /* cleanup */
    sccl_free(device_extensions);

    /* set public handle */
    *device = (sccl_device_t)device_internal;

    return sccl_success;

error_return:

    if (device_extensions != NULL) {
        sccl_free(device_extensions);
    }

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

    VkPhysicalDeviceExternalMemoryHostPropertiesEXT
        physical_device_external_memory_host_properties = {0};
    physical_device_external_memory_host_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
    VkPhysicalDeviceSubgroupProperties physical_device_subgroup_properties = {
        0};
    physical_device_subgroup_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    physical_device_subgroup_properties.pNext =
        &physical_device_external_memory_host_properties;
    VkPhysicalDeviceProperties2 physical_device_properties = {0};
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
    device_properties->min_external_buffer_host_pointer_alignment =
        physical_device_external_memory_host_properties
            .minImportedHostPointerAlignment;
}
