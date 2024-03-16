
#include "instance.h"
#include "alloc.h"
#include "enviroment_variables.h"
#include "error.h"
#include "sccl.h"
#include <stdio.h>

static const char *validation_layers[] = {"VK_LAYER_KHRONOS_validation"};
static const size_t num_validation_layers = 1;

static sccl_error_t update_physical_device_list(sccl_instance_t instance)
{

    if (instance->physical_devices != SCCL_NULL) {
        sccl_free(instance->physical_devices);
    }

    CHECK_VKRESULT_RET(vkEnumeratePhysicalDevices(
        instance->instance, &instance->physical_device_count, NULL));

    CHECK_SCCL_ERROR_RET(sccl_calloc((void **)&instance->physical_devices,
                                     instance->physical_device_count,
                                     sizeof(VkPhysicalDevice)));

    CHECK_VKRESULT_RET(vkEnumeratePhysicalDevices(
        instance->instance, &instance->physical_device_count,
        instance->physical_devices));

    return sccl_success;
}

sccl_error_t sccl_create_instance(sccl_instance_t *instance)
{
    struct sccl_instance *instance_internal;
    CHECK_SCCL_ERROR_RET(sccl_calloc((void **)&instance_internal, 1,
                                     sizeof(struct sccl_instance)));

    VkApplicationInfo app_info = {0};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Compute Shader Meme";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "SCCL";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo create_info = {0};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    const char *extention_names[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};

    if (is_enable_validation_layers_set()) {
        printf("Enabling validation layers\n");

        create_info.enabledExtensionCount = 1;
        create_info.ppEnabledExtensionNames = extention_names;
        create_info.enabledLayerCount = num_validation_layers;
        create_info.ppEnabledLayerNames = validation_layers;
    }

    CHECK_VKRESULT_RET(
        vkCreateInstance(&create_info, NULL, &instance_internal->instance));

    /* populate physical device list */
    CHECK_SCCL_ERROR_RET(update_physical_device_list(instance_internal));

    /* set public handle */
    *instance = (sccl_instance_t)instance_internal;

    return sccl_success;
}

void sccl_destroy_instance(sccl_instance_t instance)
{
    if (instance->physical_devices != SCCL_NULL) {
        sccl_free(instance->physical_devices);
    }

    vkDestroyInstance(instance->instance, NULL);

    sccl_free((void *)instance);
}

sccl_error_t sccl_get_device_count(const sccl_instance_t instance,
                                   uint32_t *device_count)
{

    CHECK_SCCL_ERROR_RET(update_physical_device_list(instance));

    *device_count = instance->physical_device_count;

    return sccl_success;
}
