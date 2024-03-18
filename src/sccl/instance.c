
#include "instance.h"
#include "alloc.h"
#include "environment_variables.h"
#include "error.h"
#include "sccl.h"
#include <stdio.h>
#include <string.h>

static const char *validation_layers[] = {"VK_LAYER_KHRONOS_validation"};
static const size_t num_validation_layers = 1;

static VkResult create_debug_utils_messenger_ext(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger)
{
    PFN_vkCreateDebugUtilsMessengerEXT func =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != NULL) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

static void
destroy_debug_utils_messenger_ext(VkInstance instance,
                                  VkDebugUtilsMessengerEXT debugMessenger,
                                  const VkAllocationCallbacks *pAllocator)
{
    PFN_vkDestroyDebugUtilsMessengerEXT func =
        (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != NULL) {
        func(instance, debugMessenger, pAllocator);
    }
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData)
{
    (void)messageSeverity;
    (void)messageType;
    (void)pUserData;
    fprintf(stderr, "validation layer: %s\n", pCallbackData->pMessage);

    if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) &&
        (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) &&
        is_assert_on_validation_error_set()) {
        assert(false);
    }

    return VK_FALSE;
}

static void populate_debug_messenger_create_info(
    VkDebugUtilsMessengerCreateInfoEXT *createInfo)
{
    memset(createInfo, 0, sizeof(VkDebugUtilsMessengerCreateInfoEXT));
    createInfo->sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo->messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo->messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo->pfnUserCallback = debug_callback;
}

static VkResult setup_debug_messenger(VkInstance instance,
                                      VkDebugUtilsMessengerEXT *debug_messenger)
{
    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populate_debug_messenger_create_info(&createInfo);

    return create_debug_utils_messenger_ext(instance, &createInfo, NULL,
                                            debug_messenger);
}

static sccl_error_t check_validation_layer_support(bool *supported)
{
    uint32_t layer_count;
    CHECK_VKRESULT_RET(vkEnumerateInstanceLayerProperties(&layer_count, NULL));

    VkLayerProperties *available_layers;
    CHECK_SCCL_ERROR_RET(sccl_calloc((void **)&available_layers, layer_count,
                                     sizeof(VkLayerProperties)));

    CHECK_VKRESULT_RET(
        vkEnumerateInstanceLayerProperties(&layer_count, available_layers));

    for (size_t layer_index = 0; layer_index < num_validation_layers;
         ++layer_index) {
        const char *layer_name = validation_layers[layer_index];
        bool layer_found = false;

        for (size_t i = 0; i < layer_count; ++i) {
            VkLayerProperties *layer_properties = &available_layers[i];
            if (strcmp(layer_name, layer_properties->layerName) == 0) {
                layer_found = true;
                break;
            }
        }

        if (!layer_found) {
            sccl_free(available_layers);
            *supported = false;
            return sccl_success;
        }
    }
    sccl_free(available_layers);

    *supported = true;

    return sccl_success;
}

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

        bool supported;
        CHECK_SCCL_ERROR_RET(check_validation_layer_support(&supported));
        if (!supported) {
            return sccl_unsupported_error;
        }

        create_info.enabledExtensionCount = 1;
        create_info.ppEnabledExtensionNames = extention_names;
        create_info.enabledLayerCount = num_validation_layers;
        create_info.ppEnabledLayerNames = validation_layers;
    }

    CHECK_VKRESULT_RET(
        vkCreateInstance(&create_info, NULL, &instance_internal->instance));

    if (is_enable_validation_layers_set()) {
        CHECK_VKRESULT_RET(setup_debug_messenger(
            instance_internal->instance, &instance_internal->debug_messenger));
    }

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

    if (is_enable_validation_layers_set()) {
        destroy_debug_utils_messenger_ext(instance->instance,
                                          instance->debug_messenger, NULL);
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
