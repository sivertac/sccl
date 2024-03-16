
#include "instance.h"
#include "alloc.h"
#include "enviroment_variables.h"
#include "error.h"
#include "sccl.h"
#include <stdio.h>

static const char *validation_layers[] = {"VK_LAYER_KHRONOS_validation"};
static const size_t num_validation_layers = 1;

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

    /* set public handle */
    *instance = (sccl_instance_t)instance_internal;

    return sccl_success;
}

void sccl_destroy_instance(sccl_instance_t instance)
{

    vkDestroyInstance(instance->instance, NULL);

    sccl_free((void *)instance);
}
