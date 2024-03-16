#pragma once
#ifndef VULKAN_COMMON_HEADER
#define VULKAN_COMMON_HEADER

#include "sccl.h"
#include <vulkan/vulkan.h>

#define UNWRAP_VKRESULT(result)                                                \
    do {                                                                       \
        if (result != VK_SUCCESS) {                                            \
            fprintf(stderr, "Vulkan error: %s\n", string_VkResult(result));    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* Check `VkResult`, if success then continue, else return
 * `sccl_unhandled_vulkan_error` */
#define CHECK_VKRESULT_RET(result)                                             \
    do {                                                                       \
        if (result != VK_SUCCESS) {                                            \
            return sccl_unhandled_vulkan_error;                                \
        }                                                                      \
    } while (0)

/* Check `sccl_error_t`, if success then continue, else return error */
#define CHECK_SCCL_ERROR_RET(error)                                            \
    do {                                                                       \
        if (error != sccl_success) {                                           \
            return error;                                                      \
        }                                                                      \
    } while (0)

#endif // VULKAN_COMMON_HEADER