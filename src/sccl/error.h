#pragma once
#ifndef SCCL_ERROR_HEADER
#define SCCL_ERROR_HEADER

#include "sccl.h"
#include <assert.h>
#include <vulkan/vulkan.h>

#include <stdio.h>
#include <vulkan/vk_enum_string_helper.h>

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

/* Check `VkResult`, if success then continue, else goto `goto_label`, , store
 * sccl error in `store_sccl_error` if error */
#define CHECK_VKRESULT_GOTO(result, goto_label, store_sccl_error)              \
    do {                                                                       \
        if (result != VK_SUCCESS) {                                            \
            fprintf(stderr, "Vulkan error: %s\n", string_VkResult(result));    \
            store_sccl_error = sccl_unhandled_vulkan_error;                    \
            goto goto_label;                                                   \
        }                                                                      \
    } while (0)

/* Check `sccl_error_t`, if success then continue, else return error */
#define CHECK_SCCL_ERROR_RET(error)                                            \
    do {                                                                       \
        if (error != sccl_success) {                                           \
            return error;                                                      \
        }                                                                      \
    } while (0)

/* Check `sccl_error_t`, if success then continue, else goto `goto_label`, store
 * sccl error in `store_sccl_error` if error */
#define CHECK_SCCL_ERROR_GOTO(error, goto_label, store_sccl_error)             \
    do {                                                                       \
        if (error != sccl_success) {                                           \
            store_sccl_error = error;                                          \
            goto goto_label;                                                   \
        }                                                                      \
    } while (0)

/* Check if pointer is NULL, return invalid argument if NULL */
#define CHECK_SCCL_NULL_RET(pointer)                                           \
    do {                                                                       \
        if (pointer == NULL) {                                                 \
            return sccl_invalid_argument;                                      \
        }                                                                      \
    } while (0)

#endif // SCCL_ERROR_HEADER