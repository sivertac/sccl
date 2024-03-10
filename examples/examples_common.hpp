#pragma once
#ifndef EXMAPLES_COMMON_HEADER
#define EXMAPLES_COMMON_HEADER

#include <binary_util.hpp>
#include <compute_interface.h>
#include <vulkan/vk_enum_string_helper.h>

#include <optional>

#define UNWRAP_VKRESULT(result)                                                \
    do {                                                                       \
        if (result != VK_SUCCESS) {                                            \
            fprintf(stderr, "Vulkan error: %s\n", string_VkResult(result));    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

void print_data_buffers(const ComputeDevice *compute_device,
                        size_t num_elements, VkDeviceMemory input_buffer_memory,
                        VkDeviceMemory output_buffer_memory);

std::optional<std::string> read_file(const char *filepath);

#endif // EXMAPLES_COMMON_HEADER