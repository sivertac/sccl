#pragma once
#ifndef SHADER_HEADER
#define SHADER_HEADER

#include "sccl.h"
#include <vulkan/vulkan.h>

struct sccl_shader {
    VkDevice device;
    VkShaderModule shader_module;
    VkDescriptorSetLayout *descriptor_set_layouts;
    size_t descriptor_set_layouts_count;
    VkDescriptorPool descriptor_pool;
    sccl_shader_push_constant_layout_t *push_constant_layouts;
    size_t push_constant_layouts_count;
    VkPipelineLayout pipeline_layout;
    VkPipeline compute_pipeline;
};

#endif // SHADER_HEADER