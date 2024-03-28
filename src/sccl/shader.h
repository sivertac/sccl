#pragma once
#ifndef SHADER_HEADER
#define SHADER_HEADER

#include <vulkan/vulkan.h>

struct sccl_shader {
    VkDevice device;
    VkShaderModule shader_module;
    VkDescriptorSetLayout *descriptor_set_layouts;
    size_t descriptor_set_layouts_count;
    VkDescriptorPool descriptor_pool;
    VkPipelineLayout pipeline_layout;
    VkPipeline compute_pipeline;
};

#endif // SHADER_HEADER