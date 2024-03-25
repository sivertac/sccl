#pragma once
#ifndef SHADER_HEADER
#define SHADER_HEADER

#include <vulkan/vulkan.h>

struct sccl_shader {
    VkDevice device;
    VkShaderModule shader_module;
    VkPipelineLayout pipeline_layout;
    VkPipeline compute_pipeline;
};

#endif