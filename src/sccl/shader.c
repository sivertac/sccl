
#include "shader.h"
#include "alloc.h"
#include "device.h"
#include "error.h"
#include "sccl.h"

sccl_error_t sccl_create_shader(const sccl_device_t device,
                                sccl_shader_t *shader,
                                const sccl_shader_config_t *config)
{
    /* validate config */
    CHECK_SCCL_NULL_RET(config);
    CHECK_SCCL_NULL_RET(config->shader_source_code);
    if (config->shader_source_code_length <= 0) {
        return sccl_invalid_argument;
    }

    /* create internal handle */
    struct sccl_shader *shader_internal;
    CHECK_SCCL_ERROR_RET(
        sccl_calloc((void **)&shader_internal, 1, sizeof(struct sccl_shader)));

    shader_internal->device = device->device;

    /* create shader module */
    VkShaderModuleCreateInfo shader_module_create_info = {0};
    shader_module_create_info.sType =
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_module_create_info.codeSize = config->shader_source_code_length;
    shader_module_create_info.pCode =
        (const uint32_t *)config->shader_source_code;
    CHECK_VKRESULT_RET(
        vkCreateShaderModule(device->device, &shader_module_create_info,
                             VK_NULL_HANDLE, &shader_internal->shader_module));

    /* create compute pipeline */
    VkPipelineLayoutCreateInfo pipeline_layout_create_info = {0};
    pipeline_layout_create_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_create_info.setLayoutCount = 0;
    pipeline_layout_create_info.pSetLayouts = VK_NULL_HANDLE;
    CHECK_VKRESULT_RET(vkCreatePipelineLayout(
        device->device, &pipeline_layout_create_info, VK_NULL_HANDLE,
        &shader_internal->pipeline_layout));

    VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info = {0};
    pipeline_shader_stage_create_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_shader_stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_shader_stage_create_info.module = shader_internal->shader_module;
    pipeline_shader_stage_create_info.pName = "main";
    pipeline_shader_stage_create_info.pSpecializationInfo = VK_NULL_HANDLE;

    VkComputePipelineCreateInfo compute_pipeline_create_info = {0};
    compute_pipeline_create_info.sType =
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_pipeline_create_info.layout = shader_internal->pipeline_layout;
    compute_pipeline_create_info.stage = pipeline_shader_stage_create_info;
    CHECK_VKRESULT_RET(vkCreateComputePipelines(
        device->device, VK_NULL_HANDLE, 1, &compute_pipeline_create_info,
        VK_NULL_HANDLE, &shader_internal->compute_pipeline));

    /* set public handle */
    *shader = (sccl_shader_t)shader_internal;

    return sccl_success;
}

void sccl_destroy_shader(sccl_shader_t shader)
{
    vkDestroyPipeline(shader->device, shader->compute_pipeline, VK_NULL_HANDLE);
    vkDestroyPipelineLayout(shader->device, shader->pipeline_layout,
                            VK_NULL_HANDLE);
    vkDestroyShaderModule(shader->device, shader->shader_module,
                          VK_NULL_HANDLE);
}

// sccl_error_t sccl_run_shader(const sccl_shader_t shader, const
// sccl_shader_run_params_t *params)
//{
//     return sccl_success;
// }
