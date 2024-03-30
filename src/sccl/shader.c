
#include "shader.h"
#include "alloc.h"
#include "buffer.h"
#include "device.h"
#include "error.h"
#include "sccl.h"
#include "stream.h"
#include "vector.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    uint32_t set;
    vector_t buffer_layouts;
} descriptor_set_entry_t;

static sccl_error_t descriptor_set_entry_create(descriptor_set_entry_t *entry,
                                                uint32_t set)
{
    entry->set = set;
    return vector_init(&entry->buffer_layouts,
                       sizeof(sccl_shader_buffer_layout_t));
}

static sccl_error_t descriptor_set_entry_add_buffer_layout(
    descriptor_set_entry_t *entry,
    const sccl_shader_buffer_layout_t *buffer_layout)
{
    return vector_add_element(&entry->buffer_layouts, buffer_layout);
}

static void descriptor_set_entry_destroy(descriptor_set_entry_t *entry)
{
    vector_destroy(&entry->buffer_layouts);
}

static int compare_descriptor_set_entry_set(const void *lhs, const void *rhs)
{
    uint32_t a = ((descriptor_set_entry_t *)lhs)->set;
    uint32_t b = ((descriptor_set_entry_t *)rhs)->set;
    return a < b ? -1 : a > b ? +1 : 0;
}

static int compare_sccl_buffer_layouts_position_binding(const void *lhs,
                                                        const void *rhs)
{
    uint32_t a = ((sccl_shader_buffer_layout_t *)lhs)->position.binding;
    uint32_t b = ((sccl_shader_buffer_layout_t *)rhs)->position.binding;
    return a < b ? -1 : a > b ? +1 : 0;
}

/**
 * Checks if we have a contiguous range of descriptor sets starting at 0.
 * Checks if we don't have any duplicate buffer entries.
 * Assumes sets and binding positions are sorted.
 */
static bool
validate_descriptor_set_layouts(const vector_t *descriptor_sets_to_allocate)
{

    for (size_t i = 0; i < vector_get_size(descriptor_sets_to_allocate); ++i) {
        descriptor_set_entry_t *e =
            (descriptor_set_entry_t *)vector_get_element(
                descriptor_sets_to_allocate, i);

        /* check contiguous set ramge */
        if (e->set != i) {
            return false;
        }

        /* check for duplicate buffer position bindings */
        for (size_t j = 0; j < vector_get_size(&e->buffer_layouts) - 1; ++j) {
            sccl_shader_buffer_layout_t *a =
                vector_get_element(&e->buffer_layouts, j);
            sccl_shader_buffer_layout_t *b =
                vector_get_element(&e->buffer_layouts, j + 1);
            if (a->position.binding == b->position.binding) {
                return false;
            }
        }
    }

    return true;
}

static sccl_error_t
count_descriptor_set_layouts(const sccl_shader_buffer_layout_t *buffer_layouts,
                             size_t buffer_layouts_count,
                             vector_t *descriptor_sets_to_allocate)
{

    for (size_t i = 0; i < buffer_layouts_count; ++i) {
        sccl_shader_buffer_layout_t layout = buffer_layouts[i];
        descriptor_set_entry_t *entry = NULL;

        /* check if entry is made for set */
        for (size_t j = 0; j < vector_get_size(descriptor_sets_to_allocate);
             ++j) {
            descriptor_set_entry_t *e =
                (descriptor_set_entry_t *)vector_get_element(
                    descriptor_sets_to_allocate, j);
            if (e->set == layout.position.set) {
                entry = e;
                break;
            }
        }

        /* if not found, allocate entry */
        if (entry == NULL) {
            descriptor_set_entry_t new_entry = {0};
            CHECK_SCCL_ERROR_RET(
                descriptor_set_entry_create(&new_entry, layout.position.set));
            CHECK_SCCL_ERROR_RET(
                vector_add_element(descriptor_sets_to_allocate, &new_entry));
            entry = vector_get_element(
                descriptor_sets_to_allocate,
                vector_get_size(descriptor_sets_to_allocate) - 1);
        }

        /* add buffer layout to entry */
        CHECK_SCCL_ERROR_RET(
            descriptor_set_entry_add_buffer_layout(entry, &layout));
    }

    /* sort sets */
    vector_sort(descriptor_sets_to_allocate, compare_descriptor_set_entry_set);

    /* sort buffer bindings */
    for (size_t i = 0; i < vector_get_size(descriptor_sets_to_allocate); ++i) {
        descriptor_set_entry_t *e =
            (descriptor_set_entry_t *)vector_get_element(
                descriptor_sets_to_allocate, i);
        vector_sort(&e->buffer_layouts,
                    compare_sccl_buffer_layouts_position_binding);
    }

    /* validate */
    if (!validate_descriptor_set_layouts(descriptor_sets_to_allocate)) {
        return sccl_invalid_argument;
    }

    return sccl_success;
}

static VkDescriptorType
sccl_buffer_type_to_vk_descriptor_type(const sccl_buffer_type_t type)
{
    switch (type) {
    case sccl_buffer_type_host_storage:
    case sccl_buffer_type_device_storage:
    case sccl_buffer_type_shared_storage:
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case sccl_buffer_type_host_uniform:
    case sccl_buffer_type_device_uniform:
    case sccl_buffer_type_shared_uniform:
        return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    default:
    }
    assert(false);
    return 0;
}

static void
count_buffer_types(const sccl_shader_buffer_layout_t *buffer_layouts,
                   size_t buffer_layouts_count, size_t *storage_buffer_count,
                   size_t *uniform_buffer_count)
{
    for (size_t i = 0; i < buffer_layouts_count; ++i) {
        switch (
            sccl_buffer_type_to_vk_descriptor_type(buffer_layouts[i].type)) {
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            ++(*storage_buffer_count);
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            ++(*uniform_buffer_count);
            break;
        default:
            assert(false);
        }
    }
}

static sccl_error_t create_descriptor_set_layouts(
    VkDevice device, const sccl_shader_buffer_layout_t *buffer_layouts,
    size_t buffer_layouts_count, VkDescriptorSetLayout **descriptor_set_layouts,
    size_t *descriptor_set_layouts_count)
{

    /* count number of descriptor sets needed */
    vector_t descriptor_sets_to_allocate = {0};
    CHECK_SCCL_ERROR_RET(vector_init(&descriptor_sets_to_allocate,
                                     sizeof(descriptor_set_entry_t)));

    CHECK_SCCL_ERROR_RET(count_descriptor_set_layouts(
        buffer_layouts, buffer_layouts_count, &descriptor_sets_to_allocate));

    /* alloc returned memory, this needs to be freed! */
    CHECK_SCCL_ERROR_RET(
        sccl_calloc((void **)descriptor_set_layouts,
                    vector_get_size(&descriptor_sets_to_allocate),
                    sizeof(VkDescriptorSetLayout)));
    *descriptor_set_layouts_count =
        vector_get_size(&descriptor_sets_to_allocate);

    for (size_t i = 0; i < vector_get_size(&descriptor_sets_to_allocate); ++i) {
        descriptor_set_entry_t *entry =
            vector_get_element(&descriptor_sets_to_allocate, i);

        VkDescriptorSetLayoutBinding *descriptor_set_bindings;
        CHECK_SCCL_ERROR_RET(
            sccl_calloc((void **)&descriptor_set_bindings,
                        vector_get_size(&entry->buffer_layouts),
                        sizeof(VkDescriptorSetLayoutBinding)));

        for (size_t j = 0; j < vector_get_size(&entry->buffer_layouts); ++j) {
            sccl_shader_buffer_layout_t *buffer_layout =
                vector_get_element(&entry->buffer_layouts, j);
            descriptor_set_bindings[j].binding =
                buffer_layout->position.binding;
            descriptor_set_bindings[j].descriptorType =
                sccl_buffer_type_to_vk_descriptor_type(buffer_layout->type);
            descriptor_set_bindings[j].descriptorCount = 1;
            descriptor_set_bindings[j].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = {0};
        descriptor_set_layout_create_info.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptor_set_layout_create_info.pBindings = descriptor_set_bindings;
        descriptor_set_layout_create_info.bindingCount =
            vector_get_size(&entry->buffer_layouts);

        CHECK_VKRESULT_RET(vkCreateDescriptorSetLayout(
            device, &descriptor_set_layout_create_info, NULL,
            &(*descriptor_set_layouts)[i]));

        sccl_free(descriptor_set_bindings);
    }

    /* destroy descriptor_sets_to_allocate */
    for (size_t i = 0; i < vector_get_size(&descriptor_sets_to_allocate); ++i) {
        descriptor_set_entry_destroy(
            vector_get_element(&descriptor_sets_to_allocate, i));
    }
    vector_destroy(&descriptor_sets_to_allocate);

    return sccl_success;
}

static sccl_error_t create_descriptor_pool(VkDevice device,
                                           size_t storage_buffer_count,
                                           size_t uniform_buffer_count,
                                           size_t max_descriptor_sets,
                                           VkDescriptorPool *descriptor_pool)
{
    const size_t max_descriptor_pool_sizes_count = 2; /* storage and uniform */
    VkDescriptorType structure_types[] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    size_t counts[] = {storage_buffer_count, uniform_buffer_count};

    VkDescriptorPoolSize descriptor_pool_sizes[max_descriptor_pool_sizes_count];
    memset(descriptor_pool_sizes, 0,
           sizeof(VkDescriptorPoolSize) * max_descriptor_pool_sizes_count);

    size_t actual_descriptor_pool_sizes_count = 0;
    for (size_t i = 0; i < max_descriptor_pool_sizes_count; ++i) {
        if (counts[i] > 0) {
            descriptor_pool_sizes[actual_descriptor_pool_sizes_count].type =
                structure_types[i];
            descriptor_pool_sizes[actual_descriptor_pool_sizes_count]
                .descriptorCount = counts[i];
            ++actual_descriptor_pool_sizes_count;
        }
    }

    VkDescriptorPoolCreateInfo descriptor_pool_create_info = {0};
    descriptor_pool_create_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_create_info.flags =
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptor_pool_create_info.poolSizeCount =
        actual_descriptor_pool_sizes_count;
    descriptor_pool_create_info.pPoolSizes = descriptor_pool_sizes;
    descriptor_pool_create_info.maxSets = max_descriptor_sets;
    CHECK_VKRESULT_RET(vkCreateDescriptorPool(
        device, &descriptor_pool_create_info, NULL, descriptor_pool));
    return sccl_success;
}

/**
 * Check for duplicate constant ids.
 */
static bool verify_specialization_constants(
    const sccl_shader_specialization_constant_t *specialization_constants,
    size_t specialization_constants_count)
{
    for (size_t i = 0; i < specialization_constants_count; ++i) {
        for (size_t j = 0; j < specialization_constants_count; ++j) {
            if (specialization_constants[j].constant_id ==
                    specialization_constants[i].constant_id &&
                j != i) {
                return false;
            }
        }
    }
    return true;
}

sccl_error_t sccl_create_shader(const sccl_device_t device,
                                sccl_shader_t *shader,
                                const sccl_shader_config_t *config)
{
    sccl_error_t error;

    /* validate config */
    CHECK_SCCL_NULL_RET(config);
    CHECK_SCCL_NULL_RET(config->shader_source_code);
    if (config->shader_source_code_length <= 0) {
        return sccl_invalid_argument;
    }

    /* create internal handle */
    struct sccl_shader *shader_internal = NULL;
    CHECK_SCCL_ERROR_GOTO(
        sccl_calloc((void **)&shader_internal, 1, sizeof(struct sccl_shader)),
        error_return, error);

    shader_internal->device = device->device;

    /* create shader module */
    VkShaderModuleCreateInfo shader_module_create_info = {0};
    shader_module_create_info.sType =
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_module_create_info.codeSize = config->shader_source_code_length;
    shader_module_create_info.pCode =
        (const uint32_t *)config->shader_source_code;
    CHECK_VKRESULT_GOTO(
        vkCreateShaderModule(device->device, &shader_module_create_info,
                             VK_NULL_HANDLE, &shader_internal->shader_module),
        error_return, error);

    /* create descriptor set layout based on provided config */
    if (config->buffer_layouts != NULL) {
        CHECK_SCCL_ERROR_GOTO(
            create_descriptor_set_layouts(
                shader_internal->device, config->buffer_layouts,
                config->buffer_layouts_count,
                &shader_internal->descriptor_set_layouts,
                &shader_internal->descriptor_set_layouts_count),
            error_return, error);
    }

    /* create descriptor pool */
    size_t storage_buffer_count = 0;
    size_t uniform_buffer_count = 0;
    count_buffer_types(config->buffer_layouts, config->buffer_layouts_count,
                       &storage_buffer_count, &uniform_buffer_count);
    if (storage_buffer_count > 0 || uniform_buffer_count > 0) {
        CHECK_SCCL_ERROR_GOTO(create_descriptor_pool(
                                  shader_internal->device, storage_buffer_count,
                                  uniform_buffer_count,
                                  shader_internal->descriptor_set_layouts_count,
                                  &shader_internal->descriptor_pool),
                              error_return, error);
    }

    /* prepare specialization info */
    /* verify */
    if (!verify_specialization_constants(
            config->specialization_constants,
            config->specialization_constants_count)) {
        error = sccl_invalid_argument;
        goto error_return;
    }
    /* count total data size (in bytes) */
    size_t total_specialization_constant_size = 0;
    for (size_t i = 0; i < config->specialization_constants_count; ++i) {
        total_specialization_constant_size +=
            config->specialization_constants[i].size;
    }
    /* allocate specialization data memory and entries */
    void *specialization_data = NULL;
    CHECK_SCCL_ERROR_GOTO(sccl_calloc(&specialization_data,
                                      total_specialization_constant_size,
                                      sizeof(uint8_t)),
                          error_return, error);
    VkSpecializationMapEntry *specialization_map_entries = NULL;
    CHECK_SCCL_ERROR_GOTO(sccl_calloc((void **)&specialization_map_entries,
                                      config->specialization_constants_count,
                                      sizeof(VkSpecializationMapEntry)),
                          error_return, error);
    /* init entries and copy data */
    size_t offset = 0;
    for (size_t i = 0; i < config->specialization_constants_count; ++i) {
        sccl_shader_specialization_constant_t *constant =
            &config->specialization_constants[i];
        memcpy((uint8_t *)specialization_data + offset, constant->data,
               constant->size);
        specialization_map_entries[i].constantID = constant->constant_id;
        specialization_map_entries[i].offset = offset;
        specialization_map_entries[i].size = constant->size;
        offset += constant->size;
    }
    VkSpecializationInfo specialization_info = {};
    specialization_info.mapEntryCount = config->specialization_constants_count;
    specialization_info.pMapEntries = specialization_map_entries;
    specialization_info.dataSize = total_specialization_constant_size;
    specialization_info.pData = specialization_data;

    /* prepare push constants */
    /* create push constant range for each entry */
    VkPushConstantRange *push_constant_ranges = NULL;
    if (config->push_constant_layouts_count > 0) {
        CHECK_SCCL_ERROR_GOTO(sccl_calloc((void **)&push_constant_ranges,
                                          config->push_constant_layouts_count,
                                          sizeof(VkPushConstantRange)),
                              error_return, error);
        for (size_t i = 0; i < config->push_constant_layouts_count; ++i) {
            push_constant_ranges[i].offset = 0;
            push_constant_ranges[i].size =
                config->push_constant_layouts[i].size;
            push_constant_ranges[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        /* store push constant sizes so we can know offsets when creating
         * command buffer */
        shader_internal->push_constant_layouts_count =
            config->push_constant_layouts_count;
        CHECK_SCCL_ERROR_GOTO(
            sccl_calloc((void **)&shader_internal->push_constant_layouts,
                        shader_internal->push_constant_layouts_count,
                        sizeof(sccl_shader_push_constant_layout_t)),
            error_return, error);
        memcpy(shader_internal->push_constant_layouts,
               config->push_constant_layouts,
               shader_internal->push_constant_layouts_count *
                   sizeof(sccl_shader_push_constant_layout_t));
    }

    /* create compute pipeline */
    VkPipelineLayoutCreateInfo pipeline_layout_create_info = {0};
    pipeline_layout_create_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_create_info.pSetLayouts =
        shader_internal->descriptor_set_layouts;
    pipeline_layout_create_info.setLayoutCount =
        shader_internal->descriptor_set_layouts_count;
    pipeline_layout_create_info.pPushConstantRanges = push_constant_ranges;
    pipeline_layout_create_info.pushConstantRangeCount =
        config->push_constant_layouts_count;
    CHECK_VKRESULT_GOTO(
        vkCreatePipelineLayout(device->device, &pipeline_layout_create_info,
                               NULL, &shader_internal->pipeline_layout),
        error_return, error);

    VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info = {0};
    pipeline_shader_stage_create_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_shader_stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_shader_stage_create_info.module = shader_internal->shader_module;
    pipeline_shader_stage_create_info.pName = "main";
    pipeline_shader_stage_create_info.pSpecializationInfo =
        &specialization_info;

    VkComputePipelineCreateInfo compute_pipeline_create_info = {0};
    compute_pipeline_create_info.sType =
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_pipeline_create_info.layout = shader_internal->pipeline_layout;
    compute_pipeline_create_info.stage = pipeline_shader_stage_create_info;
    CHECK_VKRESULT_GOTO(
        vkCreateComputePipelines(device->device, NULL, 1,
                                 &compute_pipeline_create_info, NULL,
                                 &shader_internal->compute_pipeline),
        error_return, error);

    /* set public handle */
    *shader = (sccl_shader_t)shader_internal;

    /* cleanup */
    sccl_free(push_constant_ranges);
    sccl_free(specialization_map_entries);
    sccl_free(specialization_data);

    return sccl_success;

error_return:
    if (push_constant_ranges != NULL) {
        sccl_free(push_constant_ranges);
    }
    if (specialization_map_entries != NULL) {
        sccl_free(specialization_map_entries);
    }
    if (specialization_data != NULL) {
        sccl_free(specialization_data);
    }

    if (shader_internal != NULL) {
        if (shader_internal->compute_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device->device, shader_internal->compute_pipeline,
                              NULL);
        }
        if (shader_internal->pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device->device,
                                    shader_internal->pipeline_layout, NULL);
        }
        if (shader_internal->push_constant_layouts != NULL) {
            sccl_free(shader_internal->push_constant_layouts);
        }
        if (shader_internal->descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device->device,
                                    shader_internal->descriptor_pool, NULL);
        }
        if (shader_internal->descriptor_set_layouts != NULL) {
            for (size_t i = 0;
                 i < shader_internal->descriptor_set_layouts_count; ++i) {
                vkDestroyDescriptorSetLayout(
                    device->device, shader_internal->descriptor_set_layouts[i],
                    NULL);
            }
            sccl_free(shader_internal->descriptor_set_layouts);
        }
        if (shader_internal->shader_module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device->device,
                                  shader_internal->shader_module, NULL);
        }
        sccl_free(shader_internal);
    }

    return error;
}

void sccl_destroy_shader(sccl_shader_t shader)
{
    vkDestroyPipeline(shader->device, shader->compute_pipeline, NULL);
    vkDestroyPipelineLayout(shader->device, shader->pipeline_layout, NULL);

    if (shader->push_constant_layouts != NULL) {
        sccl_free(shader->push_constant_layouts);
    }

    if (shader->descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(shader->device, shader->descriptor_pool, NULL);
    }
    if (shader->descriptor_set_layouts != NULL) {
        for (size_t i = 0; i < shader->descriptor_set_layouts_count; ++i) {
            vkDestroyDescriptorSetLayout(
                shader->device, shader->descriptor_set_layouts[i], NULL);
        }
        sccl_free(shader->descriptor_set_layouts);
    }

    vkDestroyShaderModule(shader->device, shader->shader_module, NULL);

    sccl_free(shader);
}

sccl_error_t sccl_run_shader(const sccl_stream_t stream,
                             const sccl_shader_t shader,
                             const sccl_shader_run_params_t *params)
{
    sccl_error_t error = sccl_success;

    /* validate */
    /* check if push constants are in range */
    if (params->push_constant_bindings_count >
        shader->push_constant_layouts_count) {
        return sccl_invalid_argument;
    }

    /* allocate descriptor set, store in stream, will be freed when stream is
     * complete */
    VkDescriptorSet *descriptor_sets = NULL;
    CHECK_SCCL_ERROR_GOTO(sccl_calloc((void **)&descriptor_sets,
                                      shader->descriptor_set_layouts_count,
                                      sizeof(VkDescriptorSet)),
                          error_return, error);
    for (size_t i = 0; i < shader->descriptor_set_layouts_count; ++i) {
        VkDescriptorSetAllocateInfo alloc_info = {0};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = shader->descriptor_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &shader->descriptor_set_layouts[i];
        CHECK_VKRESULT_GOTO(vkAllocateDescriptorSets(stream->device->device,
                                                     &alloc_info,
                                                     &descriptor_sets[i]),
                            error_return, error);
    }
    /* add to stream after all sets are allocated, so we don't have to clean up
     * in case one allocation fails */
    for (size_t i = 0; i < shader->descriptor_set_layouts_count; ++i) {
        CHECK_SCCL_ERROR_GOTO(
            add_descriptor_set_to_stream(stream, shader->descriptor_pool,
                                         descriptor_sets[i]),
            error_return, error);
    }

    /* update descriptor sets */
    VkWriteDescriptorSet *write_descriptor_sets = NULL;
    CHECK_SCCL_ERROR_GOTO(sccl_calloc((void **)&write_descriptor_sets,
                                      params->buffer_bindings_count,
                                      sizeof(VkWriteDescriptorSet)),
                          error_return, error);
    VkDescriptorBufferInfo *descriptor_buffer_infos = NULL;
    CHECK_SCCL_ERROR_GOTO(sccl_calloc((void **)&descriptor_buffer_infos,
                                      params->buffer_bindings_count,
                                      sizeof(VkDescriptorBufferInfo)),
                          error_return, error);
    for (size_t i = 0; i < params->buffer_bindings_count; ++i) {
        descriptor_buffer_infos[i].buffer =
            params->buffer_bindings[i].buffer->buffer;
        descriptor_buffer_infos[i].offset = 0;
        descriptor_buffer_infos[i].range = VK_WHOLE_SIZE;

        write_descriptor_sets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_descriptor_sets[i].dstSet =
            descriptor_sets[params->buffer_bindings[i]
                                .position.set]; /* descriptor sets should always
                                                   be in order and contiguous */
        write_descriptor_sets[i].dstBinding =
            params->buffer_bindings[i].position.binding;
        write_descriptor_sets[i].dstArrayElement = 0;
        write_descriptor_sets[i].descriptorType =
            sccl_buffer_type_to_vk_descriptor_type(
                params->buffer_bindings[i].buffer->type);
        write_descriptor_sets[i].descriptorCount = 1;
        write_descriptor_sets[i].pBufferInfo = &descriptor_buffer_infos[i];
    }
    vkUpdateDescriptorSets(stream->device->device,
                           params->buffer_bindings_count, write_descriptor_sets,
                           0, NULL);

    /* bind the compute pipeline */
    vkCmdBindPipeline(stream->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      shader->compute_pipeline);

    /* bind descriptor sets */
    if (shader->descriptor_set_layouts_count > 0) {
        vkCmdBindDescriptorSets(
            stream->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            shader->pipeline_layout, 0, shader->descriptor_set_layouts_count,
            descriptor_sets, 0, NULL);
    }

    /* push constants */
    if (params->push_constant_bindings_count > 0) {
        size_t offset = 0;
        for (size_t i = 0; i < params->push_constant_bindings_count; ++i) {
            vkCmdPushConstants(stream->command_buffer, shader->pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT, offset,
                               shader->push_constant_layouts[i].size,
                               params->push_constant_bindings[i].data);
            offset += shader->push_constant_layouts[i].size;
        }
    }

    /* dispatch the compute shader */
    vkCmdDispatch(stream->command_buffer, params->group_count_x,
                  params->group_count_y, params->group_count_z);

    /* create barrier so next command will wait until this is finished */
    VkMemoryBarrier memory_barrier = {0};
    memory_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memory_barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    memory_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    vkCmdPipelineBarrier(stream->command_buffer,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                         &memory_barrier, 0, NULL, 0, NULL);

    /* cleanup */
    sccl_free(descriptor_buffer_infos);
    sccl_free(write_descriptor_sets);
    sccl_free(descriptor_sets); /* vulkan handles will be destroyed when stream
                                   is done executing */

    return sccl_success;

error_return:
    if (descriptor_buffer_infos != NULL) {
        sccl_free(descriptor_buffer_infos);
    }
    if (write_descriptor_sets != NULL) {
        sccl_free(write_descriptor_sets);
    }
    if (descriptor_sets != NULL) {
        sccl_free(descriptor_sets);
    }

    return error;
}
