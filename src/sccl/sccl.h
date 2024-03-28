
#pragma once
#ifndef SCCL_HEADER
#define SCCL_HEADER

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/**
 * Sivert Collective Compute Library (SCCL)
 * ~Sivert Collective General-Purpose Graphics Processing Unit Library
 * (SCGPGPUL)~
 */

/* Error type enum */
typedef enum {
    sccl_success = 0,
    sccl_unhandled_vulkan_error = 1,
    sccl_system_error = 2,
    sccl_internal_error = 3,
    sccl_invalid_argument = 4,
    sccl_unsupported_error = 5,
    sccl_out_of_resources_error = 6
} sccl_error_t;

/* Buffer type enum */
typedef enum {
    sccl_buffer_type_host_storage = 1,
    sccl_buffer_type_host = 1,
    sccl_buffer_type_device_storage = 2,
    sccl_buffer_type_device = 2,
    sccl_buffer_type_shared_storage = 3,
    sccl_buffer_type_shared = 3,
    sccl_buffer_type_host_uniform = 4,
    sccl_buffer_type_device_uniform = 5,
    sccl_buffer_type_shared_uniform = 6
} sccl_buffer_type_t;

typedef struct sccl_instance *sccl_instance_t; /* Opaque handle */
typedef struct sccl_device *sccl_device_t;     /* Opaque handle */
typedef struct sccl_buffer *sccl_buffer_t;     /* Opaque handle */
typedef struct sccl_stream *sccl_stream_t;     /* Opaque handle */
typedef struct sccl_shader *sccl_shader_t;     /* Opaque handle */
#define SCCL_NULL NULL

typedef struct {
    uint32_t constant_id;
    size_t size;
    void *data;
} sccl_shader_specialization_constant_t;

typedef struct {
    size_t size;
} sccl_shader_push_constant_layout_t;

typedef struct {
    /* index of constant to update, index is from order of push_constants set in
     * `sccl_create_shader` */
    size_t index;
    /* size of data is set in `sccl_shader_push_constant_layout_t`*/
    void *data;
} sccl_shader_push_constant_binding_t;

typedef struct {
    uint32_t set;
    uint32_t binding;
} sccl_shader_buffer_position_t;

typedef struct {
    sccl_shader_buffer_position_t position;
    sccl_buffer_type_t type;
} sccl_shader_buffer_layout_t;

typedef struct {
    sccl_shader_buffer_position_t position;
    sccl_buffer_t buffer;
} sccl_shader_buffer_binding_t;

typedef struct {
    char *shader_source_code;         /* required */
    size_t shader_source_code_length; /* must be larger than 0 */
    sccl_shader_specialization_constant_t
        *specialization_constants; /* optional */
    size_t specialization_constants_count;
    sccl_shader_push_constant_layout_t *push_constant_layouts; /* optional */
    size_t push_constant_layouts_count;
    sccl_shader_buffer_layout_t *buffer_layouts; /* optional */
    size_t buffer_layouts_count;
} sccl_shader_config_t;

typedef struct {
    uint32_t group_count_x;
    uint32_t group_count_y;
    uint32_t group_count_z;
    sccl_shader_buffer_binding_t
        *buffer_bindings; /* required if set in `sccl_shader_config_t` */
    size_t buffer_bindings_count;
    sccl_shader_push_constant_binding_t
        *push_constant_bindings; /* required if set in `sccl_shader_config_t` */
    size_t push_constant_bindings_count;
} sccl_shader_run_params_t;

/**
 * To enable validation layers, set enviroment variable
 * `SCCL_ENABLE_VALIDATION_LAYERS=1`
 *
 * To make library assert when encountering a validation error, set enviroment
 * variable `ASSERT_ON_VALIDATION_ERROR=1`
 */
#define SCCL_ENABLE_VALIDATION_LAYERS "SCCL_ENABLE_VALIDATION_LAYERS"
#define SCCL_ASSERT_ON_VALIDATION_ERROR "SCCL_ASSERT_ON_VALIDATION_ERROR"

sccl_error_t sccl_create_instance(sccl_instance_t *instance);

void sccl_destroy_instance(sccl_instance_t instance);

sccl_error_t sccl_get_device_count(const sccl_instance_t instance,
                                   uint32_t *device_count);

sccl_error_t sccl_create_device(const sccl_instance_t instance,
                                sccl_device_t *device, uint32_t device_index);

void sccl_destroy_device(sccl_device_t device);

sccl_error_t sccl_create_buffer(const sccl_device_t device,
                                sccl_buffer_t *buffer, sccl_buffer_type_t type,
                                size_t size);

void sccl_destroy_buffer(sccl_buffer_t buffer);

/**
 * Map buffer memory on host.
 * It's only possible to have 1 map at any time per buffer.
 * It's not possible to host map buffers of type `sccl_buffer_type_device`.
 */
sccl_error_t sccl_host_map_buffer(const sccl_buffer_t buffer, void **data,
                                  size_t offset, size_t size);

/**
 * Unmap memory on host.
 * Pointer `data` returned from `sccl_host_map_buffer` is invalid after this
 * operation. It's possible to remap memory after unmapping.
 */
void sccl_host_unmap_buffer(const sccl_buffer_t buffer);

sccl_error_t sccl_create_stream(const sccl_device_t device,
                                sccl_stream_t *stream);

void sccl_destroy_stream(sccl_stream_t stream);

sccl_error_t sccl_dispatch_stream(const sccl_stream_t stream);

sccl_error_t sccl_join_stream(const sccl_stream_t stream);

sccl_error_t sccl_copy_buffer(const sccl_stream_t stream,
                              const sccl_buffer_t src, size_t src_offset,
                              const sccl_buffer_t dst, size_t dst_offset,
                              size_t size);

sccl_error_t sccl_create_shader(const sccl_device_t device,
                                sccl_shader_t *shader,
                                const sccl_shader_config_t *config);

void sccl_destroy_shader(sccl_shader_t shader);

sccl_error_t sccl_run_shader(const sccl_shader_t shader,
                             const sccl_shader_run_params_t *params);

#ifdef __cplusplus
}
#endif

#endif // SCCL_HEADER
