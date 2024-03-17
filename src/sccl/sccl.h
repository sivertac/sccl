
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

/* Error type */
typedef enum {
    sccl_success = 0,
    sccl_unhandled_vulkan_error = 1,
    sccl_system_error = 2,
    sccl_internal_error = 3,
    sccl_invalid_argument = 4,
    sccl_unsupported_error = 5,
    sccl_out_of_resources_error = 6
} sccl_error_t;

/* Opaque handles */
typedef struct sccl_instance *sccl_instance_t;
typedef struct sccl_device *sccl_device_t;
typedef struct sccl_buffer *sccl_buffer_t;
typedef struct sccl_stream *sccl_stream_t;
#define SCCL_NULL NULL

/**
 * To enable validation layers, set enviroment variable
 * `ENABLE_VALIDATION_LAYERS=1`
 */

sccl_error_t sccl_create_instance(sccl_instance_t *instance);

void sccl_destroy_instance(sccl_instance_t instance);

sccl_error_t sccl_get_device_count(const sccl_instance_t instance,
                                   uint32_t *device_count);

sccl_error_t sccl_create_device(const sccl_instance_t instance,
                                sccl_device_t *device, uint32_t device_index);

void sccl_destroy_device(sccl_device_t device);

typedef enum {
    sccl_buffer_type_host = 1,
    sccl_buffer_type_device = 2,
    sccl_buffer_type_shared = 3
} sccl_buffer_type_t;

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

#ifdef __cplusplus
}
#endif

#endif // SCCL_HEADER
