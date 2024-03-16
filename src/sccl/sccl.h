
#pragma once
#ifndef SCCL_HEADER
#define SCCL_HEADER

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Sivert Cubedo Compute Library (SCCL)
 * ~Sivert Cubedo General-Purpose Graphics Processing Unit Library (SCGPGPUL)~
 */

/* Error type */
typedef enum {
    sccl_success = 0,
    sccl_unhandled_vulkan_error = 1,
    sccl_system_error = 2,
    sccl_internal_error = 3,
    sccl_invalid_argument = 4,
    sccl_invalid_usage = 5
} sccl_error_t;

/* Opaque handles */
typedef struct sccl_instance *sccl_instance_t;
typedef struct sccl_device *sccl_device_t;
#define SCCL_NULL NULL

/**
 * To enable validation layers, set enviroment variable
 * `ENABLE_VALIDATION_LAYERS=1`
 */

sccl_error_t sccl_create_instance(sccl_instance_t *instance);

sccl_error_t sccl_create_device();

#ifdef __cplusplus
}
#endif

#endif // SCCL_HEADER
