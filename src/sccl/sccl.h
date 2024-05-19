
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

/**
 * Buffer type enum.
 */
typedef enum {
    sccl_buffer_type_host_storage = 1,
    sccl_buffer_type_host = 1,
    sccl_buffer_type_device_storage = 2,
    sccl_buffer_type_device = 2,
    sccl_buffer_type_shared_storage = 3,
    sccl_buffer_type_shared = 3,
    sccl_buffer_type_host_uniform = 4,
    sccl_buffer_type_device_uniform = 5,
    sccl_buffer_type_shared_uniform = 6,
    sccl_buffer_type_external = 7
} sccl_buffer_type_t;

typedef struct sccl_instance *sccl_instance_t; /* opaque handle */
typedef struct sccl_device *sccl_device_t;     /* opaque handle */
typedef struct sccl_buffer *sccl_buffer_t;     /* opaque handle */
typedef struct sccl_stream *sccl_stream_t;     /* opaque handle */
typedef struct sccl_shader *sccl_shader_t;     /* opaque handle */

typedef struct {
    uint32_t constant_id;
    size_t size;
    void *data;
} sccl_shader_specialization_constant_t;

typedef struct {
    size_t size;
} sccl_shader_push_constant_layout_t;

typedef struct {
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

#define SCCL_BIND_WHOLE_BUFFER (~0ul)

typedef struct {
    sccl_shader_buffer_position_t position;
    sccl_buffer_t buffer;
    /* offset in buffer (in bytes). Offset must be aligned according to buffer
     * type requirement, this can be queried in device properties. */
    size_t offset;
    /* Bytes to include after offset, set to `SCCL_BIND_WHOLE_BUFFER` to bind
     * whole buffer. Must be larger than 0 or `SCCL_BIND_WHOLE_BUFFER`. If wrong
     * size `sccl_invalid_argument` is returned.  */
    size_t size;
} sccl_shader_buffer_binding_t;

#define SCCL_DEFAULT_MAX_CONCURRENT_BUFFER_BINDINGS 8

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
    /* maximum number of concurrent buffer bindings for this shader, if 0 then
     * `SCCL_DEFAULT_MAX_CONCURRENT_BUFFER_BINDINGS` is used */
    size_t max_concurrent_buffer_bindings;
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
 * Struct containing various device properties queried from vulkan.
 */
typedef struct {
    /* index: `0 = x, 1 = y, 2 = z`. `maxComputeWorkGroupCount` from
     * https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceLimits.html
     */
    uint32_t max_work_group_count[3];
    /* index: `0 = x, 1 = y, 2 = z`. `maxComputeWorkGroupSize` from
     * https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceLimits.html
     */
    uint32_t max_work_group_size[3];
    /* `maxComputeWorkGroupInvocations` from
     * https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceLimits.html
     */
    uint32_t max_work_group_invocations;
    /* `subgroupSize` from
     * https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceSubgroupProperties.html
     */
    uint32_t native_work_group_size;
    /* `maxStorageBufferRange` from
     * https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceLimits.html
     */
    uint32_t max_storage_buffer_size;
    /* `maxUniformBufferRange` from
     * https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceLimits.html
     */
    uint32_t max_uniform_buffer_size;
    /* `maxPushConstantsSize` from
     * https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceLimits.html
     */
    uint32_t max_push_constant_size;
    /* `minStorageBufferOffsetAlignment` from
     * https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceLimits.html
     */
    size_t min_storage_buffer_offset_alignment;
    /* `minUniformBufferOffsetAlignment` from
     * https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceLimits.html
     */
    size_t min_uniform_buffer_offset_alignment;
    /* `minImportedHostPointerAlignment` from
     * https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceExternalMemoryHostPropertiesEXT.html
     */
    size_t min_external_buffer_host_pointer_alignment;
} sccl_device_properties_t;

/**
 * To enable validation layers, set enviroment variable
 * `SCCL_ENABLE_VALIDATION_LAYERS=1`
 *
 * To make library assert when encountering a validation error, set enviroment
 * variable `ASSERT_ON_VALIDATION_ERROR=1`
 */
#define SCCL_ENABLE_VALIDATION_LAYERS "SCCL_ENABLE_VALIDATION_LAYERS"
#define SCCL_ASSERT_ON_VALIDATION_ERROR "SCCL_ASSERT_ON_VALIDATION_ERROR"

/**
 * Get error string.
 * @param error Error.
 */
const char *sccl_get_error_string(sccl_error_t error);

/**
 * Create instance.
 * @param instance Instance.
 */
sccl_error_t sccl_create_instance(sccl_instance_t *instance);

/**
 * Destroy instance.
 * @param instance Instance.
 */
void sccl_destroy_instance(sccl_instance_t instance);

/**
 * Get number of devices present in instance.
 * @param instance Instance.
 * @param device_count Number of devices returned.
 */
sccl_error_t sccl_get_device_count(const sccl_instance_t instance,
                                   uint32_t *device_count);

/**
 * Create device.
 * @param instance Instance.
 * @param device Device.
 * @param device_index Index of device enumerated by Vulkan.
 */
sccl_error_t sccl_create_device(const sccl_instance_t instance,
                                sccl_device_t *device, uint32_t device_index);

/**
 * Destroy device.
 * @param device Device.
 */
void sccl_destroy_device(sccl_device_t device);

/**
 * Get device properties.
 * @param device Device.
 * @param device_properties Device properties returned.
 */
void sccl_get_device_properties(const sccl_device_t device,
                                sccl_device_properties_t *device_properties);

/**
 * Create buffer non-external buffer (memory allocated by sccl).
 * @param device Device.
 * @param buffer Buffer.
 * @param type Type of buffer, see `sccl_buffer_type_t`,
 * `sccl_buffer_type_external` is not supported by this call.
 * @param size Size in bytes.
 */
sccl_error_t sccl_create_buffer(const sccl_device_t device,
                                sccl_buffer_t *buffer, sccl_buffer_type_t type,
                                size_t size);

/**
 * Pointer must be aligned to and a multiple of
 * `sccl_device_properties_t::min_external_buffer_host_pointer_alignment`.
 * Buffer handles created by this call are destroyed with `sccl_destroy_buffer`
 * Buffers created from this call will be of type `sccl_buffer_type_external`.
 * @param device Device.
 * @param buffer Buffer.
 * @param host_pointer Host pointer to register.
 * @param size Size in bytes.
 */
sccl_error_t sccl_register_host_pointer_buffer(const sccl_device_t device,
                                               sccl_buffer_t *buffer,
                                               void *host_pointer, size_t size);

/**
 * Destroy buffer.
 * @param buffer Buffer.
 */
void sccl_destroy_buffer(sccl_buffer_t buffer);

/**
 * Get buffer type.
 * @param buffer Buffer.
 */
sccl_buffer_type_t sccl_get_buffer_type(const sccl_buffer_t buffer);

/**
 * Get buffer type.
 * @param buffer Buffer.
 */
size_t sccl_get_buffer_min_offset_alignment(const sccl_buffer_t buffer);

/**
 * Map buffer memory on host.
 * It's only possible to have 1 map at any time per buffer.
 * It's not possible to host map buffers of type `sccl_buffer_type_device`.
 * @param buffer Buffer.
 * @param data Returned pointer too mapped memory.
 * @param offset Offset in buffer to map in bytes.
 * @param size Size to map in bytes.
 */
sccl_error_t sccl_host_map_buffer(const sccl_buffer_t buffer, void **data,
                                  size_t offset, size_t size);

/**
 * Unmap memory on host.
 * Pointer `data` returned from `sccl_host_map_buffer` is invalid after this
 * operation. It's possible to remap memory after unmapping.
 * @param buffer Buffer.
 */
void sccl_host_unmap_buffer(const sccl_buffer_t buffer);

/**
 * Create stream on device.
 * @param device Device.
 * @param stream Stream.
 */
sccl_error_t sccl_create_stream(const sccl_device_t device,
                                sccl_stream_t *stream);

/**
 * Destroy stream.
 * @param stream Stream.
 */
void sccl_destroy_stream(sccl_stream_t stream);

/**
 * Dispatch stream to GPU. This call prompts the device to start executing the
 * commands added to the stream. This happens asynchronously, so this call is
 * not blocking.
 * @param stream Stream.
 */
sccl_error_t sccl_dispatch_stream(const sccl_stream_t stream);

/**
 * Join stream, and reset. Wait for stream to complete executing on device. This
 * call is blocking.
 * @param stream Stream.
 */
sccl_error_t sccl_join_stream(const sccl_stream_t stream);

/**
 * Reset stream.
 * @param stream Stream.
 */
sccl_error_t sccl_reset_stream(const sccl_stream_t stream);

/**
 * Wait for streams to complete. This call is blocking. Will unblock when first
 * stream is complete. The streams are not reset if they have been signaled.
 * @param device Device owning streams, all streams must be on a single device.
 * @param streams Array of streams to wait for.
 * @param streams_count Number of streams.
 * @param completed_streams Array to signal if stream is complete or now, this
 * must be of size `streams_count`. When this call returns, non-complete streams
 * will be set to 0, while complete will be set to 1. Array indexes corresponds
 * with streams indexes.
 */
sccl_error_t sccl_wait_streams(const sccl_device_t device,
                               const sccl_stream_t *streams,
                               size_t streams_count,
                               uint8_t *completed_streams);

/**
 * Wait for all streams to complete. The streams are not reset if they have been
 * signaled.
 * @param device Device owning streams, all streams must be on a single device.
 * @param streams Array of streams to wait for.
 * @param streams_count Number of streams.
 */
sccl_error_t sccl_wait_streams_all(const sccl_device_t device,
                                   const sccl_stream_t *streams,
                                   size_t streams_count);

/**
 * Add copy buffer command to stream.
 * @param stream Stream.
 * @param src Source buffer.
 * @param src_offset Source buffer offset in bytes.
 * @param dst Destination buffer.
 * @param dst_offset Destination buffer offset in bytes.
 * @param size Size to copy in bytes.
 */
sccl_error_t sccl_copy_buffer(const sccl_stream_t stream,
                              const sccl_buffer_t src, size_t src_offset,
                              const sccl_buffer_t dst, size_t dst_offset,
                              size_t size);

/**
 * Create shader on device.
 * @param device Device.
 * @param shader Shader.
 * @param config Config describing shader layout, see `sccl_shader_config_t`.
 */
sccl_error_t sccl_create_shader(const sccl_device_t device,
                                sccl_shader_t *shader,
                                const sccl_shader_config_t *config);

/**
 * Destroy shader.
 * @param shader Shader.
 */
void sccl_destroy_shader(sccl_shader_t shader);

/**
 * Add shader run command to stream.
 * If error `sccl_out_of_resources_error` is returned, it is possible to
 * increase the limit at shader creation by setting
 * `max_concurrent_buffer_bindings` in `sccl_shader_config_t`.
 * @param stream Stream.
 * @param shader Shader.
 * @param params Params describing how to execute the shader, see
 * `sccl_shader_run_params_t`.
 */
sccl_error_t sccl_run_shader(const sccl_stream_t stream,
                             const sccl_shader_t shader,
                             const sccl_shader_run_params_t *params);

/**
 * Set `buffer_layout` and `buffer_binding` according to contents of `buffer`,
 * `set` and `binding`.
 * @param buffer Buffer.
 * @param set Set index.
 * @param binding Binding index.
 * @param buffer_layout Buffer layout.
 * @param buffer_binding Buffer binding.
 */
void sccl_set_buffer_layout_binding(
    const sccl_buffer_t buffer, uint32_t set, uint32_t binding,
    sccl_shader_buffer_layout_t *buffer_layout,
    sccl_shader_buffer_binding_t *buffer_binding);

#ifdef __cplusplus
}
#endif

#endif // SCCL_HEADER
