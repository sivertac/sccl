
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

/**
 * @brief Enum representing error types in the SCCL API.
 *
 * This enum defines the possible error types that can occur while using the
 * SCCL API. Each error type corresponds to a specific error condition
 * encountered during the execution of SCCL functions.
 */
typedef enum {
    sccl_success =
        0, /**< Indicates that the operation completed successfully. */
    sccl_unhandled_vulkan_error =
        1,                 /**< Indicates an unhandled Vulkan error occurred. */
    sccl_system_error = 2, /**< Indicates a system error occurred. */
    sccl_internal_error =
        3, /**< Indicates an internal error occurred within the SCCL library. */
    sccl_invalid_argument =
        4, /**< Indicates an invalid argument was passed to a SCCL function. */
    sccl_unsupported_error =
        5, /**< Indicates an unsupported operation or feature was requested. */
    sccl_out_of_resources_error = 6 /**< Indicates that the operation failed due
                                       to resource limitations. */
} sccl_error_t;

/**
 * @brief Enum representing buffer types in the SCCL API.
 *
 * This enum defines the possible types of buffers that can be used in the SCCL
 * API. Each buffer type corresponds to a specific type of memory storage or
 * usage within the SCCL library.
 */
typedef enum {
    sccl_buffer_type_host_storage =
        1,                     /**< Buffer type for host storage memory. */
    sccl_buffer_type_host = 1, /**< Alias for sccl_buffer_type_host_storage. */
    sccl_buffer_type_device_storage =
        2, /**< Buffer type for device storage memory. */
    sccl_buffer_type_device =
        2, /**< Alias for sccl_buffer_type_device_storage. */
    sccl_buffer_type_shared_storage =
        3, /**< Buffer type for shared storage memory. */
    sccl_buffer_type_shared =
        3, /**< Alias for sccl_buffer_type_shared_storage. */
    sccl_buffer_type_host_uniform =
        4, /**< Buffer type for host uniform memory. */
    sccl_buffer_type_device_uniform =
        5, /**< Buffer type for device uniform memory. */
    sccl_buffer_type_shared_uniform =
        6, /**< Buffer type for shared uniform memory. */
    sccl_buffer_type_external =
        7 /**< Buffer type for externally allocated memory. */
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
 * @brief Retrieve a human-readable error message for a given error code.
 *
 * This function returns a string describing the error corresponding to the
 * provided error code. The string is suitable for logging or displaying
 * error messages to users.
 *
 * @param error The error code of type `sccl_error_t` for which to retrieve
 *              the error message.
 *
 * @return A constant character pointer to a null-terminated string
 *         describing the error. The returned string should not be modified
 *         or freed by the caller.
 */
const char *sccl_get_error_string(sccl_error_t error);

/**
 * @brief Create and initialize an SCCL instance.
 *
 * This function allocates and initializes an instance of the SCCL context.
 * It sets up all necessary resources and prepares the instance for further
 * operations. The caller must ensure that the provided instance pointer
 * is valid and that the instance is properly released using the appropriate
 * cleanup function when it is no longer needed.
 *
 * @param[out] instance A pointer to an `sccl_instance_t` structure that will
 *                      be initialized by this function. This parameter
 *                      cannot be NULL.
 *
 * @return An `sccl_error_t` code indicating the success or failure of the
 *         instance creation.
 */
sccl_error_t sccl_create_instance(sccl_instance_t *instance);

/**
 * @brief Destroy and clean up an SCCL instance.
 *
 * It should be called when the instance is no longer needed to
 * ensure proper cleanup and prevent memory leaks.
 *
 * @param[in] instance The `sccl_instance_t` instance to be destroyed. This
 *                     parameter must be a valid instance created by
 *                     `sccl_create_instance`.
 */
void sccl_destroy_instance(sccl_instance_t instance);

/**
 * @brief Retrieve the number of devices present in the given SCCL instance.
 *
 * This function obtains the number of devices associated with the specified
 * SCCL instance and returns this count through the provided pointer.
 *
 * @param[in] instance The `sccl_instance_t` instance for which to get the
 *                     device count. This parameter must be a valid instance
 *                     created by `sccl_create_instance`.
 * @param[out] device_count A pointer to a uint32_t where the number of devices
 *                          will be stored. This parameter cannot be NULL.
 *
 * @return An `sccl_error_t` code indicating the success or failure of the
 *         operation.
 */
sccl_error_t sccl_get_device_count(const sccl_instance_t instance,
                                   uint32_t *device_count);

/**
 * @brief Create and initialize a device within the given SCCL instance.
 *
 * This function creates a device associated with the specified SCCL instance,
 * initializing it for use. The device is identified by its index as enumerated
 * by Vulkan.
 *
 * @param[in] instance The `sccl_instance_t` instance to which the device will
 *                     be associated. This parameter must be a valid instance
 *                     created by `sccl_create_instance`.
 * @param[out] device A pointer to an `sccl_device_t` structure that will be
 *                    initialized by this function. This parameter cannot be
 * NULL.
 * @param[in] device_index The index of the device as enumerated by Vulkan. This
 *                         index identifies which device to create within the
 * instance.
 *
 * @return An `sccl_error_t` code indicating the success or failure of the
 *         device creation.
 */
sccl_error_t sccl_create_device(const sccl_instance_t instance,
                                sccl_device_t *device, uint32_t device_index);

/**
 * @brief Destroy and clean up the specified device.
 *
 * It should be called when the device is no longer needed to ensure
 * proper cleanup and prevent resource leaks.
 *
 * @param[in] device The `sccl_device_t` device to be destroyed. This parameter
 *                   must be a valid device created by `sccl_create_device`.
 */
void sccl_destroy_device(sccl_device_t device);

/**
 * @brief Retrieve the properties of the specified device.
 *
 * This function obtains various properties of the given SCCL device and
 * stores them in the provided `sccl_device_properties_t` structure.
 *
 * @param[in] device The `sccl_device_t` device for which properties are being
 * retrieved. This parameter must be a valid device created by
 * `sccl_create_device`.
 * @param[out] device_properties A pointer to an `sccl_device_properties_t`
 * structure where the properties of the device will be stored. This parameter
 * cannot be NULL.
 */
void sccl_get_device_properties(const sccl_device_t device,
                                sccl_device_properties_t *device_properties);

/**
 * @brief Create a buffer allocated by SCCL on the specified device.
 *
 * This function creates a buffer on the given SCCL device with the specified
 * size. The buffer can be of different types, as defined by
 * `sccl_buffer_type_t`. Note that this function does not support creating
 * external buffers; use `sccl_register_host_pointer_buffer` for that purpose.
 *
 * @param[in] device The `sccl_device_t` device on which the buffer will be
 * created. This parameter must be a valid device created by
 * `sccl_create_device`.
 * @param[out] buffer A pointer to an `sccl_buffer_t` structure that will be
 * initialized by this function. This parameter cannot be NULL.
 * @param[in] type The type of buffer to create. See `sccl_buffer_type_t` for
 * available types.
 * @param[in] size The size of the buffer to be allocated, in bytes.
 *
 * @return An `sccl_error_t` code indicating the success or failure of the
 * buffer creation.
 */
sccl_error_t sccl_create_buffer(const sccl_device_t device,
                                sccl_buffer_t *buffer, sccl_buffer_type_t type,
                                size_t size);

/**
 * @brief Register a host pointer as an external buffer with the specified
 * device.
 *
 * This function registers a host pointer as an external buffer with the given
 * SCCL device. The provided host pointer must be aligned to and a multiple of
 * the minimum external buffer host pointer alignment specified in the device
 * properties. Buffers created by this call will be of type
 * `sccl_buffer_type_external`.
 *
 * @note Buffer handles created by this call must be destroyed using
 * `sccl_destroy_buffer`.
 *
 * @param[in] device The `sccl_device_t` device with which to register the host
 * pointer buffer. This parameter must be a valid device created by
 * `sccl_create_device`.
 * @param[out] buffer A pointer to an `sccl_buffer_t` structure that will be
 * initialized by this function. This parameter cannot be NULL.
 * @param[in] host_pointer The host pointer to be registered as an external
 * buffer, must be aligned to
 * `sccl_device_properties_t::min_external_buffer_host_pointer_alignment`.
 * @param[in] size The size of the buffer to be registered, in bytes, must be
 * multiple of
 * `sccl_device_properties_t::min_external_buffer_host_pointer_alignment`.
 *
 * @return An `sccl_error_t` code indicating the success or failure of the
 * buffer registration. `sccl_unsupported_error` is returned if device does not
 * support host pointer buffers.
 */
sccl_error_t sccl_create_host_pointer_buffer(const sccl_device_t device,
                                             sccl_buffer_t *buffer,
                                             void *host_pointer, size_t size);

/**
 * @return An `sccl_error_t` code indicating the success or failure of the
 * buffer creation. `sccl_unsupported_error` is returned if device does not
 * support dmabuf buffers.
 */
sccl_error_t sccl_create_dmabuf_buffer(const sccl_device_t device,
                                       sccl_buffer_t *buffer,
                                       sccl_buffer_type_t type, size_t size);

/**
 * @note `out_fd` has to be closed when cleaning up.
 */
sccl_error_t sccl_export_dmabuf_buffer(const sccl_buffer_t buffer, int *out_fd);

sccl_error_t sccl_import_dmabuf_buffer(const sccl_device_t device,
                                       sccl_buffer_t *buffer, int in_fd,
                                       sccl_buffer_type_t type, size_t size);

/**
 * @brief Destroy the specified buffer.
 *
 * This function releases resources associated with the given buffer. It should
 * be called when the buffer is no longer needed to ensure proper cleanup and
 * prevent resource leaks.
 *
 * @param[in] buffer The `sccl_buffer_t` buffer to be destroyed. This parameter
 *                   must be a valid buffer created by `sccl_create_buffer` or
 *                   `sccl_register_host_pointer_buffer`.
 */
void sccl_destroy_buffer(sccl_buffer_t buffer);

/**
 * @brief Retrieve the type of the specified buffer.
 *
 * This function returns the type of the given buffer.
 *
 * @param[in] buffer The `sccl_buffer_t` buffer for which to retrieve the type.
 *                   This parameter must be a valid buffer.
 *
 * @return The type of the buffer, as a value of type `sccl_buffer_type_t`.
 */
sccl_buffer_type_t sccl_get_buffer_type(const sccl_buffer_t buffer);

/**
 * @brief Retrieve the minimum required alignment for buffer offsets.
 *
 * This function returns the minimum required alignment for buffer offsets
 * associated with the given buffer. The alignment specifies the minimum
 * number of bytes by which an offset into the buffer must be aligned.
 *
 * @param[in] buffer The `sccl_buffer_t` buffer for which to retrieve the
 *                   minimum offset alignment. This parameter must be a valid
 * buffer.
 *
 * @return The minimum required alignment for buffer offsets, in bytes.
 */
size_t sccl_get_buffer_min_offset_alignment(const sccl_buffer_t buffer);

/**
 * @brief Map buffer memory on the host.
 *
 * This function maps a portion of the buffer memory onto the host, allowing
 * access to the data from the CPU. Only one map operation can be active at
 * any time per buffer. It is not possible to map buffers of type
 * `sccl_buffer_type_device`.
 *
 * @param[in] buffer The `sccl_buffer_t` buffer to map on the host.
 *                   This parameter must be a valid buffer.
 * @param[out] data A pointer to a pointer where the mapped memory will be
 *                  returned. This parameter cannot be NULL.
 * @param[in] offset The offset within the buffer from which to start the
 * mapping, in bytes.
 * @param[in] size The size of the memory region to map, in bytes.
 *
 * @return An `sccl_error_t` code indicating the success or failure of the
 *         mapping operation.
 */
sccl_error_t sccl_host_map_buffer(const sccl_buffer_t buffer, void **data,
                                  size_t offset, size_t size);

/**
 * @brief Unmap memory on the host.
 *
 * This function unmaps memory previously mapped on the host using
 * `sccl_host_map_buffer`. The pointer returned by `sccl_host_map_buffer`
 * becomes invalid after this operation. It is possible to remap memory
 * after unmapping.
 *
 * @param[in] buffer The `sccl_buffer_t` buffer for which to unmap memory.
 *                   This parameter must be a valid buffer.
 */
void sccl_host_unmap_buffer(const sccl_buffer_t buffer);

/**
 * @brief Create a stream on the specified device.
 *
 * This function creates a stream on the given SCCL device. Streams provide
 * a mechanism for asynchronous execution of operations on the device.
 *
 * @param[in] device The `sccl_device_t` device on which to create the stream.
 *                   This parameter must be a valid device created by
 * `sccl_create_device`.
 * @param[out] stream A pointer to an `sccl_stream_t` structure that will be
 *                    initialized by this function. This parameter cannot be
 * NULL.
 *
 * @return An `sccl_error_t` code indicating the success or failure of the
 *         stream creation.
 */
sccl_error_t sccl_create_stream(const sccl_device_t device,
                                sccl_stream_t *stream);

/**
 * @brief Destroy the specified stream.
 *
 * This function releases resources associated with the given stream. It should
 * be called when the stream is no longer needed to ensure proper cleanup and
 * prevent resource leaks.
 *
 * @param[in] stream The `sccl_stream_t` stream to be destroyed. This parameter
 *                   must be a valid stream created by `sccl_create_stream`.
 */
void sccl_destroy_stream(sccl_stream_t stream);

/**
 * @brief Dispatch commands in the specified stream to the GPU.
 *
 * This function prompts the device associated with the stream to start
 * executing the commands that have been added to the stream. The execution
 * happens asynchronously, so this call is non-blocking.
 *
 * @param[in] stream The `sccl_stream_t` stream to be dispatched for execution.
 *                   This parameter must be a valid stream created by
 * `sccl_create_stream`.
 *
 * @return An `sccl_error_t` code indicating the success or failure of the
 *         stream dispatch operation.
 */
sccl_error_t sccl_dispatch_stream(const sccl_stream_t stream);

/**
 * @brief Wait for the specified stream to complete executing on the device.
 *
 * This function blocks until all commands in the given stream have completed
 * execution on the device. It also resets the stream for future use.
 *
 * @param[in] stream The `sccl_stream_t` stream to wait for and reset.
 *                   This parameter must be a valid stream created by
 * `sccl_create_stream`.
 *
 * @return An `sccl_error_t` code indicating the success or failure of the
 *         stream join operation.
 */
sccl_error_t sccl_join_stream(const sccl_stream_t stream);

/**
 * @brief Reset the specified stream for future use.
 *
 * This function resets the specified stream, allowing it to be reused for
 * future operations. Any pending commands in the stream are discarded.
 *
 * @param[in] stream The `sccl_stream_t` stream to be reset.
 *                   This parameter must be a valid stream created by
 * `sccl_create_stream`.
 *
 * @return An `sccl_error_t` code indicating the success or failure of the
 *         stream reset operation.
 */
sccl_error_t sccl_reset_stream(const sccl_stream_t stream);

/**
 * @brief Wait for multiple streams to complete execution on the device.
 *
 * This function blocks until 1 or more specified streams have completed
 * execution on the device. It signals the completion status of each stream in
 * the `completed_streams` array. When this function returns, non-complete
 * streams will be marked as 0, while complete streams will be marked as 1 in
 * the `completed_streams` array. The order of stream completion status
 * corresponds to the order of streams in the `streams` array. Streams are not
 * reset when they are signaled.
 *
 * @param[in] device The `sccl_device_t` device that owns the streams. All
 *                   streams must belong to this single device.
 *                   This parameter must be a valid device created by
 * `sccl_create_device`.
 * @param[in] streams An array of `sccl_stream_t` streams to wait for.
 *                    This parameter must point to a valid array of streams.
 * @param[in] streams_count The number of streams in the `streams` array.
 * @param[out] completed_streams An array to signal the completion status of
 *                               each stream. This array must be of size
 * `streams_count`. When this function returns, non-complete streams will be set
 * to 0, while complete streams will be set to 1.
 * @return An `sccl_error_t` code indicating the success or failure of the
 *         operation.
 */
sccl_error_t sccl_wait_streams(const sccl_device_t device,
                               const sccl_stream_t *streams,
                               size_t streams_count,
                               uint8_t *completed_streams);

/**
 * @brief Wait for all specified streams to complete execution on the device.
 *
 * This function blocks until all specified streams have completed execution
 * on the device. The streams are not reset if they
 * have been signaled.
 *
 * @param[in] device The `sccl_device_t` device that owns the streams. All
 *                   streams must belong to this single device.
 *                   This parameter must be a valid device created by
 * `sccl_create_device`.
 * @param[in] streams An array of `sccl_stream_t` streams to wait for.
 *                    This parameter must point to a valid array of streams.
 * @param[in] streams_count The number of streams in the `streams` array.
 *
 * @return An `sccl_error_t` code indicating the success or failure of the
 *         operation.
 */
sccl_error_t sccl_wait_streams_all(const sccl_device_t device,
                                   const sccl_stream_t *streams,
                                   size_t streams_count);

/**
 * @brief Add a copy buffer command to the specified stream.
 *
 * This function adds a command to the given stream to copy data from the source
 * buffer to the destination buffer. The specified sizes and offsets determine
 * the portion of data to copy. The copy operation is asynchronous and will be
 * executed on the device when the stream is dispatched.
 *
 * @param[in] stream The `sccl_stream_t` stream to which to add the copy
 * command. This parameter must be a valid stream created by
 * `sccl_create_stream`.
 * @param[in] src The source `sccl_buffer_t` buffer from which to copy data.
 *                This parameter must be a valid buffer.
 * @param[in] src_offset The offset within the source buffer from which to start
 *                       copying data, in bytes.
 * @param[in] dst The destination `sccl_buffer_t` buffer to which to copy data.
 *                This parameter must be a valid buffer.
 * @param[in] dst_offset The offset within the destination buffer at which to
 *                       start storing copied data, in bytes.
 * @param[in] size The size of the data to copy, in bytes.
 *
 * @return An `sccl_error_t` code indicating the success or failure of the
 *         copy operation.
 */
sccl_error_t sccl_copy_buffer(const sccl_stream_t stream,
                              const sccl_buffer_t src, size_t src_offset,
                              const sccl_buffer_t dst, size_t dst_offset,
                              size_t size);

/**
 * @brief Create a shader on the specified device.
 *
 * This function creates a shader on the given SCCL device using the provided
 * shader configuration. The shader configuration specifies the layout and
 * properties of the shader. The created shader can be used for executing
 * compute operations on the device.
 *
 * @param[in] device The `sccl_device_t` device on which to create the shader.
 *                   This parameter must be a valid device created by
 * `sccl_create_device`.
 * @param[out] shader A pointer to an `sccl_shader_t` structure that will be
 *                    initialized by this function. This parameter cannot be
 * NULL.
 * @param[in] config A pointer to an `sccl_shader_config_t` structure that
 *                   describes the layout and properties of the shader.
 *                   See `sccl_shader_config_t` for more details.
 *
 * @return An `sccl_error_t` code indicating the success or failure of the
 *         shader creation.
 */
sccl_error_t sccl_create_shader(const sccl_device_t device,
                                sccl_shader_t *shader,
                                const sccl_shader_config_t *config);

/**
 * @brief Destroy the specified shader.
 *
 * This function releases resources associated with the given shader. It should
 * be called when the shader is no longer needed to ensure proper cleanup and
 * prevent resource leaks.
 *
 * @param[in] shader The `sccl_shader_t` shader to be destroyed. This parameter
 *                   must be a valid shader created by `sccl_create_shader`.
 */
void sccl_destroy_shader(sccl_shader_t shader);

/**
 * @brief Add a shader run command to the specified stream.
 *
 * This function adds a command to the given stream to execute the specified
 * shader with the provided parameters. The shader run parameters define how the
 * shader should be executed. The command is asynchronous and will be executed
 * on the device when the stream is dispatched.
 *
 * If the error `sccl_out_of_resources_error` is returned, it is possible to
 * increase the limit at shader creation by setting the
 * `max_concurrent_buffer_bindings` field in the `sccl_shader_config_t`
 * structure.
 *
 * @param[in] stream The `sccl_stream_t` stream to which to add the shader run
 * command. This parameter must be a valid stream created by
 * `sccl_create_stream`.
 * @param[in] shader The `sccl_shader_t` shader to execute. This parameter must
 * be a valid shader created by `sccl_create_shader`.
 * @param[in] params A pointer to an `sccl_shader_run_params_t` structure that
 *                   describes how to execute the shader. See
 * `sccl_shader_run_params_t` for more details.
 *
 * @return An `sccl_error_t` code indicating the success or failure of the
 *         shader run command.
 */
sccl_error_t sccl_run_shader(const sccl_stream_t stream,
                             const sccl_shader_t shader,
                             const sccl_shader_run_params_t *params);

/**
 * @brief Set `buffer_layout` and `buffer_binding` according to the contents of
 * `buffer`.
 *
 * This function sets the buffer layout and buffer binding according to the
 * contents of the specified buffer, set index, and binding index. The buffer
 * layout and binding will be updated to match the layout and binding of the
 * buffer at the specified indices.
 *
 * @param[in] buffer The buffer for which to set the layout and binding.
 *                   This parameter must be a valid buffer.
 * @param[in] set The set index to which the buffer belongs.
 * @param[in] binding The binding index to which the buffer is bound.
 * @param[out] buffer_layout The buffer layout to set according to the contents
 * of the buffer. This parameter must be a valid buffer layout structure.
 * @param[out] buffer_binding The buffer binding to set according to the
 * contents of the buffer. This parameter must be a valid buffer binding
 * structure.
 */
void sccl_set_buffer_layout_binding(
    const sccl_buffer_t buffer, uint32_t set, uint32_t binding,
    sccl_shader_buffer_layout_t *buffer_layout,
    sccl_shader_buffer_binding_t *buffer_binding);

#ifdef __cplusplus
}
#endif

#endif // SCCL_HEADER
