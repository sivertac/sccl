#include "common.hpp"
#include <gtest/gtest.h>
#include <inttypes.h>

/**
 * `external_ptr` is only set if type is `sccl_buffer_type_external`.
 * supported set to false if buffer type is not supported.
 */
void create_buffer_generic(const sccl_device_t device, sccl_buffer_t *buffer,
                           sccl_buffer_type_t type, size_t size,
                           void **external_ptr, bool *supported)
{
    sccl_error_t error = sccl_success;

    *supported = true;

    switch (type) {
    case sccl_buffer_type_host_storage:
    case sccl_buffer_type_device_storage:
    case sccl_buffer_type_shared_storage:
    case sccl_buffer_type_host_uniform:
    case sccl_buffer_type_device_uniform:
    case sccl_buffer_type_shared_uniform: {
        SCCL_TEST_ASSERT(sccl_create_buffer(device, buffer, type, size));
        break;
    }
    case sccl_buffer_type_external_host_pointer_storage:
    case sccl_buffer_type_external_host_pointer_uniform: {
        /* query import alignment requirement */
        sccl_device_properties_t device_properties = {};
        sccl_get_device_properties(device, &device_properties);
        const size_t aligned_size =
            size +
            (size %
             device_properties.min_external_buffer_host_pointer_alignment);
        /* set external_ptr so wwe maintain a reference to underlying memory in
         * case of external memory */
        *external_ptr = aligned_alloc(
            device_properties.min_external_buffer_host_pointer_alignment,
            aligned_size);
        ASSERT_NE(*external_ptr, nullptr);
        error = sccl_create_external_host_pointer_buffer(
            device, buffer, type, *external_ptr, aligned_size);
        if (error == sccl_unsupported_error) {
            free(*external_ptr);
            *supported = false;
        }
        break;
    }
    case sccl_buffer_type_host_dmabuf_storage:
    case sccl_buffer_type_device_dmabuf_storage:
    case sccl_buffer_type_shared_dmabuf_storage:
    case sccl_buffer_type_host_dmabuf_uniform:
    case sccl_buffer_type_device_dmabuf_uniform:
    case sccl_buffer_type_shared_dmabuf_uniform:
    case sccl_buffer_type_external_dmabuf_storage:
    case sccl_buffer_type_external_dmabuf_uniform: {
        error = sccl_create_dmabuf_buffer(device, buffer, type, size);
        if (error == sccl_unsupported_error) {
            *supported = false;
        }
        break;
    }
    default:
        assert(false);
    }
}