

#include <sccl.h>

#include "common.hpp"
#include <gtest/gtest.h>

class copy_buffer_test : public testing::Test
{
protected:
    void SetUp() override
    {
        SCCL_TEST_ASSERT(sccl_create_instance(&instance));
        SCCL_TEST_ASSERT(
            sccl_create_device(instance, &device, get_environment_gpu_index()));
        SCCL_TEST_ASSERT(sccl_create_stream(device, &stream));

        /* generate test data */
        test_data.reserve(test_data_size);
        for (size_t i = 0; i < test_data_size; ++i) {
            test_data.push_back(i);
        }
    }

    void TearDown() override
    {
        sccl_destroy_stream(stream);
        sccl_destroy_device(device);
        sccl_destroy_instance(instance);
    }

    sccl_instance_t instance;
    sccl_device_t device;
    sccl_stream_t stream;

    const size_t test_data_size = 0x1000;
    const size_t test_data_byte_size =
        test_data_size * sizeof(decltype(test_data)::value_type);
    std::vector<uint32_t> test_data;

    void buffer_write_read_test(sccl_buffer_type_t source_type,
                                sccl_buffer_type_t target_type);
};

void copy_buffer_test::buffer_write_read_test(sccl_buffer_type_t source_type,
                                              sccl_buffer_type_t target_type)
{
    sccl_buffer_t source_buffer;
    sccl_buffer_t target_buffer;
    /* host buffer is twice as big so we can fit test data in first half, and
     * data received from gpu in second half */
    const size_t source_buffer_size = test_data_byte_size * 2;
    const size_t target_buffer_size = test_data_byte_size;

    /* external memory */
    void *source_external_ptr = nullptr;
    void *target_external_ptr = nullptr;

    /* create buffers */
    bool supported = false;
    create_buffer_generic(device, &source_buffer, source_type,
                          source_buffer_size, &source_external_ptr, &supported);
    if (!supported) {
        /* skip, this permutation is not supported on this device */
        return;
    }
    create_buffer_generic(device, &target_buffer, target_type,
                          target_buffer_size, &target_external_ptr, &supported);
    if (!supported) {
        /* skip, this permutation is not supported on this device */
        return;
    }

    /* zero init all of host buffer */
    void *host_data_ptr;
    SCCL_TEST_ASSERT(sccl_host_map_buffer(source_buffer, &host_data_ptr, 0,
                                          source_buffer_size));
    memset(host_data_ptr, 0, source_buffer_size);
    sccl_host_unmap_buffer(source_buffer);

    /* copy test data to first half of host buffer */
    SCCL_TEST_ASSERT(sccl_host_map_buffer(source_buffer, &host_data_ptr, 0,
                                          test_data_byte_size));
    memcpy(host_data_ptr, test_data.data(), source_buffer_size / 2);

    /* check host buffer */
    ASSERT_EQ(memcmp(static_cast<uint32_t *>(host_data_ptr) + 0,
                     test_data.data(), test_data_byte_size),
              0);
    sccl_host_unmap_buffer(source_buffer);

    /* copy to device */
    SCCL_TEST_ASSERT(sccl_copy_buffer(stream, source_buffer, 0, target_buffer,
                                      0, test_data_byte_size));
    /* copy back to host, but at offset */
    SCCL_TEST_ASSERT(sccl_copy_buffer(stream, target_buffer, 0, source_buffer,
                                      test_data_byte_size,
                                      test_data_byte_size));

    SCCL_TEST_ASSERT(sccl_dispatch_stream(stream));
    SCCL_TEST_ASSERT(sccl_join_stream(stream));

    /* map host buffer at offset and check data */
    SCCL_TEST_ASSERT(sccl_host_map_buffer(source_buffer, &host_data_ptr,
                                          test_data_byte_size,
                                          test_data_byte_size));
    ASSERT_EQ(memcmp(static_cast<uint32_t *>(host_data_ptr), test_data.data(),
                     test_data_byte_size),
              0);
    sccl_host_unmap_buffer(source_buffer);

    /* cleanup */
    sccl_destroy_buffer(target_buffer);
    sccl_destroy_buffer(source_buffer);
    /* free external memory */
    if (source_external_ptr != nullptr) {
        free(source_external_ptr);
    }
    if (target_external_ptr != nullptr) {
        free(target_external_ptr);
    }
}

TEST_F(copy_buffer_test, all_valid_permutations)
{
    for (sccl_buffer_type_t src_type :
         {sccl_buffer_type_host_storage, sccl_buffer_type_shared_storage,
          sccl_buffer_type_host_uniform, sccl_buffer_type_shared_uniform,
          sccl_buffer_type_external_host_pointer_storage,
          sccl_buffer_type_external_host_pointer_uniform}) {
        for (sccl_buffer_type_t dst_type :
             {sccl_buffer_type_host_storage, sccl_buffer_type_shared_storage,
              sccl_buffer_type_host_uniform, sccl_buffer_type_shared_uniform,
              sccl_buffer_type_device_storage, sccl_buffer_type_device_uniform,
              sccl_buffer_type_external_host_pointer_storage,
              sccl_buffer_type_external_host_pointer_uniform}) {
            buffer_write_read_test(src_type, dst_type);
        }
    }
}
