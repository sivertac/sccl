

#include <sccl.h>

#include "common.hpp"
#include <gtest/gtest.h>

class copy_buffer_test : public testing::Test
{
protected:
    void SetUp() override
    {
        EXPECT_EQ(sccl_create_instance(&instance), sccl_success);
        EXPECT_EQ(
            sccl_create_device(instance, &device, get_environment_gpu_index()),
            sccl_success);
        EXPECT_EQ(sccl_create_stream(device, &stream), sccl_success);

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
    sccl_buffer_t host_buffer;
    sccl_buffer_t device_buffer;
    /* host buffer is twice as big so we can fit test data in first half, and
     * data received from gpu in second half */
    size_t host_buffer_size = test_data_byte_size * 2;
    size_t device_buffer_size = test_data_byte_size;

    /* create buffers */
    EXPECT_EQ(
        sccl_create_buffer(device, &host_buffer, source_type, host_buffer_size),
        sccl_success);
    EXPECT_EQ(sccl_create_buffer(device, &device_buffer, target_type,
                                 device_buffer_size),
              sccl_success);

    /* zero init all of host buffer */
    void *host_data_ptr;
    EXPECT_EQ(
        sccl_host_map_buffer(host_buffer, &host_data_ptr, 0, host_buffer_size),
        sccl_success);
    memset(host_data_ptr, 0, host_buffer_size);
    sccl_host_unmap_buffer(host_buffer);

    /* copy test data to first half of host buffer */
    EXPECT_EQ(sccl_host_map_buffer(host_buffer, &host_data_ptr, 0,
                                   test_data_byte_size),
              sccl_success);
    memcpy(host_data_ptr, test_data.data(), host_buffer_size / 2);

    /* check host buffer */
    EXPECT_EQ(memcmp(static_cast<uint32_t *>(host_data_ptr) + 0,
                     test_data.data(), test_data_byte_size),
              0);
    sccl_host_unmap_buffer(host_buffer);

    /* copy to device */
    EXPECT_EQ(sccl_copy_buffer(stream, host_buffer, 0, device_buffer, 0,
                               test_data_byte_size),
              sccl_success);
    /* copy back to host, but at offset */
    EXPECT_EQ(sccl_copy_buffer(stream, device_buffer, 0, host_buffer,
                               test_data_byte_size, test_data_byte_size),
              sccl_success);

    EXPECT_EQ(sccl_dispatch_stream(stream), sccl_success);
    EXPECT_EQ(sccl_join_stream(stream), sccl_success);

    /* map host buffer at offset and check data */
    EXPECT_EQ(sccl_host_map_buffer(host_buffer, &host_data_ptr,
                                   test_data_byte_size, test_data_byte_size),
              sccl_success);
    EXPECT_EQ(memcmp(static_cast<uint32_t *>(host_data_ptr), test_data.data(),
                     test_data_byte_size),
              0);
    sccl_host_unmap_buffer(host_buffer);

    /* cleanup */
    sccl_destroy_buffer(device_buffer);
    sccl_destroy_buffer(host_buffer);
}

TEST_F(copy_buffer_test, all_valid_permutations)
{
    for (sccl_buffer_type_t src_type :
         {sccl_buffer_type_host_storage, sccl_buffer_type_shared_storage,
          sccl_buffer_type_host_uniform, sccl_buffer_type_shared_uniform}) {
        for (sccl_buffer_type_t dst_type :
             {sccl_buffer_type_host_storage, sccl_buffer_type_shared_storage,
              sccl_buffer_type_host_uniform, sccl_buffer_type_shared_uniform,
              sccl_buffer_type_device_storage,
              sccl_buffer_type_device_uniform}) {
            buffer_write_read_test(src_type, dst_type);
        }
    }
}
