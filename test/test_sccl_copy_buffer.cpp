

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
};

TEST_F(copy_buffer_test, copy_buffer_host_to_device)
{
    size_t size = 0x1000;
    const std::vector<uint32_t> test_data = {0, 1, 2, 3, 4, 5, 6, 7};
    sccl_buffer_t host_buffer;
    sccl_buffer_t device_buffer;

    /* create buffers */
    EXPECT_EQ(
        sccl_create_buffer(device, &host_buffer, sccl_buffer_type_host, size),
        sccl_success);
    EXPECT_EQ(sccl_create_buffer(device, &device_buffer,
                                 sccl_buffer_type_device, size),
              sccl_success);

    /* init host buffer */
    void *host_data_ptr;
    EXPECT_EQ(sccl_host_map_buffer(host_buffer, &host_data_ptr, 0, size),
              sccl_success);
    memset(host_data_ptr, 0, size);
    memcpy(host_data_ptr, test_data.data(),
           test_data.size() * sizeof(uint32_t));

    /* check host buffer */
    EXPECT_EQ(memcmp(static_cast<uint32_t *>(host_data_ptr) + 0,
                     test_data.data(), test_data.size() * sizeof(uint32_t)),
              0);
    EXPECT_NE(memcmp(static_cast<uint32_t *>(host_data_ptr) + test_data.size(),
                     test_data.data(), test_data.size() * sizeof(uint32_t)),
              0);

    /* copy to device */
    EXPECT_EQ(sccl_copy_buffer(stream, host_buffer, 0, device_buffer, 0,
                               test_data.size() * sizeof(uint32_t)),
              sccl_success);
    /* copy back to host, but at offset */
    EXPECT_EQ(sccl_copy_buffer(stream, device_buffer, 0, host_buffer,
                               test_data.size() * sizeof(uint32_t),
                               test_data.size() * sizeof(uint32_t)),
              sccl_success);

    EXPECT_EQ(sccl_dispatch_stream(stream), sccl_success);
    EXPECT_EQ(sccl_join_stream(stream), sccl_success);

    /* check host buffer */
    EXPECT_EQ(memcmp(static_cast<uint32_t *>(host_data_ptr) + 0,
                     test_data.data(), test_data.size() * sizeof(uint32_t)),
              0);
    EXPECT_EQ(memcmp(static_cast<uint32_t *>(host_data_ptr) + test_data.size(),
                     test_data.data(), test_data.size() * sizeof(uint32_t)),
              0);

    /* cleanup */
    sccl_host_unmap_buffer(host_buffer);
    sccl_destroy_buffer(device_buffer);
    sccl_destroy_buffer(host_buffer);
}
