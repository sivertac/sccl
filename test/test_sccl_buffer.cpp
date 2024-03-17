

#include <sccl.h>

#include "common.hpp"
#include <gtest/gtest.h>

class buffer_test : public testing::Test
{
protected:
    void SetUp() override
    {
        EXPECT_EQ(sccl_create_instance(&instance), sccl_success);
        EXPECT_EQ(
            sccl_create_device(instance, &device, get_environment_gpu_index()),
            sccl_success);
    }

    void TearDown() override
    {
        sccl_destroy_device(device);
        sccl_destroy_instance(instance);
    }

    sccl_instance_t instance;
    sccl_device_t device;
};

TEST_F(buffer_test, create_host_buffer)
{
    size_t size = 0x1000;
    sccl_buffer_t buffer;
    EXPECT_EQ(sccl_create_buffer(device, &buffer, sccl_buffer_type_host, size),
              sccl_success);
    sccl_destroy_buffer(buffer);
}

TEST_F(buffer_test, create_device_buffer)
{
    size_t size = 0x1000;
    sccl_buffer_t buffer;
    EXPECT_EQ(
        sccl_create_buffer(device, &buffer, sccl_buffer_type_device, size),
        sccl_success);
    sccl_destroy_buffer(buffer);
}

TEST_F(buffer_test, create_shared_buffer)
{
    size_t size = 0x1000;
    sccl_buffer_t buffer;
    EXPECT_EQ(
        sccl_create_buffer(device, &buffer, sccl_buffer_type_shared, size),
        sccl_success);
    sccl_destroy_buffer(buffer);
}

TEST_F(buffer_test, host_map_buffer)
{
    size_t size = 0x1000;
    const std::vector<uint32_t> test_data = {0, 1, 2, 3, 4, 5, 6, 7};
    void *data_ptr = nullptr;

    for (sccl_buffer_type_t type :
         {sccl_buffer_type_host, sccl_buffer_type_shared}) {
        sccl_buffer_t buffer;
        EXPECT_EQ(sccl_create_buffer(device, &buffer, type, size),
                  sccl_success);

        EXPECT_EQ(sccl_host_map_buffer(buffer, &data_ptr, 0, size),
                  sccl_success);

        /* write to buffer */
        memcpy(data_ptr, test_data.data(), test_data.size() * sizeof(uint32_t));

        /* read from buffer */
        EXPECT_EQ(memcmp(data_ptr, test_data.data(),
                         test_data.size() * sizeof(uint32_t)),
                  0);

        sccl_host_unmap_buffer(buffer);

        sccl_destroy_buffer(buffer);
    }
}
