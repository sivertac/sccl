

#include <sccl.h>

#include <gtest/gtest.h>

class buffer_test : public testing::Test
{
protected:
    void SetUp() override
    {
        EXPECT_EQ(sccl_create_instance(&instance), sccl_success);
        EXPECT_EQ(sccl_create_device(instance, &device, 0), sccl_success);
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