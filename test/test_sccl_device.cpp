

#include <sccl.h>

#include "common.hpp"
#include <gtest/gtest.h>

class device_test : public testing::Test
{
protected:
    void SetUp() override { SCCL_TEST_ASSERT(sccl_create_instance(&instance)); }

    void TearDown() override { sccl_destroy_instance(instance); }

    sccl_instance_t instance;
};

TEST_F(device_test, get_device_count)
{
    uint32_t device_count;
    SCCL_TEST_ASSERT(sccl_get_device_count(instance, &device_count));
}

TEST_F(device_test, create_device)
{
    uint32_t device_count;
    SCCL_TEST_ASSERT(sccl_get_device_count(instance, &device_count));
    EXPECT_GE(device_count, 1);
    sccl_device_t device;
    SCCL_TEST_ASSERT(
        sccl_create_device(instance, &device, get_environment_gpu_index()));

    sccl_destroy_device(device);
}

TEST_F(device_test, create_device_invalid_index)
{
    uint32_t device_count;
    SCCL_TEST_ASSERT(sccl_get_device_count(instance, &device_count));
    sccl_device_t device;
    ASSERT_NE(sccl_create_device(instance, &device, device_count + 1),
              sccl_success);
}