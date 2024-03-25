#include <sccl.h>

#include "common.hpp"
#include <gtest/gtest.h>

class shader_test : public testing::Test
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

TEST_F(shader_test, create_shader)
{
    std::string shader_source = read_test_shader("noop_shader.spv").value();

    sccl_shader_config_t shader_config = {};
    shader_config.shader_source_code = shader_source.data();
    shader_config.shader_source_code_length = shader_source.size();

    sccl_shader_t shader;
    EXPECT_EQ(sccl_create_shader(device, &shader, &shader_config),
              sccl_success);

    sccl_destroy_shader(shader);
}
