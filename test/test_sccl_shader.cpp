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

TEST_F(shader_test, create_shader_noop)
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

TEST_F(shader_test, create_shader_buffer_layout_valid)
{
    std::string shader_source = read_test_shader("noop_shader.spv").value();

    sccl_shader_buffer_layout_t buffer_layouts[4];
    buffer_layouts[0].position.set = 0;
    buffer_layouts[0].position.binding = 0;
    buffer_layouts[0].type = sccl_buffer_type_host_uniform;
    buffer_layouts[1].position.set = 2;
    buffer_layouts[1].position.binding = 1;
    buffer_layouts[1].type = sccl_buffer_type_host_storage;
    buffer_layouts[2].position.set = 1;
    buffer_layouts[2].position.binding = 0;
    buffer_layouts[2].type = sccl_buffer_type_device_uniform;
    buffer_layouts[3].position.set = 1;
    buffer_layouts[3].position.binding = 1;
    buffer_layouts[3].type = sccl_buffer_type_host_storage;

    sccl_shader_config_t shader_config = {};
    shader_config.shader_source_code = shader_source.data();
    shader_config.shader_source_code_length = shader_source.size();
    shader_config.buffer_layouts = buffer_layouts;
    shader_config.buffer_layouts_count = 4;

    sccl_shader_t shader;
    EXPECT_EQ(sccl_create_shader(device, &shader, &shader_config),
              sccl_success);

    sccl_destroy_shader(shader);
}
