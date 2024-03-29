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

TEST_F(shader_test, shader_noop)
{
    std::string shader_source = read_test_shader("noop_shader.spv").value();

    sccl_shader_config_t shader_config = {};
    shader_config.shader_source_code = shader_source.data();
    shader_config.shader_source_code_length = shader_source.size();

    sccl_shader_t shader;
    EXPECT_EQ(sccl_create_shader(device, &shader, &shader_config),
              sccl_success);

    /* run */
    sccl_shader_run_params_t params = {};
    params.group_count_x = 1;

    EXPECT_EQ(sccl_run_shader(stream, shader, &params), sccl_success);

    EXPECT_EQ(sccl_dispatch_stream(stream), sccl_success);

    EXPECT_EQ(sccl_join_stream(stream), sccl_success);

    sccl_destroy_shader(shader);
}

TEST_F(shader_test, shader_buffer_layout)
{
    const size_t binding_count = 4;
    std::string shader_source =
        read_test_shader("buffer_layout_shader.spv").value();

    sccl_shader_buffer_layout_t buffer_layouts[binding_count];
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
    shader_config.buffer_layouts_count = binding_count;

    sccl_shader_t shader;
    EXPECT_EQ(sccl_create_shader(device, &shader, &shader_config),
              sccl_success);

    /* create buffers */
    const size_t buffer_size = 0x1000;
    sccl_buffer_t buffers[binding_count];
    sccl_shader_buffer_binding_t bindings[binding_count];
    for (size_t i = 0; i < binding_count; ++i) {
        EXPECT_EQ(sccl_create_buffer(device, &buffers[i],
                                     buffer_layouts[i].type, buffer_size),
                  sccl_success);
        bindings[i].position = buffer_layouts[i].position;
        bindings[i].buffer = buffers[i];
    }

    /* run */
    sccl_shader_run_params_t params = {};
    params.group_count_x = 1;
    params.buffer_bindings = bindings;
    params.buffer_bindings_count = binding_count;

    EXPECT_EQ(sccl_run_shader(stream, shader, &params), sccl_success);

    EXPECT_EQ(sccl_dispatch_stream(stream), sccl_success);

    EXPECT_EQ(sccl_join_stream(stream), sccl_success);

    /* cleanup */
    for (size_t i = 0; i < binding_count; ++i) {
        sccl_destroy_buffer(buffers[i]);
    }

    sccl_destroy_shader(shader);
}

TEST_F(shader_test, shader_specialization_constants)
{
    std::string shader_source =
        read_test_shader("specialization_constants_shader.spv").value();

    uint32_t c_0 = 0;
    uint32_t c_1 = 1;
    uint32_t c_2 = 2;
    uint32_t c_3 = 3;

    sccl_shader_specialization_constant_t specialization_constants[4];
    specialization_constants[0].constant_id = 0;
    specialization_constants[0].size = sizeof(uint32_t);
    specialization_constants[0].data = &c_0;
    specialization_constants[1].constant_id = 1;
    specialization_constants[1].size = sizeof(uint32_t);
    specialization_constants[1].data = &c_1;
    specialization_constants[2].constant_id = 2;
    specialization_constants[2].size = sizeof(uint32_t);
    specialization_constants[2].data = &c_2;
    specialization_constants[3].constant_id = 3;
    specialization_constants[3].size = sizeof(uint32_t);
    specialization_constants[3].data = &c_3;

    sccl_shader_config_t shader_config = {};
    shader_config.shader_source_code = shader_source.data();
    shader_config.shader_source_code_length = shader_source.size();
    shader_config.specialization_constants = specialization_constants;
    shader_config.specialization_constants_count = 4;

    sccl_shader_t shader;
    EXPECT_EQ(sccl_create_shader(device, &shader, &shader_config),
              sccl_success);

    sccl_destroy_shader(shader);
}

TEST_F(shader_test, shader_push_constants)
{
    std::string shader_source =
        read_test_shader("push_constants_shader.spv").value();

    struct PushConstant {
        uint32_t c_0;
        uint32_t c_1;
        uint32_t c_2;
        uint32_t c_3;
    } push_constant;

    sccl_shader_push_constant_layout_t push_constants[1];
    push_constants[0].size = sizeof(push_constant);

    sccl_shader_config_t shader_config = {};
    shader_config.shader_source_code = shader_source.data();
    shader_config.shader_source_code_length = shader_source.size();
    shader_config.push_constant_layouts = push_constants;
    shader_config.push_constant_layouts_count = 1;

    sccl_shader_t shader;
    EXPECT_EQ(sccl_create_shader(device, &shader, &shader_config),
              sccl_success);

    sccl_destroy_shader(shader);
}
