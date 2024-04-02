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

static void
init_output_buffer(const sccl_device_t device, size_t size,
                   sccl_buffer_t *output_buffer,
                   sccl_shader_buffer_layout_t *output_buffer_layout,
                   sccl_shader_buffer_binding_t *output_buffer_binding)
{
    void *data;
    EXPECT_EQ(sccl_create_buffer(device, output_buffer, sccl_buffer_type_shared,
                                 size),
              sccl_success);
    /* init buffer to 0 */
    EXPECT_EQ(sccl_host_map_buffer(*output_buffer, (void **)&data, 0, size),
              sccl_success);
    memset((void *)data, 0, size);
    sccl_host_unmap_buffer(*output_buffer);
    sccl_set_buffer_layout_binding(*output_buffer, 0, 0, output_buffer_layout,
                                   output_buffer_binding);
}

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

    /* get device properties */
    sccl_device_properties_t device_properties;
    sccl_get_device_properties(device, &device_properties);

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
        /* make sure offset is at correct alignment */
        bindings[i].offset =
            i * sccl_get_buffer_min_offset_alignment(buffers[i]);
        /* try weird bind sizes */
        bindings[i].size = i + 1;
    }

    /* run */
    sccl_shader_run_params_t params = {};
    params.group_count_x = 1;
    params.group_count_y = 1;
    params.group_count_z = 1;
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
    const size_t specialization_constant_count = 4;
    std::string shader_source =
        read_test_shader("specialization_constants_shader.spv").value();

    /* setup output buffer to verify results */
    sccl_buffer_t output_buffer;
    uint32_t *output_data;
    const size_t output_buffer_size =
        sizeof(uint32_t) * specialization_constant_count; // in bytes
    sccl_shader_buffer_layout_t output_buffer_layout = {};
    sccl_shader_buffer_binding_t output_buffer_binding = {};
    init_output_buffer(device, output_buffer_size, &output_buffer,
                       &output_buffer_layout, &output_buffer_binding);

    /* setup specialization constants */
    uint32_t c_0 = 0;
    uint32_t c_1 = 1;
    uint32_t c_2 = 2;
    uint32_t c_3 = 3;

    sccl_shader_specialization_constant_t
        specialization_constants[specialization_constant_count];
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
    shader_config.specialization_constants_count =
        specialization_constant_count;
    shader_config.buffer_layouts = &output_buffer_layout;
    shader_config.buffer_layouts_count = 1;

    sccl_shader_t shader;
    EXPECT_EQ(sccl_create_shader(device, &shader, &shader_config),
              sccl_success);

    /* run */
    sccl_shader_run_params_t params = {};
    params.group_count_x = 1;
    params.group_count_y = 1;
    params.group_count_z = 1;
    params.buffer_bindings = &output_buffer_binding;
    params.buffer_bindings_count = 1;

    EXPECT_EQ(sccl_run_shader(stream, shader, &params), sccl_success);

    EXPECT_EQ(sccl_dispatch_stream(stream), sccl_success);

    EXPECT_EQ(sccl_join_stream(stream), sccl_success);

    /* verify output data */
    EXPECT_EQ(sccl_host_map_buffer(output_buffer, (void **)&output_data, 0,
                                   output_buffer_size),
              sccl_success);
    ASSERT_EQ(*(output_data + 0), c_0);
    ASSERT_EQ(*(output_data + 1), c_1);
    ASSERT_EQ(*(output_data + 2), c_2);
    ASSERT_EQ(*(output_data + 3), c_3);
    sccl_host_unmap_buffer(output_buffer);

    /* cleanup */
    sccl_destroy_shader(shader);
    sccl_destroy_buffer(output_buffer);
}

TEST_F(shader_test, shader_push_constants)
{
    const size_t push_constant_count = 1;
    std::string shader_source =
        read_test_shader("push_constants_shader.spv").value();

    struct PushConstant {
        uint32_t c_0;
        uint32_t c_1;
        uint32_t c_2;
        uint32_t c_3;
    };

    /* setup output buffer to verify results */
    sccl_buffer_t output_buffer;
    uint32_t *output_data;
    const size_t output_buffer_size = sizeof(PushConstant); /* in bytes */
    sccl_shader_buffer_layout_t output_buffer_layout = {};
    sccl_shader_buffer_binding_t output_buffer_binding = {};
    init_output_buffer(device, output_buffer_size, &output_buffer,
                       &output_buffer_layout, &output_buffer_binding);

    sccl_shader_push_constant_layout_t
        push_constant_layouts[push_constant_count];
    push_constant_layouts[0].size = sizeof(PushConstant);

    sccl_shader_config_t shader_config = {};
    shader_config.shader_source_code = shader_source.data();
    shader_config.shader_source_code_length = shader_source.size();
    shader_config.push_constant_layouts = push_constant_layouts;
    shader_config.push_constant_layouts_count = push_constant_count;
    shader_config.buffer_layouts = &output_buffer_layout;
    shader_config.buffer_layouts_count = 1;

    sccl_shader_t shader;
    EXPECT_EQ(sccl_create_shader(device, &shader, &shader_config),
              sccl_success);

    /* run */
    PushConstant push_constant = {};
    push_constant.c_0 = 0;
    push_constant.c_1 = 1;
    push_constant.c_2 = 2;
    push_constant.c_3 = 3;
    sccl_shader_push_constant_binding_t
        push_constant_bindings[push_constant_count];
    push_constant_bindings[0].data = &push_constant;

    sccl_shader_run_params_t params = {};
    params.group_count_x = 1;
    params.group_count_y = 1;
    params.group_count_z = 1;
    params.push_constant_bindings = push_constant_bindings;
    params.push_constant_bindings_count = push_constant_count;
    params.buffer_bindings = &output_buffer_binding;
    params.buffer_bindings_count = 1;

    EXPECT_EQ(sccl_run_shader(stream, shader, &params), sccl_success);

    EXPECT_EQ(sccl_dispatch_stream(stream), sccl_success);

    EXPECT_EQ(sccl_join_stream(stream), sccl_success);

    /* verify output data */
    EXPECT_EQ(sccl_host_map_buffer(output_buffer, (void **)&output_data, 0,
                                   output_buffer_size),
              sccl_success);
    EXPECT_EQ(memcmp(output_data, &push_constant, sizeof(PushConstant)), 0);
    sccl_host_unmap_buffer(output_buffer);

    /* cleanup */
    sccl_destroy_shader(shader);
    sccl_destroy_buffer(output_buffer);
}

TEST_F(shader_test, shader_copy_buffer)
{
    /* number of times to rerun pipeline to test if stream can be used again
     * after running a pipeline */
    const size_t run_count = 2;
    const size_t buffer_element_count = 0x1000;
    const size_t buffer_size = buffer_element_count * sizeof(uint32_t);
    std::string shader_source =
        read_test_shader("copy_buffer_shader.spv").value();

    /* init buffers */
    sccl_buffer_t host_input_buffer;
    sccl_buffer_t host_output_buffer;
    sccl_buffer_t device_input_buffer;
    sccl_buffer_t device_output_buffer;
    EXPECT_EQ(sccl_create_buffer(device, &host_input_buffer,
                                 sccl_buffer_type_host, buffer_size),
              sccl_success);
    EXPECT_EQ(sccl_create_buffer(device, &host_output_buffer,
                                 sccl_buffer_type_host, buffer_size),
              sccl_success);
    EXPECT_EQ(sccl_create_buffer(device, &device_input_buffer,
                                 sccl_buffer_type_device, buffer_size),
              sccl_success);
    EXPECT_EQ(sccl_create_buffer(device, &device_output_buffer,
                                 sccl_buffer_type_device, buffer_size),
              sccl_success);
    sccl_shader_buffer_layout_t device_input_buffer_layout = {};
    sccl_shader_buffer_binding_t device_input_buffer_binding = {};
    sccl_set_buffer_layout_binding(device_input_buffer, 0, 0,
                                   &device_input_buffer_layout,
                                   &device_input_buffer_binding);
    sccl_shader_buffer_layout_t device_output_buffer_layout = {};
    sccl_shader_buffer_binding_t device_output_buffer_binding = {};
    sccl_set_buffer_layout_binding(device_output_buffer, 0, 1,
                                   &device_output_buffer_layout,
                                   &device_output_buffer_binding);

    void *input_data;
    void *output_data;
    EXPECT_EQ(
        sccl_host_map_buffer(host_input_buffer, &input_data, 0, buffer_size),
        sccl_success);
    EXPECT_EQ(
        sccl_host_map_buffer(host_output_buffer, &output_data, 0, buffer_size),
        sccl_success);

    /* create shader */
    sccl_shader_buffer_layout_t buffer_layouts[] = {
        device_input_buffer_layout, device_output_buffer_layout};
    sccl_shader_config_t shader_config = {};
    shader_config.shader_source_code = shader_source.data();
    shader_config.shader_source_code_length = shader_source.size();
    shader_config.buffer_layouts = buffer_layouts;
    shader_config.buffer_layouts_count = 2;
    sccl_shader_t shader;
    EXPECT_EQ(sccl_create_shader(device, &shader, &shader_config),
              sccl_success);

    /* run */
    for (size_t run = 0; run < run_count; ++run) {

        /* zero out host output buffer */
        memset(output_data, 0, buffer_size);
        /* set input data */
        for (uint32_t i = 0; i < buffer_element_count; ++i) {
            *(((uint32_t *)input_data) + i) = i;
        }

        EXPECT_EQ(sccl_copy_buffer(stream, host_input_buffer, 0,
                                   device_input_buffer, 0, buffer_size),
                  sccl_success);

        sccl_shader_buffer_binding_t buffer_bindings[] = {
            device_input_buffer_binding, device_output_buffer_binding};
        sccl_shader_run_params_t params = {};
        params.group_count_x = buffer_element_count;
        params.group_count_y = 1;
        params.group_count_z = 1;
        params.buffer_bindings = buffer_bindings;
        params.buffer_bindings_count = 2;
        EXPECT_EQ(sccl_run_shader(stream, shader, &params), sccl_success);

        EXPECT_EQ(sccl_copy_buffer(stream, device_output_buffer, 0,
                                   host_output_buffer, 0, buffer_size),
                  sccl_success);

        EXPECT_EQ(sccl_dispatch_stream(stream), sccl_success);

        EXPECT_EQ(sccl_join_stream(stream), sccl_success);

        /* verify output data */
        for (uint32_t i = 0; i < buffer_element_count; ++i) {
            *(((uint32_t *)input_data) + i) = i;

            EXPECT_EQ(*(((uint32_t *)output_data) + i), i / 2);
        }
    }

    /* cleanup */
    sccl_destroy_shader(shader);
    sccl_host_unmap_buffer(host_input_buffer);
    sccl_host_unmap_buffer(host_output_buffer);
    sccl_destroy_buffer(host_input_buffer);
    sccl_destroy_buffer(host_output_buffer);
    sccl_destroy_buffer(device_input_buffer);
    sccl_destroy_buffer(device_output_buffer);
}

TEST_F(shader_test, shader_copy_buffer_concurrent_streams)
{
    const size_t stream_count = 10; /* number of concurrent streams */
    const size_t buffer_element_count = 0x1000;
    const size_t buffer_size = buffer_element_count * sizeof(uint32_t);
    std::string shader_source =
        read_test_shader("copy_buffer_shader.spv").value();

    /* init buffers */
    std::vector<sccl_buffer_t> host_input_buffers(stream_count);
    std::vector<sccl_buffer_t> host_output_buffers(stream_count);
    std::vector<sccl_buffer_t> device_input_buffers(stream_count);
    std::vector<sccl_buffer_t> device_output_buffers(stream_count);
    std::vector<sccl_shader_buffer_layout_t> device_input_buffer_layouts(
        stream_count);
    std::vector<sccl_shader_buffer_binding_t> device_input_buffer_bindings(
        stream_count);
    std::vector<sccl_shader_buffer_layout_t> device_output_buffer_layouts(
        stream_count);
    std::vector<sccl_shader_buffer_binding_t> device_output_buffer_bindings(
        stream_count);
    std::vector<void *> input_datas(stream_count);
    std::vector<void *> output_datas(stream_count);
    for (size_t i = 0; i < stream_count; ++i) {
        EXPECT_EQ(sccl_create_buffer(device, &host_input_buffers[i],
                                     sccl_buffer_type_host, buffer_size),
                  sccl_success);
        EXPECT_EQ(sccl_create_buffer(device, &host_output_buffers[i],
                                     sccl_buffer_type_host, buffer_size),
                  sccl_success);
        EXPECT_EQ(sccl_create_buffer(device, &device_input_buffers[i],
                                     sccl_buffer_type_device, buffer_size),
                  sccl_success);
        EXPECT_EQ(sccl_create_buffer(device, &device_output_buffers[i],
                                     sccl_buffer_type_device, buffer_size),
                  sccl_success);
        sccl_set_buffer_layout_binding(device_input_buffers[i], 0, 0,
                                       &device_input_buffer_layouts[i],
                                       &device_input_buffer_bindings[i]);
        sccl_set_buffer_layout_binding(device_output_buffers[i], 0, 1,
                                       &device_output_buffer_layouts[i],
                                       &device_output_buffer_bindings[i]);
        EXPECT_EQ(sccl_host_map_buffer(host_input_buffers[i], &input_datas[i],
                                       0, buffer_size),
                  sccl_success);
        EXPECT_EQ(sccl_host_map_buffer(host_output_buffers[i], &output_datas[i],
                                       0, buffer_size),
                  sccl_success);
    }

    /* create shader */
    /* in this case the buffer layouts are the same for each buffer, so we only
     * need to create 1 shader for all stream, just pick first layout */
    sccl_shader_buffer_layout_t buffer_layouts[] = {
        device_input_buffer_layouts.front(),
        device_output_buffer_layouts.front()};
    sccl_shader_config_t shader_config = {};
    shader_config.shader_source_code = shader_source.data();
    shader_config.shader_source_code_length = shader_source.size();
    shader_config.buffer_layouts = buffer_layouts;
    shader_config.buffer_layouts_count = 2;
    /* make sure to set `max_concurrent_buffer_bindings` to make sure we can fit
     * dispatch enought concurrent shaders */
    shader_config.max_concurrent_buffer_bindings = stream_count;
    sccl_shader_t shader;
    EXPECT_EQ(sccl_create_shader(device, &shader, &shader_config),
              sccl_success);

    /* init buffers */
    for (size_t i = 0; i < stream_count; ++i) {
        /* zero out host output buffer */
        memset(output_datas[i], 0, buffer_size);
        /* set input data */
        void *input_data = input_datas[i];
        for (uint32_t j = 0; j < buffer_element_count; ++j) {
            *(((uint32_t *)input_data) + j) = j;
        }
    }

    /* create streams */
    std::vector<sccl_stream_t> streams(stream_count);
    for (size_t i = 0; i < stream_count; ++i) {
        EXPECT_EQ(sccl_create_stream(device, &streams[i]), sccl_success);
    }

    /* run */
    for (size_t i = 0; i < stream_count; ++i) {
        EXPECT_EQ(sccl_copy_buffer(streams[i], host_input_buffers[i], 0,
                                   device_input_buffers[i], 0, buffer_size),
                  sccl_success);

        sccl_shader_buffer_binding_t buffer_bindings[] = {
            device_input_buffer_bindings[i], device_output_buffer_bindings[i]};
        sccl_shader_run_params_t params = {};
        params.group_count_x = buffer_element_count;
        params.group_count_y = 1;
        params.group_count_z = 1;
        params.buffer_bindings = buffer_bindings;
        params.buffer_bindings_count = 2;
        ASSERT_EQ(sccl_run_shader(streams[i], shader, &params), sccl_success);

        EXPECT_EQ(sccl_copy_buffer(streams[i], device_output_buffers[i], 0,
                                   host_output_buffers[i], 0, buffer_size),
                  sccl_success);

        EXPECT_EQ(sccl_dispatch_stream(streams[i]), sccl_success);
    }

    for (size_t i = 0; i < stream_count; ++i) {
        EXPECT_EQ(sccl_join_stream(streams[i]), sccl_success);
    }

    /* verify output data */
    for (size_t i = 0; i < stream_count; ++i) {
        for (uint32_t j = 0; j < buffer_element_count; ++j) {
            EXPECT_EQ(*(((uint32_t *)output_datas[i]) + j), j / 2);
        }
    }

    /* cleanup */
    for (size_t i = 0; i < stream_count; ++i) {
        sccl_destroy_stream(streams[i]);
    }
    sccl_destroy_shader(shader);
    for (size_t i = 0; i < stream_count; ++i) {
        sccl_host_unmap_buffer(host_input_buffers[i]);
        sccl_host_unmap_buffer(host_output_buffers[i]);
        sccl_destroy_buffer(host_input_buffers[i]);
        sccl_destroy_buffer(host_output_buffers[i]);
        sccl_destroy_buffer(device_input_buffers[i]);
        sccl_destroy_buffer(device_output_buffers[i]);
    }
}