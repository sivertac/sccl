#include <sccl.h>

#include "common.hpp"
#include <gtest/gtest.h>

class shader_test : public testing::Test
{
protected:
    void SetUp() override
    {
        SCCL_TEST_ASSERT(sccl_create_instance(&instance));
        SCCL_TEST_ASSERT(
            sccl_create_device(instance, &device, get_environment_gpu_index()));
        SCCL_TEST_ASSERT(sccl_create_stream(device, &stream));
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

    void buffer_passthrough_test(sccl_buffer_type_t source_type,
                                 sccl_buffer_type_t target_type);
};

static void
init_output_buffer(const sccl_device_t device, size_t size,
                   sccl_buffer_t *output_buffer,
                   sccl_shader_buffer_layout_t *output_buffer_layout,
                   sccl_shader_buffer_binding_t *output_buffer_binding)
{
    void *data;
    SCCL_TEST_ASSERT(sccl_create_buffer(device, output_buffer,
                                        sccl_buffer_type_shared, size));
    /* init buffer to 0 */
    SCCL_TEST_ASSERT(
        sccl_host_map_buffer(*output_buffer, (void **)&data, 0, size));
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
    SCCL_TEST_ASSERT(sccl_create_shader(device, &shader, &shader_config));

    /* run */
    sccl_shader_run_params_t params = {};
    params.group_count_x = 1;

    SCCL_TEST_ASSERT(sccl_run_shader(stream, shader, &params));

    SCCL_TEST_ASSERT(sccl_dispatch_stream(stream));

    SCCL_TEST_ASSERT(sccl_join_stream(stream));

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
    SCCL_TEST_ASSERT(sccl_create_shader(device, &shader, &shader_config));

    /* create buffers */
    const size_t buffer_size = 0x1000;
    sccl_buffer_t buffers[binding_count];
    sccl_shader_buffer_binding_t bindings[binding_count];
    for (size_t i = 0; i < binding_count; ++i) {
        SCCL_TEST_ASSERT(sccl_create_buffer(
            device, &buffers[i], buffer_layouts[i].type, buffer_size));
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

    SCCL_TEST_ASSERT(sccl_run_shader(stream, shader, &params));

    SCCL_TEST_ASSERT(sccl_dispatch_stream(stream));

    SCCL_TEST_ASSERT(sccl_join_stream(stream));

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
    SCCL_TEST_ASSERT(sccl_create_shader(device, &shader, &shader_config));

    /* run */
    sccl_shader_run_params_t params = {};
    params.group_count_x = 1;
    params.group_count_y = 1;
    params.group_count_z = 1;
    params.buffer_bindings = &output_buffer_binding;
    params.buffer_bindings_count = 1;

    SCCL_TEST_ASSERT(sccl_run_shader(stream, shader, &params));

    SCCL_TEST_ASSERT(sccl_dispatch_stream(stream));

    SCCL_TEST_ASSERT(sccl_join_stream(stream));

    /* verify output data */
    SCCL_TEST_ASSERT(sccl_host_map_buffer(output_buffer, (void **)&output_data,
                                          0, output_buffer_size));
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
    SCCL_TEST_ASSERT(sccl_create_shader(device, &shader, &shader_config));

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

    SCCL_TEST_ASSERT(sccl_run_shader(stream, shader, &params));

    SCCL_TEST_ASSERT(sccl_dispatch_stream(stream));

    SCCL_TEST_ASSERT(sccl_join_stream(stream));

    /* verify output data */
    SCCL_TEST_ASSERT(sccl_host_map_buffer(output_buffer, (void **)&output_data,
                                          0, output_buffer_size));
    ASSERT_EQ(memcmp(output_data, &push_constant, sizeof(PushConstant)), 0);
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
    SCCL_TEST_ASSERT(sccl_create_buffer(device, &host_input_buffer,
                                        sccl_buffer_type_host, buffer_size));
    SCCL_TEST_ASSERT(sccl_create_buffer(device, &host_output_buffer,
                                        sccl_buffer_type_host, buffer_size));
    SCCL_TEST_ASSERT(sccl_create_buffer(device, &device_input_buffer,
                                        sccl_buffer_type_device, buffer_size));
    SCCL_TEST_ASSERT(sccl_create_buffer(device, &device_output_buffer,
                                        sccl_buffer_type_device, buffer_size));
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
    SCCL_TEST_ASSERT(
        sccl_host_map_buffer(host_input_buffer, &input_data, 0, buffer_size));
    SCCL_TEST_ASSERT(
        sccl_host_map_buffer(host_output_buffer, &output_data, 0, buffer_size));

    /* create shader */
    sccl_shader_buffer_layout_t buffer_layouts[] = {
        device_input_buffer_layout, device_output_buffer_layout};
    sccl_shader_config_t shader_config = {};
    shader_config.shader_source_code = shader_source.data();
    shader_config.shader_source_code_length = shader_source.size();
    shader_config.buffer_layouts = buffer_layouts;
    shader_config.buffer_layouts_count = 2;
    sccl_shader_t shader;
    SCCL_TEST_ASSERT(sccl_create_shader(device, &shader, &shader_config));

    /* run */
    for (size_t run = 0; run < run_count; ++run) {

        /* zero out host output buffer */
        memset(output_data, 0, buffer_size);
        /* set input data */
        for (uint32_t i = 0; i < buffer_element_count; ++i) {
            *(((uint32_t *)input_data) + i) = i;
        }

        SCCL_TEST_ASSERT(sccl_copy_buffer(stream, host_input_buffer, 0,
                                          device_input_buffer, 0, buffer_size));

        sccl_shader_buffer_binding_t buffer_bindings[] = {
            device_input_buffer_binding, device_output_buffer_binding};
        sccl_shader_run_params_t params = {};
        params.group_count_x = buffer_element_count;
        params.group_count_y = 1;
        params.group_count_z = 1;
        params.buffer_bindings = buffer_bindings;
        params.buffer_bindings_count = 2;
        SCCL_TEST_ASSERT(sccl_run_shader(stream, shader, &params));

        SCCL_TEST_ASSERT(sccl_copy_buffer(stream, device_output_buffer, 0,
                                          host_output_buffer, 0, buffer_size));

        SCCL_TEST_ASSERT(sccl_dispatch_stream(stream));

        SCCL_TEST_ASSERT(sccl_join_stream(stream));

        /* verify output data */
        for (uint32_t i = 0; i < buffer_element_count; ++i) {
            *(((uint32_t *)input_data) + i) = i;

            ASSERT_EQ(*(((uint32_t *)output_data) + i), i / 2);
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
        SCCL_TEST_ASSERT(sccl_create_buffer(device, &host_input_buffers[i],
                                            sccl_buffer_type_host,
                                            buffer_size));
        SCCL_TEST_ASSERT(sccl_create_buffer(device, &host_output_buffers[i],
                                            sccl_buffer_type_host,
                                            buffer_size));
        SCCL_TEST_ASSERT(sccl_create_buffer(device, &device_input_buffers[i],
                                            sccl_buffer_type_device,
                                            buffer_size));
        SCCL_TEST_ASSERT(sccl_create_buffer(device, &device_output_buffers[i],
                                            sccl_buffer_type_device,
                                            buffer_size));
        sccl_set_buffer_layout_binding(device_input_buffers[i], 0, 0,
                                       &device_input_buffer_layouts[i],
                                       &device_input_buffer_bindings[i]);
        sccl_set_buffer_layout_binding(device_output_buffers[i], 0, 1,
                                       &device_output_buffer_layouts[i],
                                       &device_output_buffer_bindings[i]);
        SCCL_TEST_ASSERT(sccl_host_map_buffer(host_input_buffers[i],
                                              &input_datas[i], 0, buffer_size));
        SCCL_TEST_ASSERT(sccl_host_map_buffer(
            host_output_buffers[i], &output_datas[i], 0, buffer_size));
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
    SCCL_TEST_ASSERT(sccl_create_shader(device, &shader, &shader_config));

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
        SCCL_TEST_ASSERT(sccl_create_stream(device, &streams[i]));
    }

    /* run */
    for (size_t i = 0; i < stream_count; ++i) {
        SCCL_TEST_ASSERT(sccl_copy_buffer(streams[i], host_input_buffers[i], 0,
                                          device_input_buffers[i], 0,
                                          buffer_size));

        sccl_shader_buffer_binding_t buffer_bindings[] = {
            device_input_buffer_bindings[i], device_output_buffer_bindings[i]};
        sccl_shader_run_params_t params = {};
        params.group_count_x = buffer_element_count;
        params.group_count_y = 1;
        params.group_count_z = 1;
        params.buffer_bindings = buffer_bindings;
        params.buffer_bindings_count = 2;
        SCCL_TEST_ASSERT(sccl_run_shader(streams[i], shader, &params));

        SCCL_TEST_ASSERT(sccl_copy_buffer(streams[i], device_output_buffers[i],
                                          0, host_output_buffers[i], 0,
                                          buffer_size));

        SCCL_TEST_ASSERT(sccl_dispatch_stream(streams[i]));
    }

    for (size_t i = 0; i < stream_count; ++i) {
        SCCL_TEST_ASSERT(sccl_join_stream(streams[i]));
    }

    /* verify output data */
    for (size_t i = 0; i < stream_count; ++i) {
        for (uint32_t j = 0; j < buffer_element_count; ++j) {
            ASSERT_EQ(*(((uint32_t *)output_datas[i]) + j), j / 2);
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

static bool is_buffer_type_dmabuf(sccl_buffer_type_t type)
{
    switch (type) {
    case sccl_buffer_type_host_dmabuf_storage:
    case sccl_buffer_type_device_dmabuf_storage:
    case sccl_buffer_type_shared_dmabuf_storage:
    case sccl_buffer_type_host_dmabuf_uniform:
    case sccl_buffer_type_device_dmabuf_uniform:
    case sccl_buffer_type_shared_dmabuf_uniform:
    case sccl_buffer_type_external_dmabuf_storage:
    case sccl_buffer_type_external_dmabuf_uniform:
        return true;
    default:
        return false;
    }
}

void shader_test::buffer_passthrough_test(sccl_buffer_type_t source_type,
                                          sccl_buffer_type_t target_type)
{

    if (get_environment_platform_docker() &&
        (is_buffer_type_dmabuf(source_type) ||
         is_buffer_type_dmabuf(target_type))) {
        /* skip, dmabuf does not work well inside docker containers */
        return;
    }

    std::string shader_source =
        read_test_shader("copy_buffer_shader.spv").value();

    sccl_buffer_t staging_buffer;
    sccl_buffer_t source_buffer;
    sccl_buffer_t target_buffer;

    const size_t element_count = 0x1000;
    using Type = uint32_t;

    const size_t staging_buffer_size = element_count * sizeof(Type);
    const size_t source_buffer_size = element_count * sizeof(Type);
    const size_t target_buffer_size = element_count * sizeof(Type);

    /* external memory */
    void *source_external_ptr = nullptr;
    void *target_external_ptr = nullptr;

    /* create buffers */
    SCCL_TEST_ASSERT(sccl_create_buffer(
        device, &staging_buffer, sccl_buffer_type_host, staging_buffer_size));

    bool supported = false;
    create_buffer_generic(device, &source_buffer, source_type,
                          source_buffer_size, &source_external_ptr, &supported);
    if (!supported) {
        sccl_destroy_buffer(staging_buffer);
        /* skip, this permutation is not supported on this device */
        return;
    }
    create_buffer_generic(device, &target_buffer, target_type,
                          target_buffer_size, &target_external_ptr, &supported);
    if (!supported) {
        sccl_destroy_buffer(source_buffer);
        sccl_destroy_buffer(staging_buffer);
        /* skip, this permutation is not supported on this device */
        return;
    }

    sccl_shader_buffer_layout_t source_buffer_layout = {};
    sccl_shader_buffer_binding_t source_buffer_binding = {};
    sccl_set_buffer_layout_binding(source_buffer, 0, 0, &source_buffer_layout,
                                   &source_buffer_binding);
    sccl_shader_buffer_layout_t target_buffer_layout = {};
    sccl_shader_buffer_binding_t target_buffer_binding = {};
    sccl_set_buffer_layout_binding(target_buffer, 0, 1, &target_buffer_layout,
                                   &target_buffer_binding);

    /* init stageing buffer */
    uint32_t *staging_data_ptr;
    SCCL_TEST_ASSERT(sccl_host_map_buffer(
        staging_buffer, (void **)&staging_data_ptr, 0, staging_buffer_size));
    for (size_t i = 0; i < element_count; ++i) {
        staging_data_ptr[i] = i;
    }
    sccl_host_unmap_buffer(staging_buffer);

    printf("source_buffer_type = %d\n", source_type);
    printf("target_buffer_type = %d\n", target_type);

    /* create shader */
    sccl_shader_buffer_layout_t buffer_layouts[] = {source_buffer_layout,
                                                    target_buffer_layout};
    sccl_shader_config_t shader_config = {};
    shader_config.shader_source_code = shader_source.data();
    shader_config.shader_source_code_length = shader_source.size();
    shader_config.buffer_layouts = buffer_layouts;
    shader_config.buffer_layouts_count = 2;

    sccl_shader_t shader;
    SCCL_TEST_ASSERT(sccl_create_shader(device, &shader, &shader_config));

    /* run */
    sccl_shader_buffer_binding_t buffer_bindings[] = {source_buffer_binding,
                                                      target_buffer_binding};
    sccl_shader_run_params_t params = {};
    params.group_count_x = element_count;
    params.group_count_y = 1;
    params.group_count_z = 1;
    params.buffer_bindings = buffer_bindings;
    params.buffer_bindings_count = 2;

    SCCL_TEST_ASSERT(sccl_copy_buffer(stream, staging_buffer, 0, source_buffer,
                                      0, staging_buffer_size));

    SCCL_TEST_ASSERT(sccl_run_shader(stream, shader, &params));

    SCCL_TEST_ASSERT(sccl_copy_buffer(stream, target_buffer, 0, staging_buffer,
                                      0, target_buffer_size));

    SCCL_TEST_ASSERT(sccl_dispatch_stream(stream));

    SCCL_TEST_ASSERT(sccl_join_stream(stream));

    /* verify output data in staging buffer */
    SCCL_TEST_ASSERT(sccl_host_map_buffer(
        staging_buffer, (void **)&staging_data_ptr, 0, staging_buffer_size));
    for (size_t i = 0; i < element_count; ++i) {
        EXPECT_EQ(staging_data_ptr[i], i / 2);
    }
    sccl_host_unmap_buffer(staging_buffer);

    /* cleanup */
    sccl_destroy_shader(shader);
    sccl_destroy_buffer(target_buffer);
    sccl_destroy_buffer(source_buffer);
    sccl_destroy_buffer(staging_buffer);
}

TEST_F(shader_test, shader_passthrough_all_valid_buffer_permutations)
{
    for (sccl_buffer_type_t src_type :
         {sccl_buffer_type_host_storage, sccl_buffer_type_device_storage,
          sccl_buffer_type_shared_storage,
          sccl_buffer_type_external_host_pointer_storage,
          sccl_buffer_type_host_dmabuf_storage,
          sccl_buffer_type_device_dmabuf_storage,
          sccl_buffer_type_shared_dmabuf_storage}) {
        for (sccl_buffer_type_t dst_type :
             {sccl_buffer_type_host_storage, sccl_buffer_type_device_storage,
              sccl_buffer_type_shared_storage,
              sccl_buffer_type_external_host_pointer_storage,
              sccl_buffer_type_host_dmabuf_storage,
              sccl_buffer_type_device_dmabuf_storage,
              sccl_buffer_type_shared_dmabuf_storage}) {
            buffer_passthrough_test(src_type, dst_type);
        }
    }
}