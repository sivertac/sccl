
#include "examples_common.hpp"

#include <iostream>
#include <string>

#include <sccl.h>

const char *COMPUTE_SHADER_PATH = "shaders/compute_basic_shader.spv";

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    /* init gpu */
    sccl_instance_t instance;
    UNWRAP_SCCL_ERROR(sccl_create_instance(&instance));
    sccl_device_t device;
    UNWRAP_SCCL_ERROR(
        sccl_create_device(instance, &device, 0)); /* select gpu at index 0 */

    /* read shader */
    auto shader_source = read_file(COMPUTE_SHADER_PATH);
    if (!shader_source.has_value()) {
        fprintf(stderr, "Failed to open shader file: %s\n",
                COMPUTE_SHADER_PATH);
        exit(EXIT_FAILURE);
    }

    /* create buffers */
    const size_t buffer_size = 1000;
    sccl_buffer_t input_buffer;
    UNWRAP_SCCL_ERROR(sccl_create_buffer(device, &input_buffer,
                                         sccl_buffer_type_shared, buffer_size));
    sccl_buffer_t output_buffer;
    UNWRAP_SCCL_ERROR(sccl_create_buffer(device, &output_buffer,
                                         sccl_buffer_type_shared, buffer_size));
    sccl_shader_buffer_layout_t input_buffer_layout;
    sccl_shader_buffer_binding_t input_buffer_binding;
    sccl_set_buffer_layout_binding(input_buffer, 0, 0, &input_buffer_layout,
                                   &input_buffer_binding);
    sccl_shader_buffer_layout_t output_buffer_layout;
    sccl_shader_buffer_binding_t output_buffer_binding;
    sccl_set_buffer_layout_binding(output_buffer, 1, 0, &output_buffer_layout,
                                   &output_buffer_binding);
    sccl_shader_buffer_layout_t buffer_layouts[] = {input_buffer_layout,
                                                    output_buffer_layout};
    sccl_shader_buffer_binding_t buffer_bindings[] = {input_buffer_binding,
                                                      output_buffer_binding};

    /* prepare specialization constants */
    uint32_t work_group_sizes[] = {
        1, /* x */
        1, /* y */
        1  /* z */
    };
    const size_t specialization_constants_count = 3;
    sccl_shader_specialization_constant_t
        specialization_constants[specialization_constants_count];
    for (size_t i = 0; i < specialization_constants_count; ++i) {
        specialization_constants[i].constant_id = static_cast<uint32_t>(i);
        specialization_constants[i].size = sizeof(uint32_t);
        specialization_constants[i].data = &work_group_sizes[i];
    }

    /* create shader */
    sccl_shader_config_t shader_config = {};
    shader_config.shader_source_code = shader_source.value().data();
    shader_config.shader_source_code_length = shader_source.value().size();
    shader_config.buffer_layouts = buffer_layouts;
    shader_config.buffer_layouts_count = 2;
    shader_config.specialization_constants = specialization_constants;
    shader_config.specialization_constants_count =
        specialization_constants_count;
    sccl_shader_t shader;
    UNWRAP_SCCL_ERROR(sccl_create_shader(device, &shader, &shader_config));

    /* create stream */
    sccl_stream_t stream;
    UNWRAP_SCCL_ERROR(sccl_create_stream(device, &stream));

    /* run shader */
    const size_t size = 10;
    sccl_shader_run_params_t params = {};
    params.group_count_x = size;
    params.group_count_y = 1;
    params.group_count_z = 1;
    params.buffer_bindings = buffer_bindings;
    params.buffer_bindings_count = 2;
    UNWRAP_SCCL_ERROR(sccl_run_shader(stream, shader, &params));
    UNWRAP_SCCL_ERROR(sccl_dispatch_stream(stream));

    /* wait for stream to complete */
    UNWRAP_SCCL_ERROR(sccl_join_stream(stream));

    /* print */
    printf("Input buffer:\n");
    print_data_buffer(input_buffer, size * sizeof(int32_t));
    printf("Output buffer:\n");
    print_data_buffer(output_buffer, size * sizeof(int32_t));

    /* cleanup */
    sccl_destroy_stream(stream);
    sccl_destroy_shader(shader);
    sccl_destroy_buffer(output_buffer);
    sccl_destroy_buffer(input_buffer);
    sccl_destroy_device(device);
    sccl_destroy_instance(instance);

    return EXIT_SUCCESS;
}
