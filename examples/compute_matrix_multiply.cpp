
#include "examples_common.hpp"
#include <cstring>
#include <sccl.h>
#include <vector>

const char *COMPUTE_SHADER_PATH = "shaders/compute_matrix_multiply_shader.spv";

/**
 * @brief Perform matrix multiplication on CPU.
 *
 * This function multiplies two matrices `matrix_a` and `matrix_b` and stores
 * the result in `matrix_c`. The matrices are assumed to be stored in row-major
 * order.
 *
 * @param matrix_a Pointer to the first matrix (`height_a` x `width_a`).
 * @param matrix_b Pointer to the second matrix (`width_a` x `width_b`).
 * @param matrix_c Pointer to the resulting matrix (`height_a` x `width_b`).
 * @param height_a Number of rows in matrix `matrix_a`.
 * @param width_a Number of columns in matrix `matrix_a`, which also equals the
 * number of rows in matrix `matrix_b`.
 * @param width_b Number of columns in matrix `matrix_b`.
 */
void matrix_multiply_cpu(const float *matrix_a, const float *matrix_b,
                         float *matrix_c, size_t height_a, size_t width_a,
                         size_t width_b)
{
    for (size_t i = 0; i < height_a; ++i) {
        for (size_t j = 0; j < width_b; ++j) {
            double sum = 0;

            for (size_t k = 0; k < width_a; ++k) {
                double a = matrix_a[i * width_a + k];
                double b = matrix_b[k * width_b + j];
                sum += a * b;
            }

            matrix_c[i * width_b + j] = static_cast<float>(sum);
        }
    }
}

struct PushConstant {
    uint height_a;
    uint width_a;
    uint width_b;
};
void matrix_multiply_gpu(sccl_device_t device, const float *matrix_a,
                         const float *matrix_b, float *matrix_c,
                         size_t height_a, size_t width_a, size_t width_b)
{

    const size_t matrix_a_size_bytes = width_a * height_a * sizeof(float);
    const size_t matrix_b_size_bytes = width_a * width_b * sizeof(float);
    const size_t matrix_c_size_bytes = height_a * width_b * sizeof(float);

    sccl_buffer_t matrix_a_buffer;
    sccl_buffer_t matrix_b_buffer;
    sccl_buffer_t matrix_c_buffer;
    sccl_shader_buffer_layout_t matrix_a_buffer_layout = {};
    sccl_shader_buffer_layout_t matrix_b_buffer_layout = {};
    sccl_shader_buffer_layout_t matrix_c_buffer_layout = {};
    sccl_shader_buffer_binding_t matrix_a_buffer_binding = {};
    sccl_shader_buffer_binding_t matrix_b_buffer_binding = {};
    sccl_shader_buffer_binding_t matrix_c_buffer_binding = {};

    /* create buffers */
    UNWRAP_SCCL_ERROR(sccl_create_buffer(device, &matrix_a_buffer,
                                         sccl_buffer_type_shared,
                                         matrix_a_size_bytes));
    UNWRAP_SCCL_ERROR(sccl_create_buffer(device, &matrix_b_buffer,
                                         sccl_buffer_type_shared,
                                         matrix_b_size_bytes));
    UNWRAP_SCCL_ERROR(sccl_create_buffer(device, &matrix_c_buffer,
                                         sccl_buffer_type_shared,
                                         matrix_c_size_bytes));

    float *matrix_a_data;
    float *matrix_b_data;
    float *matrix_c_data;
    UNWRAP_SCCL_ERROR(sccl_host_map_buffer(
        matrix_a_buffer, (void **)&matrix_a_data, 0, matrix_a_size_bytes));
    UNWRAP_SCCL_ERROR(sccl_host_map_buffer(
        matrix_b_buffer, (void **)&matrix_b_data, 0, matrix_b_size_bytes));
    UNWRAP_SCCL_ERROR(sccl_host_map_buffer(
        matrix_c_buffer, (void **)&matrix_c_data, 0, matrix_c_size_bytes));

    sccl_set_buffer_layout_binding(matrix_a_buffer, 0, 0,
                                   &matrix_a_buffer_layout,
                                   &matrix_a_buffer_binding);
    sccl_set_buffer_layout_binding(matrix_b_buffer, 0, 1,
                                   &matrix_b_buffer_layout,
                                   &matrix_b_buffer_binding);
    sccl_set_buffer_layout_binding(matrix_c_buffer, 0, 2,
                                   &matrix_c_buffer_layout,
                                   &matrix_c_buffer_binding);

    /* create shader */
    sccl_device_properties_t device_properties = {};
    sccl_get_device_properties(device, &device_properties);
    const uint32_t work_group_count =
        height_a * width_b / device_properties.native_work_group_size + 1;

    sccl_shader_specialization_constant_t specialization_constant;
    specialization_constant.constant_id = 0;
    specialization_constant.data = &device_properties.native_work_group_size;
    specialization_constant.size =
        sizeof(device_properties.native_work_group_size);

    auto shader_source = read_file(COMPUTE_SHADER_PATH);
    if (!shader_source.has_value()) {
        fprintf(stderr, "Failed to open shader file: %s\n",
                COMPUTE_SHADER_PATH);
        exit(EXIT_FAILURE);
    }
    sccl_shader_buffer_layout_t buffer_layouts[] = {
        matrix_a_buffer_layout, matrix_b_buffer_layout, matrix_c_buffer_layout};

    sccl_shader_push_constant_layout_t push_constant_layout = {};
    push_constant_layout.size = sizeof(PushConstant);
    sccl_shader_config_t shader_config = {};
    shader_config.shader_source_code = shader_source.value().data();
    shader_config.shader_source_code_length = shader_source.value().size();
    shader_config.buffer_layouts = buffer_layouts;
    shader_config.buffer_layouts_count = 3;
    shader_config.specialization_constants = &specialization_constant;
    shader_config.specialization_constants_count = 1;
    shader_config.push_constant_layouts = &push_constant_layout;
    shader_config.push_constant_layouts_count = 1;
    sccl_shader_t shader;
    UNWRAP_SCCL_ERROR(sccl_create_shader(device, &shader, &shader_config));

    PushConstant push_constant = {};
    push_constant.height_a = height_a;
    push_constant.width_a = width_a;
    push_constant.width_b = width_b;
    sccl_shader_push_constant_binding_t push_constant_binding = {};
    push_constant_binding.data = &push_constant;

    /* copy data to input buffers */
    std::memcpy(matrix_a_data, matrix_a, matrix_a_size_bytes);
    std::memcpy(matrix_b_data, matrix_b, matrix_b_size_bytes);

    /* run shader */
    sccl_stream_t stream;
    UNWRAP_SCCL_ERROR(sccl_create_stream(device, &stream));
    sccl_shader_buffer_binding_t buffer_bindings[] = {matrix_a_buffer_binding,
                                                      matrix_b_buffer_binding,
                                                      matrix_c_buffer_binding};
    sccl_shader_run_params_t params = {};
    params.group_count_x = work_group_count;
    params.group_count_y = 1;
    params.group_count_z = 1;
    params.buffer_bindings = buffer_bindings;
    params.buffer_bindings_count = 3;
    params.push_constant_bindings = &push_constant_binding;
    params.push_constant_bindings_count = 1;
    UNWRAP_SCCL_ERROR(sccl_run_shader(stream, shader, &params));
    UNWRAP_SCCL_ERROR(sccl_dispatch_stream(stream));
    UNWRAP_SCCL_ERROR(sccl_join_stream(stream));

    /* copy output data back */
    std::memcpy(matrix_c, matrix_c_data, matrix_c_size_bytes);

    /* cleanup */
    sccl_destroy_stream(stream);
    sccl_destroy_shader(shader);
    sccl_host_unmap_buffer(matrix_a_buffer);
    sccl_host_unmap_buffer(matrix_b_buffer);
    sccl_host_unmap_buffer(matrix_c_buffer);
    sccl_destroy_buffer(matrix_a_buffer);
    sccl_destroy_buffer(matrix_b_buffer);
    sccl_destroy_buffer(matrix_c_buffer);
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    const size_t height_a = 10;
    size_t width_a = 10;
    size_t width_b = 10;
    std::vector<float> matrix_a(height_a * width_a);
    std::vector<float> matrix_b(width_a * width_b);
    std::vector<float> matrix_c(height_a * width_b);

    fill_array_random(matrix_a.data(), matrix_a.size());
    fill_array_random(matrix_b.data(), matrix_b.size());

    std::printf("matrix_a:\n");
    print_container(matrix_a);

    std::printf("matrix_b:\n");
    print_container(matrix_b);

    /* init gpu */
    sccl_instance_t instance;
    UNWRAP_SCCL_ERROR(sccl_create_instance(&instance));
    sccl_device_t device;
    UNWRAP_SCCL_ERROR(sccl_create_device(instance, &device, 1));
    matrix_multiply_gpu(device, matrix_a.data(), matrix_b.data(),
                        matrix_c.data(), height_a, width_a, width_b);

    std::printf("matrix_c:\n");
    print_container(matrix_c);

    std::printf("verify on cpu\n");
    std::vector<float> matrix_verify(matrix_c.size());
    matrix_multiply_cpu(matrix_a.data(), matrix_b.data(), matrix_verify.data(),
                        height_a, width_a, width_b);
    for (size_t i = 0; i < matrix_c.size(); ++i) {
        if (!float_equal(matrix_c[i], matrix_verify[i])) {
            std::printf("diff on %lu: %f != %f\n", i, matrix_c[i],
                        matrix_verify[i]);
        }
    }

    sccl_destroy_device(device);
    sccl_destroy_instance(instance);

    return EXIT_SUCCESS;
}
