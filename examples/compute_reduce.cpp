
#include "examples_common.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <getopt.h>
#include <inttypes.h>
#include <iostream>
#include <numeric>
#include <sccl.h>
#include <string>
#include <vector>

const char *COMPUTE_SHADER_PATH = "shaders/compute_reduce_shader.spv";

struct UniformBufferObject {
    uint numberOfRanks;
    uint rankSize;
};

int main(int argc, char **argv)
{
    /* cmd input */
    int gpu_index = 0;
    int number_of_ranks = 4;
    int input_rank_size = -1;
    bool copy_buffer = false;
    int iterations = 1;
    bool verify = true;
    while (true) {
        static struct option long_options[] = {
            {"help", no_argument, 0, 'h'},
            {"gpu", required_argument, 0, 'g'},
            {"ranks", required_argument, 0, 'r'},
            {"ranksize", required_argument, 0, 's'},
            {"copybuffer", required_argument, 0, 'c'},
            {"iterations", required_argument, 0, 'i'},
            {"verify", required_argument, 0, 'v'},
            {0, 0, 0, 0}};
        /* getopt_long stores the option index here */
        int option_index = 0;
        int c = getopt_long(argc, argv, "hg:r:s:c:i:v:", long_options,
                            &option_index);
        /* Detect the end of the options */
        if (c == -1) {
            break;
        }
        switch (c) {
        case 'h':
            printf("usage: %s [-h] [--gpu <gpu index>][--ranks <number of "
                   "ranks>][--ranksize <rank size (in elements)>][--copybuffer "
                   "<0 or 1>][--iterations <iterations>][--verify <0 or 1>]\n",
                   argv[0]);
            return EXIT_SUCCESS;
            break;
        case 'g':
            gpu_index = atoi(optarg);
            break;
        case 'r':
            number_of_ranks = atoi(optarg);
            break;
        case 's':
            input_rank_size = atoi(optarg);
            break;
        case 'c':
            copy_buffer = static_cast<bool>(atoi(optarg));
            break;
        case 'i':
            iterations = atoi(optarg);
            break;
        case 'v':
            verify = atoi(optarg);
            break;
        case '?':
            /* getopt_long already printed an error message */
            break;
        default:
            printf("invalid arg?!");
            abort();
        }
    }

    printf("User args:\n");
    printf("gpu = %d\n", gpu_index);
    printf("ranks = %d\n", number_of_ranks);
    printf("copybuffer = %d\n", copy_buffer);
    printf("iterations = %d\n", iterations);
    printf("verify = %d\n", verify);
    printf("\n");

    /* init gpu */
    sccl_instance_t instance;
    UNWRAP_SCCL_ERROR(sccl_create_instance(&instance));
    sccl_device_t device;
    UNWRAP_SCCL_ERROR(sccl_create_device(instance, &device, gpu_index));

    /* get device properties */
    printf("Device properties:\n");
    sccl_device_properties_t device_properties = {};
    sccl_get_device_properties(device, &device_properties);
    for (size_t i = 0; i < 3; ++i) {
        printf("device_properties.max_work_group_count[%lu] = "
               "%" PRIu32 "\n",
               i, device_properties.max_work_group_count[i]);
    }
    for (size_t i = 0; i < 3; ++i) {
        printf("device_properties.max_work_group_size[%lu] = "
               "%" PRIu32 "\n",
               i, device_properties.max_work_group_size[i]);
    }
    printf("device_properties.max_work_group_invocations = "
           "%" PRIu32 "\n",
           device_properties.max_work_group_invocations);
    printf("device_properties.native_work_group_size = %" PRIu32 "\n",
           device_properties.native_work_group_size);
    printf("device_properties.max_storage_buffer_size "
           "= %" PRIu32 "\n",
           device_properties.max_storage_buffer_size);
    printf("device_properties.max_uniform_buffer_size "
           "= %" PRIu32 "\n",
           device_properties.max_uniform_buffer_size);
    printf("\n");

    /* set rank sizes */
    uint32_t shader_work_group_size[3];
    shader_work_group_size[0] = device_properties.native_work_group_size;
    shader_work_group_size[1] = 1;
    shader_work_group_size[2] = 1;

    /* distribute `device_properties.max_storage_buffer_size` across dimensions
     */
    size_t allocated_size;
    if (input_rank_size != -1) {
        allocated_size = device_properties.max_storage_buffer_size /
                         sizeof(int) / number_of_ranks /
                         shader_work_group_size[0];
    } else {
        allocated_size = input_rank_size * sizeof(int) * number_of_ranks;
    }

    uint32_t shader_work_group_count[3];
    fill_until(allocated_size, shader_work_group_count[0],
               device_properties.max_work_group_count[0]);
    allocated_size /= shader_work_group_count[0];
    fill_until(allocated_size, shader_work_group_count[1],
               device_properties.max_work_group_count[1]);
    allocated_size /= shader_work_group_count[1];
    fill_until(allocated_size, shader_work_group_count[2],
               device_properties.max_work_group_count[2]);

    printf("Compute shape:\n");
    for (size_t i = 0; i < 3; ++i) {
        printf("shader_work_group_count[%lu] = %" PRIu32 "\n", i,
               shader_work_group_count[i]);
    }

    unsigned int rank_size =
        shader_work_group_count[0] * shader_work_group_count[1] *
        shader_work_group_count[2] * shader_work_group_size[0] *
        shader_work_group_size[1] * shader_work_group_size[2];
    size_t rank_size_bytes = rank_size * sizeof(int);
    printf("number_of_ranks = %d\n", number_of_ranks);
    printf("rank_size = %u\n", rank_size);
    printf("rank_size_bytes = %lu\n", rank_size_bytes);
    printf("\n");

    UniformBufferObject ubo = {};
    ubo.numberOfRanks = number_of_ranks;
    ubo.rankSize = rank_size;

    /* read shader */
    auto shader_source = read_file(COMPUTE_SHADER_PATH);
    if (!shader_source.has_value()) {
        fprintf(stderr, "Failed to open shader file: %s\n",
                COMPUTE_SHADER_PATH);
        exit(EXIT_FAILURE);
    }

    /* create input buffer */
    const size_t input_buffer_size_bytes = rank_size_bytes * number_of_ranks;
    sccl_buffer_t input_buffer;
    sccl_shader_buffer_layout_t input_buffer_layout;
    sccl_shader_buffer_binding_t input_buffer_binding;
    sccl_buffer_t input_staging_buffer;
    int *input_data;
    if (!copy_buffer) {
        UNWRAP_SCCL_ERROR(sccl_create_buffer(device, &input_buffer,
                                             sccl_buffer_type_shared_storage,
                                             input_buffer_size_bytes));
        sccl_set_buffer_layout_binding(input_buffer, 0, 0, &input_buffer_layout,
                                       &input_buffer_binding);
        UNWRAP_SCCL_ERROR(sccl_host_map_buffer(
            input_buffer, (void **)&input_data, 0, input_buffer_size_bytes));
    } else {
        UNWRAP_SCCL_ERROR(sccl_create_buffer(device, &input_buffer,
                                             sccl_buffer_type_device_storage,
                                             input_buffer_size_bytes));
        sccl_set_buffer_layout_binding(input_buffer, 0, 0, &input_buffer_layout,
                                       &input_buffer_binding);
        UNWRAP_SCCL_ERROR(sccl_create_buffer(device, &input_staging_buffer,
                                             sccl_buffer_type_host_storage,
                                             input_buffer_size_bytes));
        UNWRAP_SCCL_ERROR(sccl_host_map_buffer(input_staging_buffer,
                                               (void **)&input_data, 0,
                                               input_buffer_size_bytes));
    }

    /* create output buffer */
    const size_t output_buffer_size_bytes = rank_size_bytes;
    sccl_buffer_t output_buffer;
    sccl_shader_buffer_layout_t output_buffer_layout;
    sccl_shader_buffer_binding_t output_buffer_binding;
    sccl_buffer_t output_staging_buffer;
    int *output_data;
    if (!copy_buffer) {
        UNWRAP_SCCL_ERROR(sccl_create_buffer(device, &output_buffer,
                                             sccl_buffer_type_shared_storage,
                                             output_buffer_size_bytes));
        sccl_set_buffer_layout_binding(
            output_buffer, 1, 0, &output_buffer_layout, &output_buffer_binding);
        UNWRAP_SCCL_ERROR(sccl_host_map_buffer(
            output_buffer, (void **)&output_data, 0, output_buffer_size_bytes));
    } else {
        UNWRAP_SCCL_ERROR(sccl_create_buffer(device, &output_buffer,
                                             sccl_buffer_type_device_storage,
                                             output_buffer_size_bytes));
        sccl_set_buffer_layout_binding(
            output_buffer, 1, 0, &output_buffer_layout, &output_buffer_binding);
        UNWRAP_SCCL_ERROR(sccl_create_buffer(device, &output_staging_buffer,
                                             sccl_buffer_type_host_storage,
                                             output_buffer_size_bytes));
        UNWRAP_SCCL_ERROR(sccl_host_map_buffer(output_staging_buffer,
                                               (void **)&output_data, 0,
                                               output_buffer_size_bytes));
    }

    /* create ubo */
    const size_t uniform_buffer_size_bytes = sizeof(UniformBufferObject);
    sccl_buffer_t uniform_buffer;
    sccl_shader_buffer_layout_t uniform_buffer_layout;
    sccl_shader_buffer_binding_t uniform_buffer_binding;
    void *uniform_data;
    UNWRAP_SCCL_ERROR(sccl_create_buffer(device, &uniform_buffer,
                                         sccl_buffer_type_shared_uniform,
                                         uniform_buffer_size_bytes));
    sccl_set_buffer_layout_binding(uniform_buffer, 2, 0, &uniform_buffer_layout,
                                   &uniform_buffer_binding);
    UNWRAP_SCCL_ERROR(sccl_host_map_buffer(uniform_buffer, &uniform_data, 0,
                                           uniform_buffer_size_bytes));

    /* fill uniform */
    std::memcpy(uniform_data, &ubo, sizeof(UniformBufferObject));

    /* prepare specialization constants */
    const size_t specialization_constants_count = 3;
    sccl_shader_specialization_constant_t
        specialization_constants[specialization_constants_count];
    for (size_t i = 0; i < specialization_constants_count; ++i) {
        specialization_constants[i].constant_id = static_cast<uint32_t>(i);
        specialization_constants[i].size = sizeof(uint32_t);
        specialization_constants[i].data = &shader_work_group_size[i];
    }

    /* create shader */
    sccl_shader_buffer_layout_t buffer_layouts[] = {
        input_buffer_layout, output_buffer_layout, uniform_buffer_layout};
    sccl_shader_config_t shader_config = {};
    shader_config.shader_source_code = shader_source.value().data();
    shader_config.shader_source_code_length = shader_source.value().size();
    shader_config.buffer_layouts = buffer_layouts;
    shader_config.buffer_layouts_count = 3;
    shader_config.specialization_constants = specialization_constants;
    shader_config.specialization_constants_count =
        specialization_constants_count;
    sccl_shader_t shader;
    UNWRAP_SCCL_ERROR(sccl_create_shader(device, &shader, &shader_config));

    /* create stream */
    sccl_stream_t stream;
    UNWRAP_SCCL_ERROR(sccl_create_stream(device, &stream));

    std::chrono::high_resolution_clock::time_point total_start_time =
        std::chrono::high_resolution_clock::now();
    std::vector<std::chrono::high_resolution_clock::duration> iteration_times;
    iteration_times.reserve(iterations);
    std::vector<std::chrono::high_resolution_clock::duration> shader_times;
    shader_times.reserve(iterations);
    for (int iter = 0; iter < iterations; ++iter) {
        std::chrono::high_resolution_clock::time_point iteration_start_time =
            std::chrono::high_resolution_clock::now();
        /* run stream */
        /* fill input buffer */
        for (size_t i = 0; i < rank_size * number_of_ranks; ++i) {
            input_data[i] = static_cast<int>(i);
        }

        /* clear output buffer */
        std::memset(output_data, 0, output_buffer_size_bytes);

        std::chrono::high_resolution_clock::time_point time_point;
        std::chrono::high_resolution_clock::duration duration;
        time_point = std::chrono::high_resolution_clock::now();

        sccl_shader_buffer_binding_t buffer_bindings[] = {
            input_buffer_binding, output_buffer_binding,
            uniform_buffer_binding};
        sccl_shader_run_params_t params = {};
        params.group_count_x = shader_work_group_count[0];
        params.group_count_y = shader_work_group_count[1];
        params.group_count_z = shader_work_group_count[2];
        params.buffer_bindings = buffer_bindings;
        params.buffer_bindings_count = 3;

        if (copy_buffer) {
            UNWRAP_SCCL_ERROR(sccl_copy_buffer(stream, input_staging_buffer, 0,
                                               input_buffer, 0,
                                               input_buffer_size_bytes));
        }

        UNWRAP_SCCL_ERROR(sccl_run_shader(stream, shader, &params));

        if (copy_buffer) {
            UNWRAP_SCCL_ERROR(sccl_copy_buffer(stream, output_buffer, 0,
                                               output_staging_buffer, 0,
                                               output_buffer_size_bytes));
        }

        UNWRAP_SCCL_ERROR(sccl_dispatch_stream(stream));

        /* wait for stream to complete */
        UNWRAP_SCCL_ERROR(sccl_join_stream(stream));

        duration = std::chrono::high_resolution_clock::now() - time_point;
        shader_times.push_back(duration);

        // verify output data
        if (verify) {
            for (size_t i = 0; i < rank_size; ++i) {
                // compute expected data
                int expected_value = 0;
                for (size_t rank = 0;
                     rank < static_cast<size_t>(number_of_ranks); ++rank) {
                    expected_value += input_data[rank * rank_size + i];
                }
                if (output_data[i] != expected_value) {
                    printf("Unexpected value at i = %lu: output_data[i] = %d, "
                           "expected_value = %d\n",
                           i, output_data[i], expected_value);
                    return EXIT_FAILURE;
                }
            }
        }

        duration =
            std::chrono::high_resolution_clock::now() - iteration_start_time;
        iteration_times.push_back(duration);
    }

    std::chrono::high_resolution_clock::time_point total_end_time =
        std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::duration total_duration =
        total_end_time - total_start_time;
    double total_run_time = static_cast<double>(total_duration.count());
    double mean_iteration_time =
        std::accumulate(std::begin(iteration_times), std::end(iteration_times),
                        double{0.0},
                        [](double sum, auto e) {
                            return static_cast<double>(sum + e.count());
                        }) /
        static_cast<double>(iterations);
    double mean_shader_time =
        std::accumulate(std::begin(shader_times), std::end(shader_times),
                        double{0.0},
                        [](double sum, auto e) {
                            return static_cast<double>(sum + e.count());
                        }) /
        static_cast<double>(iterations);
    printf("Total run time (ns), Mean iteration time (ns), Mean shader time "
           "(ns)\n");
    printf("%f, %f, %f\n", total_run_time, mean_iteration_time,
           mean_shader_time);

    /* cleanup */
    sccl_destroy_stream(stream);
    sccl_destroy_shader(shader);
    sccl_host_unmap_buffer(uniform_buffer);
    if (!copy_buffer) {
        sccl_host_unmap_buffer(output_buffer);
        sccl_host_unmap_buffer(input_buffer);
    } else {
        sccl_host_unmap_buffer(output_staging_buffer);
        sccl_host_unmap_buffer(input_staging_buffer);
        sccl_destroy_buffer(output_staging_buffer);
        sccl_destroy_buffer(input_staging_buffer);
    }
    sccl_destroy_buffer(uniform_buffer);
    sccl_destroy_buffer(output_buffer);
    sccl_destroy_buffer(input_buffer);
    sccl_destroy_device(device);
    sccl_destroy_instance(instance);

    return EXIT_SUCCESS;
}
