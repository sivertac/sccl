
#include "examples_common.hpp"
#include <cassert>
#include <cstring>
#include <getopt.h>
#include <inttypes.h>
#include <sccl.h>
#include <vector>

const char *COMPUTE_SHADER_PATH = "shaders/compute_reduce_shader_batch.spv";

struct UniformBufferObject {
    uint numberOfRanks;
    uint batchOffset;
    uint batchSize;
};

using ReduceDataType = int;

int main(int argc, char **argv)
{
    /* cmd input */
    int gpu_index = 0;
    int number_of_ranks = 4;
    int input_rank_size = -1;
    bool copy_buffer = false;
    int iterations = 1;
    bool verify = true;
    size_t batch_count = 0;
    size_t batch_size = 0;
    while (true) {
        static struct option long_options[] = {
            {"help", no_argument, 0, 'h'},
            {"gpu", required_argument, 0, 'g'},
            {"ranks", required_argument, 0, 'r'},
            {"ranksize", required_argument, 0, 's'},
            {"copybuffer", required_argument, 0, 'c'},
            {"batchcount", required_argument, 0, 0},
            {"batchsize", required_argument, 0, 0},
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
        case 0:
            /* anon longopts */
            if (std::strcmp(long_options[option_index].name, "batchcount") ==
                0) {
                batch_count = atoi(optarg);
            } else if (std::strcmp(long_options[option_index].name,
                                   "batchsize") == 0) {
                batch_size = atoi(optarg);
            }
            break;
        case 'h':
            printf("usage: %s [-h] [--gpu <gpu index>][--ranks <number of "
                   "ranks>][--ranksize <rank size (in elements)>][--copybuffer "
                   "<0 or 1>][--iterations <iterations>][--verify <0 or "
                   "1>][--batchcount <batch count>][--batchsize <batch size in "
                   "elements>]\n",
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

    if (batch_count == 0) {
        fprintf(stderr, "Must set batchcount\n");
        return EXIT_FAILURE;
    }
    if (batch_size == 0) {
        fprintf(stderr, "Must set batchsize\n");
        return EXIT_FAILURE;
    }

    printf("User args:\n");
    printf("gpu = %d\n", gpu_index);
    printf("ranks = %d\n", number_of_ranks);
    printf("ranksize = %d\n", input_rank_size);
    printf("copybuffer = %d\n", copy_buffer);
    printf("iterations = %d\n", iterations);
    printf("verify = %d\n", verify);
    printf("batchcount = %lu\n", batch_count);
    printf("batchsize = %lu\n", batch_size);
    printf("\n");

    /* init gpu */
    sccl_instance_t instance;
    UNWRAP_SCCL_ERROR(sccl_create_instance(&instance));
    sccl_device_t device;
    UNWRAP_SCCL_ERROR(sccl_create_device(instance, &device, gpu_index));
    sccl_device_properties_t device_properties = {};
    sccl_get_device_properties(device, &device_properties);

    /* check batch size is multiple of warp size */
    if (batch_size % device_properties.native_work_group_size != 0) {
        fprintf(stderr, "Batchsize must be multiple of %" PRIu32 "\n",
                device_properties.native_work_group_size);
        return EXIT_FAILURE;
    }

    /* set rank sizes */
    uint32_t shader_work_group_size[3];
    shader_work_group_size[0] = device_properties.native_work_group_size;
    shader_work_group_size[1] = 1;
    shader_work_group_size[2] = 1;

    size_t allocated_size = batch_size;

    uint32_t shader_work_group_count[3];
    fill_until(allocated_size, shader_work_group_count[0],
               device_properties.max_work_group_count[0]);
    allocated_size /= shader_work_group_count[0];
    fill_until(allocated_size, shader_work_group_count[1],
               device_properties.max_work_group_count[1]);
    allocated_size /= shader_work_group_count[1];
    fill_until(allocated_size, shader_work_group_count[2],
               device_properties.max_work_group_count[2]);

    size_t rank_size = (input_rank_size != -1)
                           ? input_rank_size
                           : device_properties.max_storage_buffer_size /
                                 sizeof(ReduceDataType) / number_of_ranks;

    /* check if rank size is a multiple of batch size */
    if (rank_size % batch_size != 0) {
        fprintf(stderr, "Rank size must be multiple of batch size\n");
        return EXIT_FAILURE;
    }

    const size_t rank_size_bytes = rank_size * sizeof(ReduceDataType);
    const size_t batch_size_bytes = batch_size * sizeof(ReduceDataType);
    printf("Compute shape:\n");
    for (size_t i = 0; i < 3; ++i) {
        printf("shader_work_group_count[%lu] = %" PRIu32 "\n", i,
               shader_work_group_count[i]);
    }
    printf("number_of_ranks = %d\n", number_of_ranks);
    printf("rank_size = %lu\n", rank_size);
    printf("rank_size_bytes = %lu\n", rank_size_bytes);
    printf("\n");

    /* create input buffers */
    const size_t input_staging_buffer_size_bytes =
        rank_size_bytes * number_of_ranks;
    const size_t input_device_buffer_size_bytes =
        batch_size_bytes * number_of_ranks * batch_count;
    sccl_buffer_t input_device_buffer;
    sccl_shader_buffer_layout_t input_buffer_layout;
    sccl_buffer_t input_staging_buffer;
    int *input_data;
    UNWRAP_SCCL_ERROR(sccl_create_buffer(device, &input_device_buffer,
                                         sccl_buffer_type_device_storage,
                                         input_device_buffer_size_bytes));
    sccl_set_buffer_layout_binding(input_device_buffer, 0, 0,
                                   &input_buffer_layout, NULL);
    UNWRAP_SCCL_ERROR(sccl_create_buffer(device, &input_staging_buffer,
                                         sccl_buffer_type_host_storage,
                                         input_staging_buffer_size_bytes));
    UNWRAP_SCCL_ERROR(sccl_host_map_buffer(input_staging_buffer,
                                           (void **)&input_data, 0,
                                           input_staging_buffer_size_bytes));

    /* create output buffers */
    const size_t output_staging_buffer_size_bytes = rank_size_bytes;
    const size_t output_device_buffer_size_bytes =
        batch_size_bytes * batch_count;
    sccl_buffer_t output_device_buffer;
    sccl_shader_buffer_layout_t output_buffer_layout;
    sccl_buffer_t output_staging_buffer;
    int *output_data;
    UNWRAP_SCCL_ERROR(sccl_create_buffer(device, &output_device_buffer,
                                         sccl_buffer_type_device_storage,
                                         output_device_buffer_size_bytes));
    sccl_set_buffer_layout_binding(output_device_buffer, 1, 0,
                                   &output_buffer_layout, NULL);
    UNWRAP_SCCL_ERROR(sccl_create_buffer(device, &output_staging_buffer,
                                         sccl_buffer_type_host_storage,
                                         output_staging_buffer_size_bytes));
    UNWRAP_SCCL_ERROR(sccl_host_map_buffer(output_staging_buffer,
                                           (void **)&output_data, 0,
                                           output_staging_buffer_size_bytes));

    /* fill input buffer */
    for (size_t i = 0; i < rank_size * number_of_ranks; ++i) {
        input_data[i] = static_cast<ReduceDataType>(i);
    }

    /* clear output buffer */
    std::memset(output_data, 0, output_staging_buffer_size_bytes);

    /* create UBOs */
    const size_t uniform_buffer_size_bytes = sizeof(UniformBufferObject);
    std::vector<sccl_buffer_t> uniform_buffers(batch_count);
    for (size_t i = 0; i < uniform_buffers.size(); ++i) {
        /* create */
        UNWRAP_SCCL_ERROR(sccl_create_buffer(device, &uniform_buffers[i],
                                             sccl_buffer_type_shared_uniform,
                                             uniform_buffer_size_bytes));
        sccl_buffer_t ubo = uniform_buffers[i];

        /* assign buffer offset to each ubo */
        UniformBufferObject *ubo_ptr = nullptr;
        UNWRAP_SCCL_ERROR(sccl_host_map_buffer(ubo, (void **)&ubo_ptr, 0,
                                               sizeof(UniformBufferObject)));
        ubo_ptr->numberOfRanks = number_of_ranks;
        ubo_ptr->batchOffset = i * batch_size;
        ubo_ptr->batchSize = batch_size;
        sccl_host_unmap_buffer(ubo);
    }

    /* prepare specialization constants */
    const size_t specialization_constants_count = 3;
    sccl_shader_specialization_constant_t
        specialization_constants[specialization_constants_count];
    for (size_t i = 0; i < specialization_constants_count; ++i) {
        specialization_constants[i].constant_id = static_cast<uint32_t>(i);
        specialization_constants[i].size = sizeof(uint32_t);
        specialization_constants[i].data = &shader_work_group_size[i];
    }
    sccl_shader_buffer_layout_t uniform_buffer_layout = {};
    uniform_buffer_layout.position.set = 2;
    uniform_buffer_layout.position.binding = 0;
    uniform_buffer_layout.type = sccl_buffer_type_shared_uniform;

    /* read shader */
    auto shader_source = read_file(COMPUTE_SHADER_PATH);
    if (!shader_source.has_value()) {
        fprintf(stderr, "Failed to open shader file: %s\n",
                COMPUTE_SHADER_PATH);
        exit(EXIT_FAILURE);
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

    /* create streams */
    std::vector<sccl_stream_t> streams(batch_count);
    for (size_t i = 0; i < streams.size(); ++i) {
        UNWRAP_SCCL_ERROR(sccl_create_stream(device, &streams[i]));
    }

    /* run compute */
    std::vector<sccl_stream_t> pending_streams;
    std::vector<sccl_buffer_t> pending_ubos;
    std::vector<sccl_stream_t> ready_streams;
    std::vector<sccl_buffer_t> ready_ubos;
    std::vector<uint8_t> completed_list;
    pending_streams.reserve(streams.size());
    pending_ubos.reserve(streams.size());
    ready_streams.reserve(streams.size());
    ready_ubos.reserve(streams.size());
    completed_list.reserve(streams.size());

    ready_streams.insert(std::end(ready_streams), std::begin(streams),
                         std::end(streams));
    ready_ubos.insert(std::end(ready_ubos), std::begin(uniform_buffers),
                      std::end(uniform_buffers));

    printf("Run shader\n");
    size_t elements_remaining = rank_size;
    while (elements_remaining > 0) {
        /* if we are out of streams, wait for streams to complete */
        if (ready_streams.empty()) {
            completed_list.resize(pending_streams.size());
            UNWRAP_SCCL_ERROR(sccl_wait_streams(device, pending_streams.data(),
                                                pending_streams.size(),
                                                completed_list.data()));
            for (auto it = std::rbegin(completed_list);
                 it != std::rend(completed_list); std::advance(it, 1)) {
                size_t index =
                    std::distance(std::begin(completed_list), it.base()) - 1;
                if (*it == 1) {
                    /* reset stream so it can be used again */
                    UNWRAP_SCCL_ERROR(
                        sccl_reset_stream(pending_streams[index]));

                    /* move stream and ubo into ready lists */
                    ready_streams.push_back(pending_streams[index]);
                    ready_ubos.push_back(pending_ubos[index]);
                    pending_streams.erase(std::begin(pending_streams) + index);
                    pending_ubos.erase(std::begin(pending_ubos) + index);
                }
            }
        }

        /* aquire stream */
        assert(ready_streams.size() > 0);
        assert(ready_ubos.size() > 0);
        assert(ready_streams.size() == ready_ubos.size());
        sccl_stream_t stream = ready_streams.back();
        sccl_buffer_t ubo = ready_ubos.back();
        ready_streams.pop_back();
        ready_ubos.pop_back();
        pending_streams.push_back(stream);
        pending_ubos.push_back(ubo);

        /* record and dispatch stream */
        /* prepare ubo */
        UniformBufferObject *ubo_ptr = nullptr;
        sccl_shader_buffer_binding_t ubo_binding = {};
        UNWRAP_SCCL_ERROR(sccl_host_map_buffer(ubo, (void **)&ubo_ptr, 0,
                                               sizeof(UniformBufferObject)));
        const size_t batch_offset_bytes =
            ubo_ptr->batchOffset * sizeof(ReduceDataType);
        const size_t batch_index = ubo_ptr->batchOffset / batch_size;
        sccl_host_unmap_buffer(ubo);
        sccl_set_buffer_layout_binding(ubo, 2, 0, nullptr, &ubo_binding);

        sccl_shader_buffer_binding_t input_buffer_binding = {};
        sccl_set_buffer_layout_binding(input_device_buffer, 0, 0, nullptr,
                                       &input_buffer_binding);

        printf("batch_offset_bytes = %lu\n", batch_offset_bytes);
        printf("batch_index = %lu\n", batch_index);
        input_buffer_binding.offset =
            batch_index * batch_size_bytes; /* 1 input batch for each rank */
        input_buffer_binding.size = batch_size_bytes * number_of_ranks;
        printf("input_buffer_binding.offset = %lu\n",
               input_buffer_binding.offset);

        sccl_shader_buffer_binding_t output_buffer_binding = {};
        sccl_set_buffer_layout_binding(output_device_buffer, 1, 0, nullptr,
                                       &output_buffer_binding);
        output_buffer_binding.offset = batch_offset_bytes;
        output_buffer_binding.size = batch_size_bytes;

        sccl_shader_buffer_binding_t buffer_bindings[] = {
            input_buffer_binding, output_buffer_binding, ubo_binding};
        sccl_shader_run_params_t params = {};
        params.group_count_x = shader_work_group_count[0];
        params.group_count_y = shader_work_group_count[1];
        params.group_count_z = shader_work_group_count[2];
        params.buffer_bindings = buffer_bindings;
        params.buffer_bindings_count = 3;

        /* record to stream */
        /* copy each input rank */
        const size_t rank_offset_bytes =
            (rank_size - elements_remaining) * sizeof(ReduceDataType);
        for (size_t i = 0; i < static_cast<size_t>(number_of_ranks); ++i) {
            const size_t src_offset = rank_size_bytes * i + rank_offset_bytes;
            const size_t dst_offset = batch_size_bytes * i + batch_offset_bytes;
            printf("src_offset = %lu\n", src_offset);
            printf("dst_offset = %lu\n", dst_offset);
            UNWRAP_SCCL_ERROR(sccl_copy_buffer(stream, input_staging_buffer,
                                               src_offset, input_device_buffer,
                                               dst_offset, batch_size_bytes));
        }
        UNWRAP_SCCL_ERROR(sccl_run_shader(stream, shader, &params));

        const size_t output_src_offset = batch_offset_bytes;
        const size_t output_dst_offset = rank_offset_bytes;
        printf("output_src_offset = %lu\n", output_src_offset);
        printf("output_dst_offset = %lu\n", output_dst_offset);
        UNWRAP_SCCL_ERROR(sccl_copy_buffer(
            stream, output_device_buffer, output_src_offset,
            output_staging_buffer, output_dst_offset, batch_size_bytes));

        UNWRAP_SCCL_ERROR(sccl_dispatch_stream(stream));

        /* decriment */
        elements_remaining = (elements_remaining > batch_size)
                                 ? elements_remaining - batch_size
                                 : 0;
    }
    /* wait for remaining streams */
    UNWRAP_SCCL_ERROR(sccl_wait_streams_all(device, pending_streams.data(),
                                            pending_streams.size()));

    // verify output data
    if (verify) {
        printf("Verify\n");
        for (size_t i = 0; i < rank_size; ++i) {
            // compute expected data
            int expected_value = 0;
            for (size_t rank = 0; rank < static_cast<size_t>(number_of_ranks);
                 ++rank) {
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

    /* cleanup */
    for (size_t i = 0; i < streams.size(); i++) {
        sccl_destroy_stream(streams[i]);
    }
    sccl_destroy_shader(shader);
    for (size_t i = 0; i < uniform_buffers.size(); ++i) {
        sccl_destroy_buffer(uniform_buffers[i]);
    }
    sccl_host_unmap_buffer(output_staging_buffer);
    sccl_host_unmap_buffer(input_staging_buffer);
    sccl_destroy_buffer(output_staging_buffer);
    sccl_destroy_buffer(input_staging_buffer);
    sccl_destroy_buffer(output_device_buffer);
    sccl_destroy_buffer(input_device_buffer);
    sccl_destroy_device(device);
    sccl_destroy_instance(instance);

    return EXIT_SUCCESS;
}
