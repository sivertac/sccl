
#include "examples_common.hpp"

#include <getopt.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define OPT_GPU_0 1000
#define OPT_GPU_1 1001
#define OPT_BUFFER_SIZE 's'

int main(int argc, char **argv)
{
    /* cmd input */
    int gpu_index_0 = 0;
    int gpu_index_1 = 0;
    size_t buffer_size = 0x1000;
    bool verify = true;
    while (true) {
        static struct option long_options[] = {
            {"help", no_argument, 0, 'h'},
            {"gpu0", required_argument, 0, OPT_GPU_0},
            {"gpu1", required_argument, 0, OPT_GPU_1},
            {"buffersize", required_argument, 0, OPT_BUFFER_SIZE},
            {"verify", required_argument, 0, 'v'},
            {0, 0, 0, 0}};
        /* getopt_long stores the option index here */
        int option_index = 0;
        int c = getopt_long(argc, argv, "hs:v:", long_options, &option_index);
        /* Detect the end of the options */
        if (c == -1) {
            break;
        }
        switch (c) {
        case 'h':
            printf("usage: %s [-h] [--gpu0 <gpu0 index>][--gpu1 <gpu1 "
                   "index>][--verify <0 or 1>]\n",
                   argv[0]);
            return EXIT_SUCCESS;
            break;
        case OPT_GPU_0:
            gpu_index_0 = atoi(optarg);
            break;
        case OPT_GPU_1:
            gpu_index_1 = atoi(optarg);
            break;
        case OPT_BUFFER_SIZE:
            buffer_size = atoi(optarg);
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
    printf("gpu0 = %d\n", gpu_index_0);
    printf("gpu1 = %d\n", gpu_index_1);
    printf("buffer_size = %lu\n", buffer_size);
    printf("verify = %d\n", verify);
    printf("\n");

    /* init gpus */
    sccl_instance_t instance;
    UNWRAP_SCCL_ERROR(sccl_create_instance(&instance));
    sccl_device_t device_0;
    sccl_device_t device_1;
    UNWRAP_SCCL_ERROR(sccl_create_device(instance, &device_0, gpu_index_0));
    UNWRAP_SCCL_ERROR(sccl_create_device(instance, &device_1, gpu_index_1));

    /* create buffers */
    sccl_buffer_t device_0_staging_buffer;
    sccl_buffer_t device_1_staging_buffer;
    sccl_buffer_t device_0_dmabuf_buffer;
    sccl_buffer_t device_1_dmabuf_buffer;
    sccl_buffer_t device_0_external_buffer;

    UNWRAP_SCCL_ERROR(sccl_create_buffer(device_0, &device_0_staging_buffer,
                                         sccl_buffer_type_host_storage,
                                         buffer_size));
    UNWRAP_SCCL_ERROR(sccl_create_buffer(device_1, &device_1_staging_buffer,
                                         sccl_buffer_type_host_storage,
                                         buffer_size));
    UNWRAP_SCCL_ERROR(sccl_create_dmabuf_buffer(
        device_0, &device_0_dmabuf_buffer,
        sccl_buffer_type_device_dmabuf_storage, buffer_size));
    UNWRAP_SCCL_ERROR(sccl_create_dmabuf_buffer(
        device_1, &device_1_dmabuf_buffer,
        sccl_buffer_type_device_dmabuf_storage, buffer_size));

    /* export device_buf 1 dmabuf fd */
    int device_1_fd;
    UNWRAP_SCCL_ERROR(
        sccl_export_dmabuf_buffer(device_1_dmabuf_buffer, &device_1_fd));

    /* import dmabuf on device 0 */
    UNWRAP_SCCL_ERROR(sccl_import_dmabuf_buffer(
        device_0, &device_0_external_buffer, device_1_fd,
        sccl_buffer_type_external_dmabuf_storage, buffer_size));

    /* init staging buffers */
    const size_t element_count = buffer_size / sizeof(uint32_t);
    uint32_t *device_0_staging_buffer_ptr = nullptr;
    uint32_t *device_1_staging_buffer_ptr = nullptr;
    UNWRAP_SCCL_ERROR(sccl_host_map_buffer(
        device_0_staging_buffer, (void **)&device_0_staging_buffer_ptr, 0,
        buffer_size));
    UNWRAP_SCCL_ERROR(sccl_host_map_buffer(
        device_1_staging_buffer, (void **)&device_1_staging_buffer_ptr, 0,
        buffer_size));
    fill_array_random(device_0_staging_buffer_ptr, element_count);
    memset((void *)device_1_staging_buffer_ptr, 0, buffer_size);

    /* create stream */
    sccl_stream_t stream_0;
    sccl_stream_t stream_1;
    UNWRAP_SCCL_ERROR(sccl_create_stream(device_0, &stream_0));
    UNWRAP_SCCL_ERROR(sccl_create_stream(device_1, &stream_1));

    sccl_copy_buffer(stream_0, device_0_staging_buffer, 0,
                     device_0_dmabuf_buffer, 0, buffer_size);
    sccl_copy_buffer(stream_0, device_0_dmabuf_buffer, 0,
                     device_0_external_buffer, 0, buffer_size);
    sccl_dispatch_stream(stream_0);
    sccl_join_stream(stream_0);
    sccl_copy_buffer(stream_1, device_1_dmabuf_buffer, 0,
                     device_1_staging_buffer, 0, buffer_size);
    sccl_dispatch_stream(stream_1);
    sccl_join_stream(stream_1);

    /* verify */
    if (verify) {
        if (memcmp(device_1_staging_buffer_ptr, device_0_staging_buffer_ptr,
                   buffer_size) != 0) {
            printf("Staging buffers not equal\n");
        }
    }

    /* cleanup */
    sccl_destroy_stream(stream_1);
    sccl_destroy_stream(stream_0);
    sccl_host_unmap_buffer(device_1_staging_buffer);
    sccl_host_unmap_buffer(device_0_staging_buffer);
    sccl_destroy_buffer(device_0_external_buffer);
    close(device_1_fd);
    sccl_destroy_buffer(device_1_dmabuf_buffer);
    sccl_destroy_buffer(device_0_dmabuf_buffer);
    sccl_destroy_buffer(device_1_staging_buffer);
    sccl_destroy_buffer(device_0_staging_buffer);
    sccl_destroy_device(device_1);
    sccl_destroy_device(device_0);
    sccl_destroy_instance(instance);
    return EXIT_SUCCESS;
}