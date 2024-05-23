

#include <sccl.h>

#include "common.hpp"
#include <gtest/gtest.h>
#include <stdlib.h>

class buffer_test : public testing::Test
{
protected:
    void SetUp() override
    {
        SCCL_TEST_ASSERT(sccl_create_instance(&instance));
        SCCL_TEST_ASSERT(
            sccl_create_device(instance, &device, get_environment_gpu_index()));
    }

    void TearDown() override
    {
        sccl_destroy_device(device);
        sccl_destroy_instance(instance);
    }

    sccl_instance_t instance;
    sccl_device_t device;
};

TEST_F(buffer_test, create_host_buffer)
{
    size_t size = 0x1000;
    sccl_buffer_t buffer;
    SCCL_TEST_ASSERT(
        sccl_create_buffer(device, &buffer, sccl_buffer_type_host, size));
    sccl_destroy_buffer(buffer);
}

TEST_F(buffer_test, create_device_buffer)
{
    size_t size = 0x1000;
    sccl_buffer_t buffer;
    SCCL_TEST_ASSERT(
        sccl_create_buffer(device, &buffer, sccl_buffer_type_device, size));
    sccl_destroy_buffer(buffer);
}

TEST_F(buffer_test, create_shared_buffer)
{
    size_t size = 0x1000;
    sccl_buffer_t buffer;
    SCCL_TEST_ASSERT(
        sccl_create_buffer(device, &buffer, sccl_buffer_type_shared, size));
    sccl_destroy_buffer(buffer);
}

TEST_F(buffer_test, host_map_buffer)
{
    size_t size = 0x1000;
    const std::vector<uint32_t> test_data = {0, 1, 2, 3, 4, 5, 6, 7};
    void *data_ptr = nullptr;

    for (sccl_buffer_type_t type :
         {sccl_buffer_type_host_storage, sccl_buffer_type_host_uniform,
          sccl_buffer_type_shared_storage, sccl_buffer_type_shared_uniform}) {
        sccl_buffer_t buffer;
        SCCL_TEST_ASSERT(sccl_create_buffer(device, &buffer, type, size));

        SCCL_TEST_ASSERT(sccl_host_map_buffer(buffer, &data_ptr, 0, size));

        /* write to buffer */
        memcpy(data_ptr, test_data.data(), test_data.size() * sizeof(uint32_t));

        /* read from buffer */
        ASSERT_EQ(memcmp(data_ptr, test_data.data(),
                         test_data.size() * sizeof(uint32_t)),
                  0);

        sccl_host_unmap_buffer(buffer);

        sccl_destroy_buffer(buffer);
    }
}

TEST_F(buffer_test, create_host_pointer_buffer)
{
    /* query import alignment requirement */
    sccl_device_properties_t device_properties = {};
    sccl_get_device_properties(device, &device_properties);

    const std::vector<uint32_t> test_data = {0, 1, 2, 3, 4, 5, 6, 7};
    const size_t size =
        device_properties.min_external_buffer_host_pointer_alignment;
    void *data_ptr = aligned_alloc(
        device_properties.min_external_buffer_host_pointer_alignment, size);
    ASSERT_NE(data_ptr, nullptr);

    sccl_buffer_t buffer;

    sccl_error_t error =
        sccl_create_host_pointer_buffer(device, &buffer, data_ptr, size);
    if (error == sccl_unsupported_error) {
        GTEST_SKIP() << "Skipping test, sccl_unsupported_error";
    }
    SCCL_TEST_ASSERT(error);

    /* write to buffer */
    memcpy(data_ptr, test_data.data(), test_data.size() * sizeof(uint32_t));

    /* read from buffer */
    ASSERT_EQ(
        memcmp(data_ptr, test_data.data(), test_data.size() * sizeof(uint32_t)),
        0);

    /* cleanup */
    sccl_destroy_buffer(buffer);
    free(data_ptr);
}

TEST_F(buffer_test, create_dmabuf_buffer)
{
    /* query import alignment requirement */
    sccl_device_properties_t device_properties = {};
    sccl_get_device_properties(device, &device_properties);

    const std::vector<uint32_t> test_data = {0, 1, 2, 3, 4, 5, 6, 7};
    const size_t size =
        device_properties.min_external_buffer_host_pointer_alignment;

    sccl_buffer_t buffer;

    sccl_error_t error = sccl_create_dmabuf_buffer(
        device, &buffer, sccl_buffer_type_host_uniform, size);
    if (error == sccl_unsupported_error) {
        GTEST_SKIP() << "Skipping test, sccl_unsupported_error";
    }
    SCCL_TEST_ASSERT(error);

    int fd;
    SCCL_TEST_ASSERT(sccl_export_dmabuf_buffer(buffer, &fd));

    sccl_buffer_t import_buffer;
    SCCL_TEST_ASSERT(sccl_import_dmabuf_buffer(
        device, &import_buffer, fd, sccl_buffer_type_host_uniform, size));

    /* map initial buffer and write to it */
    void *data_ptr = nullptr;
    SCCL_TEST_ASSERT(sccl_host_map_buffer(buffer, &data_ptr, 0, size));
    memset(data_ptr, 0, size);
    memcpy(data_ptr, test_data.data(), test_data.size() * sizeof(uint32_t));
    sccl_host_unmap_buffer(buffer);

    /* map imported buffer and read from it */
    /* read from buffer */
    SCCL_TEST_ASSERT(sccl_host_map_buffer(import_buffer, &data_ptr, 0, size));
    ASSERT_EQ(
        memcmp(data_ptr, test_data.data(), test_data.size() * sizeof(uint32_t)),
        0);
    sccl_host_unmap_buffer(import_buffer);

    /* cleanup */
    sccl_destroy_buffer(import_buffer);
    close(fd);
    sccl_destroy_buffer(buffer);
}