

#include <sccl.h>

#include "common.hpp"
#include <gtest/gtest.h>

class stream_test : public testing::Test
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

TEST_F(stream_test, create_stream)
{
    sccl_stream_t stream;
    EXPECT_EQ(sccl_create_stream(device, &stream), sccl_success);
    sccl_destroy_stream(stream);
}

TEST_F(stream_test, dispatch_and_join)
{
    sccl_stream_t stream;

    EXPECT_EQ(sccl_create_stream(device, &stream), sccl_success);

    EXPECT_EQ(sccl_dispatch_stream(stream), sccl_success);

    EXPECT_EQ(sccl_join_stream(stream), sccl_success);

    EXPECT_EQ(sccl_dispatch_stream(stream), sccl_success);

    EXPECT_EQ(sccl_join_stream(stream), sccl_success);

    sccl_destroy_stream(stream);
}

TEST_F(stream_test, dispatch_and_join_multiple)
{
    /* arbitrary number of streams */
    const size_t stream_count = 10;
    std::vector<sccl_stream_t> streams;

    for (size_t i = 0; i < stream_count; ++i) {
        streams.push_back({});
        EXPECT_EQ(sccl_create_stream(device, &streams.back()), sccl_success);
    }

    for (sccl_stream_t stream : streams) {
        EXPECT_EQ(sccl_dispatch_stream(stream), sccl_success);
    }

    for (sccl_stream_t stream : streams) {
        EXPECT_EQ(sccl_join_stream(stream), sccl_success);
    }

    for (sccl_stream_t stream : streams) {
        sccl_destroy_stream(stream);
    }
}

TEST_F(stream_test, dispatch_and_wait)
{
    /* arbitrary number of streams */
    const size_t stream_count = 10;
    std::vector<sccl_stream_t> streams;

    for (size_t i = 0; i < stream_count; ++i) {
        streams.push_back({});
        EXPECT_EQ(sccl_create_stream(device, &streams.back()), sccl_success);
    }

    for (sccl_stream_t stream : streams) {
        EXPECT_EQ(sccl_dispatch_stream(stream), sccl_success);
    }

    /* wait for first to trigger for actual test */
    EXPECT_EQ(sccl_wait_streams(device, streams.data(), stream_count, NULL),
              sccl_success);

    /* wait for rest so valitaion layer won't complain about unsignaled fences
     */
    EXPECT_EQ(sccl_wait_streams_all(device, streams.data(), stream_count),
              sccl_success);

    for (sccl_stream_t stream : streams) {
        sccl_destroy_stream(stream);
    }
}

TEST_F(stream_test, dispatch_and_wait_completed_list)
{
    /* arbitrary number of streams */
    const size_t stream_count = 10;
    std::vector<sccl_stream_t> streams;
    std::vector<uint8_t> completed_list(stream_count);

    for (size_t i = 0; i < stream_count; ++i) {
        streams.push_back({});
        EXPECT_EQ(sccl_create_stream(device, &streams.back()), sccl_success);
    }

    for (sccl_stream_t stream : streams) {
        EXPECT_EQ(sccl_dispatch_stream(stream), sccl_success);
    }

    /* wait */
    while (true) {
        EXPECT_EQ(sccl_wait_streams(device, streams.data(), stream_count,
                                    completed_list.data()),
                  sccl_success);
        bool not_complete = false;
        for (size_t i = 0; i < stream_count; ++i) {
            if (completed_list[i] == 0) {
                not_complete = true;
            }
        }
        if (!not_complete) {
            break;
        }
    }

    /* check if all streams are complete */
    for (size_t i = 0; i < stream_count; ++i) {
        EXPECT_FALSE(completed_list[i] == 0);
    }

    for (sccl_stream_t stream : streams) {
        sccl_destroy_stream(stream);
    }
}

TEST_F(stream_test, dispatch_and_wait_all)
{
    /* arbitrary number of streams */
    const size_t stream_count = 10;
    std::vector<sccl_stream_t> streams;

    for (size_t i = 0; i < stream_count; ++i) {
        streams.push_back({});
        EXPECT_EQ(sccl_create_stream(device, &streams.back()), sccl_success);
    }

    for (sccl_stream_t stream : streams) {
        EXPECT_EQ(sccl_dispatch_stream(stream), sccl_success);
    }

    EXPECT_EQ(sccl_wait_streams_all(device, streams.data(), stream_count),
              sccl_success);

    for (sccl_stream_t stream : streams) {
        sccl_destroy_stream(stream);
    }
}
