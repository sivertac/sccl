

#include <sccl.h>

#include "common.hpp"
#include <gtest/gtest.h>

class stream_test : public testing::Test
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

TEST_F(stream_test, create_stream)
{
    sccl_stream_t stream;
    SCCL_TEST_ASSERT(sccl_create_stream(device, &stream));
    sccl_destroy_stream(stream);
}

TEST_F(stream_test, dispatch_and_join)
{
    sccl_stream_t stream;

    SCCL_TEST_ASSERT(sccl_create_stream(device, &stream));

    SCCL_TEST_ASSERT(sccl_dispatch_stream(stream));

    SCCL_TEST_ASSERT(sccl_join_stream(stream));

    SCCL_TEST_ASSERT(sccl_dispatch_stream(stream));

    SCCL_TEST_ASSERT(sccl_join_stream(stream));

    sccl_destroy_stream(stream);
}

TEST_F(stream_test, dispatch_and_join_multiple)
{
    /* arbitrary number of streams */
    const size_t stream_count = 10;
    std::vector<sccl_stream_t> streams;

    for (size_t i = 0; i < stream_count; ++i) {
        streams.push_back({});
        SCCL_TEST_ASSERT(sccl_create_stream(device, &streams.back()));
    }

    for (sccl_stream_t stream : streams) {
        SCCL_TEST_ASSERT(sccl_dispatch_stream(stream));
    }

    for (sccl_stream_t stream : streams) {
        SCCL_TEST_ASSERT(sccl_join_stream(stream));
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
        SCCL_TEST_ASSERT(sccl_create_stream(device, &streams.back()));
    }

    for (sccl_stream_t stream : streams) {
        SCCL_TEST_ASSERT(sccl_dispatch_stream(stream));
    }

    /* wait for first to trigger for actual test */
    SCCL_TEST_ASSERT(
        sccl_wait_streams(device, streams.data(), stream_count, nullptr));

    /* wait for rest so valitaion layer won't complain about unsignaled fences
     */
    SCCL_TEST_ASSERT(
        sccl_wait_streams_all(device, streams.data(), stream_count));

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
        SCCL_TEST_ASSERT(sccl_create_stream(device, &streams.back()));
    }

    for (sccl_stream_t stream : streams) {
        SCCL_TEST_ASSERT(sccl_dispatch_stream(stream));
    }

    /* wait */
    while (true) {
        SCCL_TEST_ASSERT(sccl_wait_streams(device, streams.data(), stream_count,
                                           completed_list.data()));
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
        SCCL_TEST_ASSERT(sccl_create_stream(device, &streams.back()));
    }

    for (sccl_stream_t stream : streams) {
        SCCL_TEST_ASSERT(sccl_dispatch_stream(stream));
    }

    SCCL_TEST_ASSERT(
        sccl_wait_streams_all(device, streams.data(), stream_count));

    for (sccl_stream_t stream : streams) {
        sccl_destroy_stream(stream);
    }
}
