

#include <sccl.h>

#include "common.hpp"
#include <gtest/gtest.h>

TEST(sccl_instance, create_instance)
{

    sccl_instance_t instance;
    SCCL_TEST_ASSERT(sccl_create_instance(&instance));

    sccl_destroy_instance(instance);
}
