#pragma once
#ifndef COMMON_HEADER
#define COMMON_HEADER

#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

inline uint32_t get_environment_gpu_index()
{
    const char *str = getenv("GPU_INDEX");
    if (str == NULL) {
        return 0;
    }
    uint32_t gpu_index = static_cast<uint32_t>(std::atoi(str));
    printf("environment - GPU_INDEX = %" PRIu32 "\n", gpu_index);
    return gpu_index;
}

#endif // COMMON_HEADER