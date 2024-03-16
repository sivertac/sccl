
#include "alloc.h"
#include <stdlib.h>

sccl_error_t sccl_calloc(void **ptr, size_t nmem, size_t size)
{
    *ptr = calloc(nmem, size);
    if (*ptr == NULL) {
        return sccl_system_error;
    }
    return sccl_success;
}

void sccl_free(void *ptr) { free(ptr); }
