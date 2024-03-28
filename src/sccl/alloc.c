
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

sccl_error_t sccl_reallocarray(void **ptr, size_t nmemb, size_t size)
{
    void *new_ptr = NULL;
    new_ptr = reallocarray(*ptr, nmemb, size);
    if (new_ptr == NULL) {
        return sccl_system_error;
    }
    *ptr = new_ptr;
    return sccl_success;
}

void sccl_free(void *ptr) { free(ptr); }
