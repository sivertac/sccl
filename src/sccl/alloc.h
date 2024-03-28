#pragma once
#ifndef ALLOC_HEADER
#define ALLOC_HEADER

#include "error.h"

/**
 * Internal heap alloc.
 * Memory returned is zero initilized.
 */
sccl_error_t sccl_calloc(void **ptr, size_t nmem, size_t size);

sccl_error_t sccl_reallocarray(void **ptr, size_t nmemb, size_t size);

void sccl_free(void *ptr);

#endif // ALLOC_HEADER
