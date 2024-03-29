#pragma once
#ifndef SCCL_VECTOR_HEADER
#define SCCL_VECTOR_HEADER

#include "sccl.h"

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    void *data;
    size_t size;
    size_t capacity;
    size_t element_size;
} vector_t;

sccl_error_t vector_init(vector_t *vec, size_t element_size);

sccl_error_t vector_add_element(vector_t *vec, const void *element);

void vector_clear(vector_t *vec);

size_t vector_get_size(const vector_t *vec);

void *vector_get_element(const vector_t *vec, size_t index);

void vector_destroy(vector_t *vec);

void vector_sort(vector_t *vec, int (*compar)(const void *, const void *));

#endif // SCCL_VECTOR_HEADER