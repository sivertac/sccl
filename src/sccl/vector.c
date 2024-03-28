
#include "vector.h"
#include "alloc.h"
#include <stdlib.h>
#include <string.h>

sccl_error_t vector_init(vector_t *vec, size_t element_size)
{
    memset(vec, 0, sizeof(vector_t));
    vec->size = 0;
    vec->capacity = 2;
    vec->element_size = element_size;
    return sccl_calloc(&vec->data, vec->capacity, vec->element_size);
}

static void *get_element_internal(const vector_t *vec, size_t index)
{
    return (void *)(((uint8_t *)vec->data) + index * vec->element_size);
}

sccl_error_t vector_add_element(vector_t *vec, const void *element)
{
    assert(vector_get_size(vec) <= vec->capacity);
    if (vector_get_size(vec) == vec->capacity) {
        size_t new_capacity = vec->capacity * 2;
        CHECK_SCCL_ERROR_RET(
            sccl_reallocarray(&vec->data, new_capacity, vec->element_size));
        vec->capacity = new_capacity;
    }
    void *new_element = get_element_internal(vec, vector_get_size(vec));
    memcpy(new_element, element, vec->element_size);
    ++vec->size;
    return sccl_success;
}

size_t vector_get_size(const vector_t *vec) { return vec->size; }

void *vector_get_element(const vector_t *vec, size_t index)
{
    assert(vector_get_size(vec) > 0);
    assert(index < vector_get_size(vec));
    return get_element_internal(vec, index);
}

void vector_destroy(vector_t *vec)
{
    assert(vec != NULL);
    sccl_free(vec->data);
}

void vector_sort(vector_t *vec, int (*compar)(const void *, const void *))
{
    qsort(vec->data, vec->size, vec->element_size, compar);
}
