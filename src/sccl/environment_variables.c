
#include "environment_variables.h"
#include "sccl.h"
#include <stddef.h>
#include <stdlib.h>

static bool parse_input(const char *str)
{
    if (str == NULL) {
        return false;
    }
    int val = atoi(str);
    return val > 0;
}

bool is_enable_validation_layers_set()
{
    const char *str = getenv(SCCL_ENABLE_VALIDATION_LAYERS);
    return parse_input(str);
}

bool is_assert_on_validation_error_set()
{
    const char *str = getenv(SCCL_ASSERT_ON_VALIDATION_ERROR);
    return parse_input(str);
}
