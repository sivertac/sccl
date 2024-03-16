
#include "enviroment_variables.h"
#include <stddef.h>
#include <stdlib.h>

bool is_enable_validation_layers_set()
{
    const char *str = getenv("ENABLE_VALIDATION_LAYERS");
    return str != NULL;
}
