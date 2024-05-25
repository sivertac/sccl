
#include "error.h"

const char *sccl_get_error_string(sccl_error_t error)
{
    switch (error) {
    case sccl_success:
        return "Success";
    case sccl_unhandled_vulkan_error:
        return "Unhandled Vulkan error";
    case sccl_system_error:
        return "System error";
    case sccl_internal_error:
        return "Internal error";
    case sccl_invalid_argument:
        return "Invalid argument";
    case sccl_unsupported_error:
        return "Unsupported argument";
    case sccl_out_of_resources_error:
        return "Out of resources error";
    default:
        return NULL;
    }
}
