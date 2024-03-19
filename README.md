# Vulkan Compute Shader Test

# SCCL
Sivert Collective Compute Library (SCCL) is a wrapper library for simplifying writing compute applications in Vulkan.

## Interesting ideas for SCCL

### External host memory

Use `VkImportMemoryHostPointerInfoEXT` to allow user provided memory for buffers, instead of Vulkan allocating and mapping host memory.

Links:
* https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_external_memory_host.html

### Direct device to device transfers

Use `VK_KHR_device_group` to enable direct device to device transfers.

Links: 
* https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_device_group.html

### 64-bit device buffer addresses

Use `VK_KHR_buffer_device_address` to enable 64-bit buffer addresses in shaders. 

Links:
* https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_buffer_device_address.html
* https://docs.vulkan.org/samples/latest/samples/extensions/buffer_device_address/README.html


## Build

### Dependencies
* CMake
* C17 compatible compiler
* C++20 compatible compiler
* Vulkan SDK https://vulkan.lunarg.com/

### Run tests
Run this is build directory:
```
make -j $(nproc) && SCCL_ENABLE_VALIDATION_LAYERS=1 SCCL_ASSERT_ON_VALIDATION_ERROR=1 GPU_INDEX=0 make test_verbose  && cat test/Testing/Temporary/LastTest.log 
```

### Pre-commit
https://pre-commit.com/
```
pip install pre-commit
```
