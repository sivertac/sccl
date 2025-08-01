# SCCL
Sivert Collective Compute Library (SCCL) is a wrapper library for simplifying writing compute applications in Vulkan.

## Build

### Dependencies
* Linux
* CMake
* C17 compatible compiler
* C++20 compatible compiler
* Vulkan SDK https://vulkan.lunarg.com/
* clang-tidy-19

### Run tests
Run this is build directory:
```
make -j $(nproc) && SCCL_ENABLE_VALIDATION_LAYERS=1 SCCL_ASSERT_ON_VALIDATION_ERROR=1 SCCL_TEST_GPU_INDEX=0 make test_verbose  && cat test/Testing/Temporary/LastTest.log 
```

`SCCL_TEST_PLATFORM_DOCKER=1` will disable dmabuf tests, this is useful for docker containers, as dmabuf does not work well inside containers.

## Pre-commit
https://pre-commit.com/
```
pip install pre-commit
```
or
```
apt-get install pre-commit
pre-commit install
```

## Interesting ideas for SCCL

### External host memory

Use `VkImportMemoryHostPointerInfoEXT` to allow user provided memory for buffers, instead of Vulkan allocating and mapping host memory.

Links:
* https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_external_memory_host.html

### Direct device to device transfers

Use `VK_EXT_external_memory_dma_buf` to manually control DMA transfers, might be possible to do direct device to device.

```
/* interesting stuff in this header */
#include <linux/dma-buf.h>
```

~~Use `VK_KHR_device_group` to enable direct device to device transfers.~~

Links:
* https://docs.kernel.org/driver-api/dma-buf.html
* https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_external_memory_dma_buf.html
* https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_device_group.html

### 64-bit device buffer addresses

Use `VK_KHR_buffer_device_address` to enable 64-bit buffer addresses in shaders. 

Links:
* https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_buffer_device_address.html
* https://docs.vulkan.org/samples/latest/samples/extensions/buffer_device_address/README.html
