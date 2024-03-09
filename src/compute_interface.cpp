#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "compute_interface.hpp"

namespace {

const char* validation_layers[] = {
    "VK_LAYER_KHRONOS_validation"
};
const size_t num_validation_layers = 1;

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks *pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void *pUserData) {
    (void)messageSeverity;
    (void)messageType;
    (void)pUserData;
    fprintf(stderr, "validation layer: %s\n", pCallbackData->pMessage);

    return VK_FALSE;
}

void populateDebugMessengerCreateInfo(
    VkDebugUtilsMessengerCreateInfoEXT &createInfo) {
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
}

VkResult setupDebugMessenger(VkInstance instance, VkDebugUtilsMessengerEXT* debug_messenger) {
    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);

    return CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, debug_messenger);
}

bool checkValidationLayerSupport() {
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

    VkLayerProperties* available_layers = NULL;
    available_layers = (VkLayerProperties*)malloc(layer_count * sizeof(VkLayerProperties));

    vkEnumerateInstanceLayerProperties(&layer_count, available_layers);

    for (const char *layer_name : validation_layers) {
        bool layer_found = false;

        for (size_t i = 0; i < layer_count; ++i) {
            VkLayerProperties* layer_properties = &available_layers[i];
            if (strcmp(layer_name, layer_properties->layerName) == 0) {
                layer_found = true;
                break;
            }
        }

        if (!layer_found) {
            free(available_layers);
            return false;
        }
    }

    return true;
}

VkResult createInstance(bool enable_validation_layers, VkInstance* instance) {
    *instance = VK_NULL_HANDLE;

    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Compute Shader Example";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "No Engine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    if (enable_validation_layers) {
        const char* extention_names[] = {
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME
        };
        create_info.enabledExtensionCount = 1;
        create_info.ppEnabledExtensionNames = extention_names;
        create_info.enabledLayerCount = num_validation_layers;
        create_info.ppEnabledLayerNames = validation_layers;
    }

    return vkCreateInstance(&create_info, NULL, instance);
}

VkResult pickPhysicalDevice(VkInstance instance, VkPhysicalDevice* physical_device) {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
    
    if (deviceCount == 0) {
        fprintf(stderr, "No Vulkan-compatible physical devices found!\n");
        return VK_ERROR_UNKNOWN;
    }

    VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(deviceCount * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices);

    // print device names and memory sizes
    for (uint32_t i = 0; i < deviceCount; i++) {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(devices[i], &deviceProperties);
        printf("Device %d: %s\n", i, deviceProperties.deviceName);
    }
    
    printf("Selecting device at index 0\n");
    *physical_device = devices[0]; // Select the first available device

    free(devices);
}

VkResult createLogicalDevice(VkPhysicalDevice physical_device, VkDevice* device) {
    float queuePriority = 1.0f; // Priority of the compute queue (0.0 to 1.0)

    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = 0; // Assuming compute queue is in the first family
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    return vkCreateDevice(physical_device, &deviceCreateInfo, NULL, device);
}

VkResult createShaderModule(VkDevice device, const char* shader_source, size_t shader_source_size, VkShaderModule* shader_module) {
    // Create shader module
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = shader_source_size;
    createInfo.pCode = (const uint32_t*)shader_source;

    return vkCreateShaderModule(device, &createInfo, NULL, shader_module);
}

VkResult createDescriptorSetLayout(VkDevice device, uint32_t num_input_descriptors, uint32_t num_output_descriptors, VkDescriptorSetLayout* descriptor_set_layout) {
    VkDescriptorSetLayoutBinding inputBinding = {};
    inputBinding.binding = 0;
    inputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    inputBinding.descriptorCount = num_input_descriptors;
    inputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding outputBinding = {};
    outputBinding.binding = 1;
    outputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    outputBinding.descriptorCount = num_output_descriptors;
    outputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding bindings[] = {inputBinding, outputBinding};

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2; // Number of bindings
    layoutInfo.pBindings = bindings;

    return vkCreateDescriptorSetLayout(device, &layoutInfo, NULL, descriptor_set_layout);
}

VkResult createDescriptorPool(VkDevice device, VkDescriptorPoolSize* descriptor_pool_sizes, uint32_t num_descriptor_pool_sizes, uint32_t max_sets, VkDescriptorPool* descriptor_pool) {
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = num_descriptor_pool_sizes;
    poolInfo.pPoolSizes = descriptor_pool_sizes;
    poolInfo.maxSets = max_sets;

    return vkCreateDescriptorPool(device, &poolInfo, nullptr, descriptor_pool);
}

VkResult createComputePipeline(VkDevice device, VkDescriptorSetLayout descriptor_set_layout, VkShaderModule shader_module, VkPipelineLayout* pipeline_layout, VkPipeline* compute_pipeline) {
    VkResult res = VK_SUCCESS;

    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &descriptor_set_layout; // Use the descriptor set layout you created

    res = vkCreatePipelineLayout(device, &layoutInfo, NULL, pipeline_layout);
    if (res != VK_SUCCESS) {
        return res;
    }

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = *pipeline_layout;
    
    VkPipelineShaderStageCreateInfo stageInfo = {};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shader_module;
    stageInfo.pName = "main";

    pipelineInfo.stage = stageInfo;

    res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, compute_pipeline);
    if (res != VK_SUCCESS) {
        return res;
    }

    return res;
}

VkResult createCommandBuffer(VkDevice device, VkCommandPool* command_pool, VkCommandBuffer* command_buffer) {
    VkResult res = VK_SUCCESS;
    
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = 0; // Replace with the appropriate queue family index
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    res = vkCreateCommandPool(device, &poolInfo, NULL, command_pool); 
    if (res != VK_SUCCESS) {
        return res;
    }

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = *command_pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    res = vkAllocateCommandBuffers(device, &allocInfo, command_buffer);
    if (res != VK_SUCCESS) {
        return res;
    }

    return res;
}

VkResult findMemoryType(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties, uint32_t* output_index) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            *output_index = i;
            return VK_SUCCESS;
        }
    }

    return VK_ERROR_UNKNOWN;
}

} // namespace

VkResult createComputeDevice(bool enable_validation_layers, ComputeDevice* compute_device) {
    VkResult res = VK_SUCCESS;
    
    // zero init
    *compute_device = {};

    // check if validation layers are supported
    compute_device->m_validation_layers_enabled = enable_validation_layers;
    if (compute_device->m_validation_layers_enabled) {
        if (!checkValidationLayerSupport()) {
            fprintf(stderr, "Validation layers not supported\n");
            return VK_ERROR_UNKNOWN;
        }
    }

    // get vulkan instance
    res = createInstance(enable_validation_layers, &compute_device->m_instance);
    if (res != VK_SUCCESS) {
        return res;
    }

    // enable validation layers
    if (compute_device->m_validation_layers_enabled) {
        setupDebugMessenger(compute_device->m_instance, &compute_device->m_debug_messenger);
    }

    // pick physical device
    res = pickPhysicalDevice(compute_device->m_instance, &compute_device->m_physical_device);
    if (res != VK_SUCCESS) {
        return res;
    }

    // create logical device
    res = createLogicalDevice(compute_device->m_physical_device, &compute_device->m_device);
    if (res != VK_SUCCESS) {
        return res;
    }

    return res;
}

void destroyComputeDevice(ComputeDevice* compute_device) {
    vkDestroyDevice(compute_device->m_device, NULL);

    if (compute_device->m_validation_layers_enabled) {
        DestroyDebugUtilsMessengerEXT(compute_device->m_instance, compute_device->m_debug_messenger, NULL);
    }

    vkDestroyInstance(compute_device->m_instance, NULL);
}

VkResult createComputePipeline(
    const ComputeDevice* compute_device, 
    const char* shader_source, 
    const size_t shader_source_size, 
    const int num_input_buffers, 
    const int num_output_buffers,
    ComputePipeline* compute_pipeline
) {
    VkResult res = VK_SUCCESS;

    // zero init
    *compute_pipeline = {};

    // get queue
    vkGetDeviceQueue(compute_device->m_device, 0, 0, &compute_pipeline->m_queue); // Queue family index 0, queue index 0

    // create descriptor set layout
    res = createDescriptorSetLayout(compute_device->m_device, num_input_buffers, num_output_buffers, &compute_pipeline->m_descriptor_set_layout);
    if (res != VK_SUCCESS) {
        return res;
    }

    // create descriptor pool
    const int num_descriptor_types = 2;
    VkDescriptorPoolSize descriptor_pool_sizes[num_descriptor_types] = {}; 
    descriptor_pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptor_pool_sizes[0].descriptorCount = static_cast<uint32_t>(num_input_buffers + num_output_buffers);
    descriptor_pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_pool_sizes[1].descriptorCount = static_cast<uint32_t>(num_input_buffers + num_output_buffers);
    res = createDescriptorPool(compute_device->m_device, descriptor_pool_sizes, num_descriptor_types, 1, &compute_pipeline->m_descriptor_pool);
    if (res != VK_SUCCESS) {
        return res;
    }

    // create shader module
    res = createShaderModule(compute_device->m_device, shader_source, shader_source_size, &compute_pipeline->m_shader_module);
    if (res != VK_SUCCESS) {
        return res;
    }

    // create compute pipeline
    res = createComputePipeline(compute_device->m_device, compute_pipeline->m_descriptor_set_layout, compute_pipeline->m_shader_module, &compute_pipeline->m_pipeline_layout, &compute_pipeline->m_compute_pipeline);
    if (res != VK_SUCCESS) {
        return res;
    }

    // create command buffer
    res = createCommandBuffer(compute_device->m_device, &compute_pipeline->m_command_pool, &compute_pipeline->m_command_buffer);
    if (res != VK_SUCCESS) {
        return res;
    }

    return res;
}

VkResult updateComputeDescriptorSet(
    const ComputeDevice* compute_device, 
    const ComputeBuffer* input_buffers,
    const int num_input_buffers,
    const ComputeBuffer* output_buffers,
    const int num_output_buffers,    
    ComputeDescriptorSet* compute_descriptor_set
) {
    VkResult res = VK_SUCCESS;

    // Update descriptor sets
    VkWriteDescriptorSet* descriptor_writes = NULL;
    size_t num_descriptor_writes = num_input_buffers + num_output_buffers;
    descriptor_writes = (VkWriteDescriptorSet*)malloc(num_descriptor_writes * sizeof(VkWriteDescriptorSet));
    if (descriptor_writes == NULL) {
        return VK_ERROR_UNKNOWN;
    }
    memset(descriptor_writes, 0, num_descriptor_writes * sizeof(VkWriteDescriptorSet));

    for (size_t i = 0; i < num_input_buffers; ++i) {
        size_t write_index = i;

        VkDescriptorBufferInfo buffer_info = {};
        buffer_info.buffer = input_buffers[i].m_buffer;
        buffer_info.offset = 0;
        buffer_info.range = input_buffers[i].m_size;

        descriptor_writes[write_index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[write_index].dstSet = compute_descriptor_set->m_descriptor_set;
        descriptor_writes[write_index].dstBinding = write_index;
        descriptor_writes[write_index].dstArrayElement = 0;
        descriptor_writes[write_index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptor_writes[write_index].descriptorCount = 1;
        descriptor_writes[write_index].pBufferInfo = &buffer_info;
    }
    for (size_t i = 0; i < num_output_buffers; ++i) {
        size_t write_index = i + num_input_buffers;
        
        VkDescriptorBufferInfo buffer_info = {};
        buffer_info.buffer = output_buffers[i].m_buffer;
        buffer_info.offset = 0;
        buffer_info.range = output_buffers[i].m_size;

        descriptor_writes[write_index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[write_index].dstSet = compute_descriptor_set->m_descriptor_set;
        descriptor_writes[write_index].dstBinding = write_index;
        descriptor_writes[write_index].dstArrayElement = 0;
        descriptor_writes[write_index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptor_writes[write_index].descriptorCount = 1;
        descriptor_writes[write_index].pBufferInfo = &buffer_info;
    }

    vkUpdateDescriptorSets(compute_device->m_device, num_descriptor_writes, descriptor_writes, 0, nullptr);

    free(descriptor_writes);

    return res;
}

VkResult runComputePipelineSync(
    const ComputeDevice* compute_device,
    const ComputePipeline* compute_pipeline,
    const ComputeDescriptorSet* compute_descriptor_set,
    const uint32_t group_count_x,
    const uint32_t group_count_y,
    const uint32_t group_count_z
) {
    VkResult res = VK_SUCCESS;

    vkResetCommandBuffer(compute_pipeline->m_command_buffer, 0);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(compute_pipeline->m_command_buffer, &begin_info);

    // Bind the compute pipeline
    vkCmdBindPipeline(compute_pipeline->m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline->m_compute_pipeline);

    // Bind descriptor sets
    vkCmdBindDescriptorSets(compute_pipeline->m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline->m_pipeline_layout, 0, 1, &compute_descriptor_set->m_descriptor_set, 0, nullptr);

    // Dispatch the compute shader
    vkCmdDispatch(compute_pipeline->m_command_buffer, group_count_x, group_count_y, group_count_z);

    vkEndCommandBuffer(compute_pipeline->m_command_buffer);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &compute_pipeline->m_command_buffer;

    res = vkQueueSubmit(compute_pipeline->m_queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (res != VK_SUCCESS) {
        return res;
    }

    vkQueueWaitIdle(compute_pipeline->m_queue); // Wait for the compute queue to finish

    return res;
}

void destroyComputePipeline(const ComputeDevice* compute_device, ComputePipeline* compute_pipeline) {
    vkDestroyDescriptorPool(compute_device->m_device, compute_pipeline->m_descriptor_pool, VK_NULL_HANDLE);
    vkDestroyDescriptorSetLayout(compute_device->m_device, compute_pipeline->m_descriptor_set_layout, VK_NULL_HANDLE);
    vkDestroyPipeline(compute_device->m_device, compute_pipeline->m_compute_pipeline, NULL);
    vkDestroyPipelineLayout(compute_device->m_device, compute_pipeline->m_pipeline_layout, NULL);
    vkDestroyShaderModule(compute_device->m_device, compute_pipeline->m_shader_module, VK_NULL_HANDLE);
    vkFreeCommandBuffers(compute_device->m_device, compute_pipeline->m_command_pool, 1, &compute_pipeline->m_command_buffer);
    vkDestroyCommandPool(compute_device->m_device, compute_pipeline->m_command_pool, NULL);
}

VkResult createComputeDescriptorSet(const ComputeDevice* compute_device, const ComputePipeline* compute_pipeline, ComputeDescriptorSet* compute_descriptor_set) {
    VkResult res = VK_SUCCESS;

    // zero init
    *compute_descriptor_set = {};

    // Create descriptor set
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = compute_pipeline->m_descriptor_pool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(1);
    allocInfo.pSetLayouts = &compute_pipeline->m_descriptor_set_layout;
    
    res = vkAllocateDescriptorSets(compute_device->m_device, &allocInfo, &compute_descriptor_set->m_descriptor_set);
    if (res != VK_SUCCESS) {
        return res;
    }

    return res;
}

VkResult createComputeBuffer(const ComputeDevice* compute_device, VkDeviceSize size, ComputeBuffer* compute_buffer) {
    VkResult res = VK_SUCCESS;

    // zero init
    *compute_buffer = {};

    // create buffer
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    res = vkCreateBuffer(compute_device->m_device, &buffer_info, nullptr, &compute_buffer->m_buffer);
    if (res != VK_SUCCESS) {
        return res;
    }

    // get memory requirements
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(compute_device->m_device, compute_buffer->m_buffer, &mem_requirements);

    // find memory type
    uint32_t memory_type_index;
    res = findMemoryType(compute_device->m_physical_device, mem_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &memory_type_index);
    if (res != VK_SUCCESS) {
        return res;
    }

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = memory_type_index;

    res = vkAllocateMemory(compute_device->m_device, &alloc_info, nullptr, &compute_buffer->m_buffer_memory);
    if (res != VK_SUCCESS) {
        return res;
    }

    res = vkBindBufferMemory(compute_device->m_device, compute_buffer->m_buffer, compute_buffer->m_buffer_memory, 0);
    if (res != VK_SUCCESS) {
        return res;
    }

    compute_buffer->m_size = size;

    return res;
}

void destroyComputeBuffer(const ComputeDevice* compute_device, ComputeBuffer* compute_buffer) {
    vkFreeMemory(compute_device->m_device, compute_buffer->m_buffer_memory, VK_NULL_HANDLE);
    vkDestroyBuffer(compute_device->m_device, compute_buffer->m_buffer, VK_NULL_HANDLE);
}

