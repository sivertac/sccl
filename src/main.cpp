#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const char* COMPUTE_SHADER_PATH = "shaders/test_compute_shader.spv";

VkInstance createInstance() {
    VkInstance instance = VK_NULL_HANDLE;

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Compute Shader Example";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&createInfo, NULL, &instance) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create Vulkan instance!\n");
        exit(EXIT_FAILURE);
    }

    return instance;
}

VkShaderModule createShaderModule(VkDevice device, const char* filePath) {
    // Read the shader code from the file
    FILE* file = fopen(filePath, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open shader file: %s\n", filePath);
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    rewind(file);
    char* shaderCode = (char*)malloc(fileSize);
    fread(shaderCode, 1, fileSize, file);
    fclose(file);

    // Create shader module
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = fileSize;
    createInfo.pCode = (const uint32_t*)shaderCode;

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, NULL, &shaderModule) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create shader module for file: %s\n", filePath);
        exit(EXIT_FAILURE);
    }

    free(shaderCode);
    return shaderModule;
}

class ComputeShaderApp {
public:
    void run() {
        initWindow();
        m_instance = createInstance();
        pickPhysicalDevice();
        createLogicalDevice();
        createDescriptorSetLayout();
        createComputePipeline();
        createCommandBuffer();

        while (!glfwWindowShouldClose(m_window)) {
            glfwPollEvents();
            runComputeShader(1, 1, 1);
        }

        vkDestroyPipeline(m_device, m_computePipeline, NULL);
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, NULL);
        vkDestroyCommandPool(m_device, m_commandPool, NULL);
        vkDestroyDevice(m_device, NULL);
        vkDestroyInstance(m_instance, NULL);
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }

private:
    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        m_window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Compute Shader Example", NULL, NULL);
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(m_instance, &deviceCount, NULL);
        
        if (deviceCount == 0) {
            fprintf(stderr, "No Vulkan-compatible physical devices found!\n");
            exit(EXIT_FAILURE);
        }

        VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(deviceCount * sizeof(VkPhysicalDevice));
        vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices);

        // print device names and memory sizes
        for (uint32_t i = 0; i < deviceCount; i++) {
            VkPhysicalDeviceProperties deviceProperties;
            vkGetPhysicalDeviceProperties(devices[i], &deviceProperties);
            printf("Device %d: %s\n", i, deviceProperties.deviceName);
        }
        
        printf("Selecting device at index 0\n");
        m_physicalDevice = devices[0]; // Select the first available device

        free(devices);
    }

    void createLogicalDevice() {
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

        VkDevice vkDevice;
        if (vkCreateDevice(m_physicalDevice, &deviceCreateInfo, NULL, &vkDevice) != VK_SUCCESS) {
            fprintf(stderr, "Failed to create logical device!\n");
            exit(EXIT_FAILURE);
        }

        vkGetDeviceQueue(vkDevice, 0, 0, &m_computeQueue); // Queue family index 0, queue index 0

        m_device = vkDevice;
    }

    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding inputBinding = {};
        inputBinding.binding = 0;
        inputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        inputBinding.descriptorCount = 1;
        inputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding outputBinding = {};
        outputBinding.binding = 1;
        outputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        outputBinding.descriptorCount = 1;
        outputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding bindings[] = {inputBinding, outputBinding};

        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 2; // Number of bindings
        layoutInfo.pBindings = bindings;

        if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, NULL, &m_descriptorSetLayout) != VK_SUCCESS) {
            fprintf(stderr, "Failed to create descriptor set layout!\n");
            exit(EXIT_FAILURE);
        }
    }

    void createComputePipeline() {
        VkPipelineLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &m_descriptorSetLayout; // Use the descriptor set layout you created

        if (vkCreatePipelineLayout(m_device, &layoutInfo, NULL, &m_pipelineLayout) != VK_SUCCESS) {
            fprintf(stderr, "Failed to create pipeline layout!\n");
            exit(EXIT_FAILURE);
        }

        VkComputePipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = m_pipelineLayout;
        
        VkPipelineShaderStageCreateInfo stageInfo = {};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = createShaderModule(m_device, COMPUTE_SHADER_PATH); // You need to implement createShaderModule
        stageInfo.pName = "main";

        pipelineInfo.stage = stageInfo;

        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &m_computePipeline) != VK_SUCCESS) {
            fprintf(stderr, "Failed to create compute pipeline!\n");
            exit(EXIT_FAILURE);
        }
    }

    void createCommandBuffer() {
        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = 0; // Replace with the appropriate queue family index
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (vkCreateCommandPool(m_device, &poolInfo, NULL, &m_commandPool) != VK_SUCCESS) {
            fprintf(stderr, "Failed to create command pool!\n");
            exit(EXIT_FAILURE);
        }

        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = m_commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(m_device, &allocInfo, &m_commandBuffer) != VK_SUCCESS) {
            fprintf(stderr, "Failed to allocate command buffer!\n");
            exit(EXIT_FAILURE);
        }
    }

    void runComputeShader(uint32_t numGroupsX, uint32_t numGroupsY, uint32_t numGroupsZ) {
        vkResetCommandBuffer(m_commandBuffer, 0);

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(m_commandBuffer, &beginInfo);

        // Bind the compute pipeline
        vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);

        // Bind descriptor sets if needed
        // ...

        // Dispatch the compute shader
        vkCmdDispatch(m_commandBuffer, numGroupsX, numGroupsY, numGroupsZ);

        vkEndCommandBuffer(m_commandBuffer);

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_commandBuffer;

        if (vkQueueSubmit(m_computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            fprintf(stderr, "Failed to submit compute command buffer!\n");
            exit(EXIT_FAILURE);
        }

        vkQueueWaitIdle(m_computeQueue); // Wait for the compute queue to finish
    }

    GLFWwindow* m_window;
    VkInstance m_instance;
    VkPhysicalDevice m_physicalDevice;
    VkDevice m_device;
    VkQueue m_computeQueue;
    VkDescriptorSetLayout m_descriptorSetLayout;
    VkPipeline m_computePipeline;
    VkPipelineLayout m_pipelineLayout;
    VkCommandPool m_commandPool;
    VkCommandBuffer m_commandBuffer;
};

/*

void createDataBuffers(VkDevice device, VkBuffer* inputBuffer, VkDeviceMemory* inputBufferMemory, VkBuffer* outputBuffer, VkDeviceMemory* outputBufferMemory, size_t numElements) {
    // Create input and output buffers

    VkDeviceSize bufferSize = sizeof(int) * numElements;

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSize;

    vkCreateBuffer(device, &bufferInfo, NULL, inputBuffer);
    vkCreateBuffer(device, &bufferInfo, NULL, outputBuffer);

    // Allocate memory for the buffers

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, *inputBuffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;

    // Find a memory type index that fits the properties of the buffer
    // ...
    allocInfo.memoryTypeIndex = 0; // Replace with the appropriate memory type index

    vkAllocateMemory(device, &allocInfo, NULL, inputBufferMemory);
    vkBindBufferMemory(device, *inputBuffer, *inputBufferMemory, 0);

    vkAllocateMemory(device, &allocInfo, NULL, outputBufferMemory);
    vkBindBufferMemory(device, *outputBuffer, *outputBufferMemory, 0);

    // Map and fill the buffers
    void* inputDataPtr;
    void* outputDataPtr;

    vkMapMemory(device, *inputBufferMemory, 0, bufferSize, 0, &inputDataPtr);
    vkMapMemory(device, *outputBufferMemory, 0, bufferSize, 0, &outputDataPtr);

    // Fill inputDataPtr and outputDataPtr with your data

    vkUnmapMemory(device, *inputBufferMemory);
    vkUnmapMemory(device, *outputBufferMemory);

}
*/

int main() {
    ComputeShaderApp app;
    app.run();

    return EXIT_SUCCESS;
}