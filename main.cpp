#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan_core.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "main.h"
#include <glm/gtc/matrix_transform.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

VkResult CreateDebugUtilsMessengerEXT(
	VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
	const VkAllocationCallbacks *pAllocator,
	VkDebugUtilsMessengerEXT *pMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
		instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pMessenger);
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

static std::vector<char> readFile(const std::string &f) {
	std::ifstream file(f, std::ios::ate | std::ios::binary);

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();
	return buffer;
}

class HelloTriangleApplication {
  public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

  private:
	///////////////
	// Variables //
	///////////////

	// Setup //
	GLFWwindow *window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;

	// Devices //
	VkPhysicalDevice physicalDevice;
	VkDevice device;

	// Queue Families //
	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkQueue transferQueue;

	// Swap Chains //
	VkSurfaceKHR surface;
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;

	// Graphics Pipeline //
	VkRenderPass renderPass;
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;
	VkViewport viewport;
	VkRect2D scissor;

	// Drawing //
	std::vector<VkFramebuffer> swapChainFramebuffers;
	VkCommandPool graphicsPool;
	VkCommandPool transferPool;
	std::vector<VkCommandBuffer> commandBuffers;

	// Synchronisation //
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	VkFence transferFence;
	bool framebufferResized = false; // Force image resizing
	uint32_t currentFrame = 0;		 // Frames-in-flight feature

	// Model //
	std::vector<Vertex> vertices;
	std::vector<index_t> indices;

	// Index/Vertex Buffers //
	VkBuffer ivBuffer;
	VkDeviceMemory ivBufferMemory;
	VkDeviceSize verticesOffset;
	VkDeviceSize indicesOffset;

	// Uniforms //
	std::vector<VkBuffer> uniformBuffers;
	VkDeviceMemory uniformBufferMemory;
	std::vector<VkDeviceSize> uniformBufferOffsets;
	char *uniformBuffersMap;
	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;

	// Textures //
	uint32_t mipLevels;
	std::vector<VkImage> images;
	std::vector<VkDeviceSize> imageMemoryOffsets;
	VkDeviceMemory imageMemory;
	VkImageView textureImageView;
	VkSampler textureSampler;

	// Depth //
	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;

	// Multi-Sampling //
	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
	VkImage colorImage;
	VkDeviceMemory colorImageMemory;
	VkImageView colorImageView;

	////////////////////
	// Initialization //
	////////////////////

	void initVulkan() {
		// Setup
		createInstance();
		setupDebugMessenger();

		// Devices and syncs
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSyncObjects();

		// Graphics Pipeline
		createSwapChain();
		createSwapChainImageViews();
		createRenderPass();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();

		createDepthResources();
		createColorResources();

		// Textures
		createTextureImage();
		createTextureImageView();
		createTextureSampler();

		// Vertices/Indices/Uniforms
		loadModel();
		createIVBuffer();
		createUniformBuffer();
		createDescriptorPool();
		createDescriptorSets();

		createFramebuffers();
		createCommandBuffers();
	}

	void initWindow() {
		glfwInitHint(GLFW_PLATFORM, GLFW_ANY_PLATFORM);
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHintString(GLFW_X11_CLASS_NAME, "vulkan_tutorial");
		glfwWindowHintString(GLFW_WAYLAND_APP_ID, WAYLAND_APP_ID);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
		glfwWindowHint(GLFW_REFRESH_RATE, 60);

		window = glfwCreateWindow(WIDTH, HEIGHT, WINDOW_NAME, nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(
			window, [](GLFWwindow *window, int width, int height) {
				auto app = reinterpret_cast<HelloTriangleApplication *>(
					glfwGetWindowUserPointer(window));
				app->framebufferResized = true;
			});
	}

	void createInstance() {
		// Get extensions.
		auto glfwExtensions = getRequiredExtensions();

		uint32_t extensionCount = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
											   nullptr);
		std::vector<VkExtensionProperties> extensions(extensionCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
											   extensions.data());

		// Create instance, handle errors
		VkInstanceCreateInfo createInfo{
			.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pApplicationInfo = &APP_INFO,
			.enabledLayerCount = 0,
			.enabledExtensionCount =
				static_cast<uint32_t>(glfwExtensions.size()),
			.ppEnabledExtensionNames = glfwExtensions.data(),
		};

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers) {
			if (!checkValidationLayerSupport()) {
				throw std::runtime_error(
					std::string(VK_CREATE_INSTANCE_ERROR) +
					std::string(ERROR_VALIDATION_LAYERS_MSG));
			}
			createInfo.enabledLayerCount =
				static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext =
				(VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
		} else {
			createInfo.enabledLayerCount = 0;
			createInfo.pNext = nullptr;
		}
		VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

		switch (result) {
		case VK_SUCCESS:
			break;
		case VK_ERROR_EXTENSION_NOT_PRESENT:
			throw std::runtime_error(std::string(VK_CREATE_INSTANCE_ERROR) +
									 std::string(VK_ERROR_EXT_NOT_PRESENT_MSG));
		default:
			throw std::runtime_error(std::string(VK_GENERAL_ERR_MSG) +
									 std::string(VK_CREATE_INSTANCE_ERROR));
			break;
		}
	}

	// Check the support of the selected Vulkan validation layers.
	bool checkValidationLayerSupport() {
		uint32_t layerCount = 0;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char *layerName : validationLayers) {
			for (const auto &p : availableLayers) {
				if (strcmp(p.layerName, layerName) == 0) {
					goto validation_loop;
				}
			}
			return false; // layer not found
		validation_loop:  // layer found - continue
			continue;
		}
		return true;
	}

	////////////////
	// Extensions //
	////////////////

	std::vector<const char *> getRequiredExtensions() {
		uint32_t glfwExtensionCount = 0;
		const char **glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char *> extensions(
			glfwExtensions, glfwExtensions + glfwExtensionCount);
		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}
		return extensions;
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL
	debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
				  VkDebugUtilsMessageTypeFlagsEXT messageType,
				  const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
				  void *pUserData) {
		std::cerr << "[Validation layer] " << pCallbackData->pMessage
				  << std::endl;
		return VK_FALSE;
	}

	void populateDebugMessengerCreateInfo(
		VkDebugUtilsMessengerCreateInfoEXT &createInfo) {
		createInfo = {
			.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
			.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
							   VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
							   VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
							   VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
			.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
						   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
						   VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
			.pfnUserCallback = debugCallback,
			.pUserData = nullptr,
		};
	}

	void setupDebugMessenger() {
		if (!enableValidationLayers) {
			return;
		}
		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);
		VkResult res = CreateDebugUtilsMessengerEXT(instance, &createInfo,
													nullptr, &debugMessenger);

		switch (res) {
		case VK_SUCCESS:
			break;
		case VK_ERROR_EXTENSION_NOT_PRESENT:
			throw std::runtime_error(std::string(VK_CREATE_INSTANCE_ERROR) +
									 std::string(ERROR_DEBUG_MESSENGER));
		default:
			throw std::runtime_error(std::string(VK_GENERAL_ERR_MSG) +
									 std::string(ERROR_DEBUG_MESSENGER));
			break;
		}
	}

	//////////////
	// Surfaces //
	//////////////

	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
			VK_SUCCESS) {
			throw std::runtime_error(std::string(ERROR_CREATE_SURFACE));
		}
	}

	////////////////////
	// Queue Families //
	////////////////////

	QueueFamilyIndices
	selectFamilies(std::vector<VkQueueFamilyProperties> queueFamilies,
				   VkPhysicalDevice device, SelectFamilyInfo *info) {
		QueueFamilyIndices res;

		int i = 0;
		for (const auto &qf : queueFamilies) {
			if (!res.graphicsFamily.has_value() &&
				qf.queueFlags & info->graphicsDo &&
				!(qf.queueFlags & info->graphicsDont)) {
				res.graphicsFamily = i;
			}
			VkBool32 presentSupport = false;
			int result = vkGetPhysicalDeviceSurfaceSupportKHR(
				device, i, surface, &presentSupport);
			if (!res.presentFamily.has_value() && presentSupport) {
				res.presentFamily = i;
			}
			if (!res.transferFamily.has_value() &&
				qf.queueFlags & info->transferDo &&
				!(qf.queueFlags & info->transferDont)) {
				res.transferFamily = i;
			}
			if (res.isComplete()) {
				return res;
			}
			i++;
		}

		// Fallback for transfer
		for (size_t i = 0; i < queueFamilies.size(); i++) {
			auto &qf = queueFamilies[i];
			if (qf.queueFlags & info->transferDo) {
				res.transferFamily = i;
			}
			if (res.transferFamily.has_value()) {
				return res;
			}
		}
		return res;
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
		uint32_t familyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilies(familyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount,
												 queueFamilies.data());

		SelectFamilyInfo familyInfo{
			.graphicsDo = VK_QUEUE_GRAPHICS_BIT,
			.graphicsDont = 0,
			.transferDo = VK_QUEUE_TRANSFER_BIT,
			.transferDont = VK_QUEUE_GRAPHICS_BIT,
		};

		return selectFamilies(queueFamilies, device, &familyInfo);
	}

	//////////////////////
	// Physical Devices //
	//////////////////////

	// Pick a physical device to run graphical computation on this window.
	void pickPhysicalDevice() {
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
		if (!deviceCount) {
			throw std::runtime_error(std::string(ERROR_NO_DEVICES) +
									 std::string(ERROR_PICK_DEVICES));
		}
		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		std::multimap<int, VkPhysicalDevice> candidates;
		for (const auto &device : devices) {
			int score = getDeviceScore(device);
			if (score > 0) {
				candidates.insert(std::make_pair(score, device));
			}
		}
		if (!candidates.empty()) {
			physicalDevice = candidates.rend()->second;
			msaaSamples = getMaxUsableSampleCount(physicalDevice);
		}
		if (this->physicalDevice == VK_NULL_HANDLE) {
			throw std::runtime_error(std::string(ERROR_NO_SUITABLE_DEVICES) +
									 std::string(ERROR_PICK_DEVICES));
		}
	}

	// Get the score of a physical device, rated on how suitable it is.
	int getDeviceScore(VkPhysicalDevice device) {
		VkPhysicalDeviceProperties properties;
		VkPhysicalDeviceFeatures features;
		vkGetPhysicalDeviceProperties(device, &properties);
		vkGetPhysicalDeviceFeatures(device, &features);

		// Initial suitability checks
		SwapChainSupportDetails details = querySwapChainSupport(device);
		if (!features.geometryShader ||
			!findQueueFamilies(device).isComplete() ||
			!checkDeviceExtensionSupport(device, deviceExtensions) ||
			details.formats.empty() || details.presentModes.empty()) {
			return 0;
		}

		// Give score from physical device type
		int score = 0;
		switch (properties.deviceType) {
		case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
			score += 5000;
			break;
		case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
			score += 500;
			break;
		case VK_PHYSICAL_DEVICE_TYPE_CPU:
			score += 100;
			break;
		case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
			score += 50;
			break;
		default:
			score = 0;
		}
		score += properties.limits.maxImageDimension2D;
		score *= 1 + features.samplerAnisotropy;

		return score;
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice dev,
									 std::vector<const char *> extensions) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(dev, nullptr, &extensionCount,
											 nullptr);
		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(dev, nullptr, &extensionCount,
											 availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(),
												 deviceExtensions.end());
		for (const auto &ext : availableExtensions) {
			requiredExtensions.erase(ext.extensionName);
		}

		return requiredExtensions.empty();
	}

	///////////////////
	// Multisampling //
	///////////////////

	VkSampleCountFlagBits
	getMaxUsableSampleCount(VkPhysicalDevice physicalDevice) {
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(physicalDevice, &props);

		VkSampleCountFlags counts = props.limits.framebufferColorSampleCounts &
									props.limits.framebufferDepthSampleCounts;
		for (size_t i = 6; i >= 0; i--) {
			if (counts & 1 << i) {
				return (VkSampleCountFlagBits)(1 << i);
			}
		}
		return VK_SAMPLE_COUNT_1_BIT;
	}

	/////////////////////
	// Logical Devices //
	/////////////////////

	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		float queuePriority = 1.0f;

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

		std::set<uint32_t> uniqueQueueFamilies = {
			indices.graphicsFamily.value(), indices.presentFamily.value(),
			indices.transferFamily.value()};

		for (uint32_t queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo{
				.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
				.queueFamilyIndex = queueFamily,
				.queueCount = 1,
				.pQueuePriorities = &queuePriority,
			};
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures features;
		vkGetPhysicalDeviceFeatures(physicalDevice, &features);

		VkDeviceCreateInfo deviceCreateInfo{
			.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.queueCreateInfoCount =
				static_cast<uint32_t>(queueCreateInfos.size()),
			.pQueueCreateInfos = queueCreateInfos.data(),
			.enabledLayerCount = 0,
			.enabledExtensionCount =
				static_cast<uint32_t>(deviceExtensions.size()),
			.ppEnabledExtensionNames = deviceExtensions.data(),
			.pEnabledFeatures = &features,
		};

		// COMPAT: No device-specific validation layers in modern Vulkan
		if (enableValidationLayers) {
			deviceCreateInfo.enabledLayerCount =
				static_cast<uint32_t>(validationLayers.size());
			deviceCreateInfo.ppEnabledLayerNames = validationLayers.data();
		}

		if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr,
						   &device) != VK_SUCCESS) {
			throw std::runtime_error(std::string(ERROR_CREATE_DEVICE));
		}
		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0,
						 &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0,
						 &presentQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0,
						 &transferQueue);
	}

	////////////////
	// Swap Chain //
	////////////////

	struct SwapChainSupportDetails {
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice dev) {
		SwapChainSupportDetails details;
		int res = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
			dev, surface, &details.capabilities);

		uint32_t formatCount;
		res = vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &formatCount,
												   nullptr);
		switch (res) {
		case VK_SUCCESS:
			break;
		case VK_ERROR_OUT_OF_HOST_MEMORY:
			throw std::runtime_error(
				"Error fetching available surface format: Out of host memory");
		case VK_ERROR_OUT_OF_DEVICE_MEMORY:
			throw std::runtime_error("Error fetching available surface format: "
									 "Out of device memory");
		case VK_ERROR_SURFACE_LOST_KHR:
			throw std::runtime_error("Error fetching available surface format: "
									 "Surface has been lost");
		}
		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &formatCount,
												 details.formats.data());
		}

		uint32_t presentCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &presentCount,
												  nullptr);
		if (presentCount != 0) {
			details.presentModes.resize(presentCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(
				dev, surface, &presentCount, details.presentModes.data());
		}

		return details;
	}

	VkSurfaceFormatKHR
	chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats) {
		for (const auto &format : formats) {
			if (format.format == SWAP_SURFACE_TARGET_FORMAT &&
				format.colorSpace == SWAP_SURFACE_TARGET_COLORSPACE) {
				return format;
			}
		}
		return formats[0];
	}

	VkPresentModeKHR
	choosePresentMode(const std::vector<VkPresentModeKHR> presentModes) {
		std::set<VkPresentModeKHR> availablePresents(presentModes.begin(),
													 presentModes.end());
		for (const auto &presentMode : presentModeOrder) {
			auto found = availablePresents.find(presentMode);
			if (found != availablePresents.end()) {
				return *found;
			}
		}
		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) {
		if (capabilities.currentExtent.width !=
			std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		} else {
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actualExtent = {static_cast<uint32_t>(width),
									   static_cast<uint32_t>(height)};

			actualExtent.width = std::clamp(actualExtent.width,
											capabilities.minImageExtent.width,
											capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(
				actualExtent.height, capabilities.minImageExtent.height,
				capabilities.maxImageExtent.height);
			return actualExtent;
		}
	}

	void createSwapChain() {
		SwapChainSupportDetails swapChainSupport =
			querySwapChainSupport(physicalDevice);
		VkSurfaceFormatKHR surfaceFormat =
			chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode =
			choosePresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 &&
			imageCount > swapChainSupport.capabilities.maxImageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{
			.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
			.surface = surface,
			.minImageCount = imageCount,
			.imageFormat = surfaceFormat.format,
			.imageColorSpace = surfaceFormat.colorSpace,
			.imageExtent = extent,
			.imageArrayLayers = 1,
			.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.preTransform = swapChainSupport.capabilities.currentTransform,
			.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			.presentMode = presentMode,
			.clipped = VK_TRUE,
			.oldSwapchain = VK_NULL_HANDLE,
		};

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
										 indices.presentFamily.value()};

		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}

		int res =
			vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain);
		switch (res) {
		case VK_SUCCESS:
			break;
		case VK_ERROR_DEVICE_LOST:
			throw std::runtime_error(std::string(ERROR_DEVICE_LOST) +
									 std::string(ERROR_CREATE_SWAPCHAIN));

		case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
			throw std::runtime_error(std::string(ERROR_WINDOW_IN_USE) +
									 std::string(ERROR_CREATE_SWAPCHAIN));
		case VK_ERROR_SURFACE_LOST_KHR:
			throw std::runtime_error("Surface lost: " +
									 std::string(ERROR_CREATE_SWAPCHAIN));
		default:
			throw std::runtime_error("Unknown error occured: " +
									 std::string(ERROR_CREATE_SWAPCHAIN));
		}

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount,
								swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	/////////////////
	// Image Views //
	/////////////////

	VkImageView createImageView(VkImage image, VkFormat format,
								VkImageSubresourceRange subresource) {
		VkImageViewCreateInfo createInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = image,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = format,
			.components =
				{
					.r = VK_COMPONENT_SWIZZLE_IDENTITY,
					.g = VK_COMPONENT_SWIZZLE_IDENTITY,
					.b = VK_COMPONENT_SWIZZLE_IDENTITY,
					.a = VK_COMPONENT_SWIZZLE_IDENTITY,
				},
			.subresourceRange = subresource,
		};

		VkImageView imageView;
		int res = vkCreateImageView(device, &createInfo, nullptr, &imageView);
		switch (res) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(ERROR_CREATE_IMAGEVIEW);
		}
		return imageView;
	}

	void createSwapChainImageViews() {
		swapChainImageViews.resize(swapChainImages.size());
		VkImageSubresourceRange swapChainSubresource{
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		};
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			swapChainImageViews[i] = createImageView(
				swapChainImages[i], swapChainImageFormat, swapChainSubresource);
		}
	}

	///////////////////////
	// Graphics Pipeline //
	///////////////////////

	void createRenderPass() {
		// 1. Color Attachment
		VkAttachmentDescription colorAttachment{
			.format = swapChainImageFormat,
			.samples = msaaSamples,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};
		VkAttachmentReference colorAttachmentRef{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		// 2. Depth Attachment
		VkAttachmentDescription depthAttachment{
			.format = findDepthFormat(),
			.samples = msaaSamples,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};
		VkAttachmentReference depthAttachmentRef{
			.attachment = 1,
			.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};

		// 3. Color Resolve Attachment
		VkAttachmentDescription colorAttachmentResolve{
			.format = swapChainImageFormat,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		};
		VkAttachmentReference colorAttachmentResolveRef{
			.attachment = 2,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};
		VkSubpassDescription subpass{
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments = &colorAttachmentRef,
			.pResolveAttachments = &colorAttachmentResolveRef,
			.pDepthStencilAttachment = &depthAttachmentRef,
		};

		VkSubpassDependency dependency{
			.srcSubpass = VK_SUBPASS_EXTERNAL,
			.dstSubpass = 0,
			.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
							VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
							VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
							 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		};

		std::vector<VkAttachmentDescription> attachments = {
			colorAttachment, depthAttachment, colorAttachmentResolve};

		VkRenderPassCreateInfo renderPassInfo{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = (uint32_t)attachments.size(),
			.pAttachments = attachments.data(),
			.subpassCount = 1,
			.pSubpasses = &subpass,
			.dependencyCount = 1,
			.pDependencies = &dependency,
		};

		int res =
			vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass);
		switch (res) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(std::string(ERROR_CREATE_RENDERPASS));
		}
	}

	void createGraphicsPipeline() {
		VkShaderModule vertShader =
			createShaderModule(readFile("shaders/tutorial_vert.spv"));
		VkShaderModule fragShader =
			createShaderModule(readFile("shaders/tutorial_frag.spv"));

		VkPipelineShaderStageCreateInfo shaderStages[] = {
			VkPipelineShaderStageCreateInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_VERTEX_BIT,
				.module = vertShader,
				.pName = "main",

			},
			VkPipelineShaderStageCreateInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
				.module = fragShader,
				.pName = "main",

			}};

		// Setup vertex input, and assembly state
		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();
		VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &bindingDescription,
			.vertexAttributeDescriptionCount =
				static_cast<uint32_t>(attributeDescriptions.size()),
			.pVertexAttributeDescriptions = attributeDescriptions.data(),
		};

		// Setup viewport and scissor
		viewport = {
			.x = 0.0f,
			.y = (float)swapChainExtent.height,
			.width = (float)swapChainExtent.width,
			.height = -(float)swapChainExtent.height,
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};
		scissor = {
			.offset = {0, 0},
			.extent = swapChainExtent,
		};

		VkPipelineViewportStateCreateInfo viewportState{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.scissorCount = 1,
		};

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptorSetLayout,
		};
		int res = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
										 &pipelineLayout);

		switch (res) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(std::string(ERROR_CREATE_PIPELINE_LAYOUT));
		}

		auto multisampling = DEFAULT_MULTISAMPLING;
		multisampling.rasterizationSamples = msaaSamples;

		// Create graphics pipeline
		VkGraphicsPipelineCreateInfo pipelineInfo{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = shaderStages,
			.pVertexInputState = &vertexInputCreateInfo,
			.pInputAssemblyState = &DEFAULT_INPUT_ASSEMBLY,
			.pViewportState = &viewportState,
			.pRasterizationState = &RASTERIZER_DEPTH_OFF,
			.pMultisampleState = &multisampling,
			.pDepthStencilState = hasStencilComponent(findDepthFormat())
									  ? &DEFAULT_DEPTH_STENCIL
									  : &DEFAULT_DEPTH_NO_STENCIL,
			.pColorBlendState = &DEFAULT_COLOR_BLEND_INFO,
			.pDynamicState = &DEFAULT_DYNAMIC_STATE,
			.layout = pipelineLayout,
			.renderPass = renderPass,
			.subpass = 0,
			.basePipelineHandle = VK_NULL_HANDLE, // Optional
			.basePipelineIndex = -1,			  // Optional
		};

		res =
			vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
									  nullptr, &graphicsPipeline);
		switch (res) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(std::string(ERROR_CREATE_PIPELINE));
		}

		vkDestroyShaderModule(device, vertShader, nullptr);
		vkDestroyShaderModule(device, fragShader, nullptr);
	}

	VkShaderModule createShaderModule(const std::vector<char> &code) {
		VkShaderModuleCreateInfo createInfo{
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = code.size(),
			.pCode = reinterpret_cast<const uint32_t *>(code.data()),
		};

		VkShaderModule shaderModule;
		int res =
			vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
		switch (res) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(std::string(ERROR_CREATE_SHADERMODULE));
		}

		return shaderModule;
	}

	/////////////////
	// Framebuffer //
	/////////////////

	void createFramebuffers() {
		swapChainFramebuffers.resize(swapChainImageViews.size());

		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			std::vector<VkImageView> attachments = {
				colorImageView, depthImageView, swapChainImageViews[i]};
			VkFramebufferCreateInfo framebufferInfo{
				.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				.renderPass = renderPass,
				.attachmentCount = (uint32_t)attachments.size(),
				.pAttachments = attachments.data(),
				.width = swapChainExtent.width,
				.height = swapChainExtent.height,
				.layers = 1,
			};

			int res = vkCreateFramebuffer(device, &framebufferInfo, nullptr,
										  &swapChainFramebuffers[i]);
			switch (res) {
			case VK_SUCCESS:
				break;
			default:
				throw std::runtime_error(std::string(ERROR_CREATE_FRAMEBUFFER));
			}
		}
	}

	/////////////////////
	// Command Buffers //
	/////////////////////

	void createCommandPool() {
		QueueFamilyIndices queueFamilyIndices =
			findQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo graphicsPoolInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(),
		};
		VkCommandPoolCreateInfo transferPoolInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = queueFamilyIndices.transferFamily.value(),
		};

		int res = vkCreateCommandPool(device, &graphicsPoolInfo, nullptr,
									  &graphicsPool);
		switch (res) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(std::string(ERROR_CREATE_COMMAND_POOL));
		}
		res = vkCreateCommandPool(device, &transferPoolInfo, nullptr,
								  &transferPool);
		switch (res) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(std::string(ERROR_CREATE_COMMAND_POOL));
		}
	}

	VkCommandBuffer beginSingleTimeCommand(VkCommandPool commandPool) {
		VkCommandBufferAllocateInfo allocInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = commandPool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		vkBeginCommandBuffer(commandBuffer, &beginInfo);
		return commandBuffer;
	}

	void endSingleTimeCommand(std::vector<VkCommandBuffer> commandBuffers,
							  VkCommandPool commandPool, VkQueue queue) {
		VkFence singleTimeFence;
		for (auto &commandBuffer : commandBuffers) {
			vkEndCommandBuffer(commandBuffer);
		}
		VkFenceCreateInfo fenceInfo{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT,
		};
		vkCreateFence(device, &fenceInfo, nullptr, &singleTimeFence);
		vkResetFences(device, 1, &singleTimeFence);

		VkSubmitInfo submitInfo{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = (uint32_t)commandBuffers.size(),
			.pCommandBuffers = commandBuffers.data(),
		};
		vkQueueSubmit(queue, 1, &submitInfo, singleTimeFence);
		vkWaitForFences(device, 1, &singleTimeFence, VK_TRUE, UINT64_MAX);
		vkDestroyFence(device, singleTimeFence, nullptr);
		vkFreeCommandBuffers(device, commandPool, commandBuffers.size(),
							 commandBuffers.data());
	}

	void runSingleTimeCommand(VkCommandPool commandPool, VkQueue queue,
							  std::function<void(VkCommandBuffer &)> lambda) {

		VkCommandBuffer commandBuffer = beginSingleTimeCommand(commandPool);
		lambda(commandBuffer);
		endSingleTimeCommand({commandBuffer}, commandPool, queue);
	}

	void createCommandBuffers() {
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		VkCommandBufferAllocateInfo allocInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = graphicsPool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = (uint32_t)commandBuffers.size(),
		};

		int res =
			vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data());
		switch (res) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(std::string(ERROR_CREATE_COMMAND_BUFFER));
		}
	}

	void recordCommandBuffer(VkCommandBuffer commandBuffer,
							 uint32_t imageIndex) {
		VkCommandBufferBeginInfo beginInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = 0,
			.pInheritanceInfo = 0,
		};
		int res = vkBeginCommandBuffer(commandBuffer, &beginInfo);
		switch (res) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(std::string(ERROR_BEGIN_COMMAND_BUFFER));
		}

		std::vector<VkClearValue> clearValues{
			{.color = {{0.0f, 0.0f, 0.0f, 1.0f}}}, {.depthStencil = {1.0f, 0}}};
		VkRenderPassBeginInfo renderPassInfo{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = renderPass,
			.framebuffer = swapChainFramebuffers[imageIndex],
			.renderArea =
				{
					.offset = {0, 0},
					.extent = swapChainExtent,
				},
			.clearValueCount = (uint32_t)clearValues.size(),
			.pClearValues = clearValues.data(),
		};

		// Run commands
		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
							 VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
						  graphicsPipeline);

		VkDeviceSize vertexOffsets[] = {verticesOffset};
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &ivBuffer, vertexOffsets);
		vkCmdBindIndexBuffer(commandBuffer, ivBuffer, indicesOffset,
							 INDEX_TYPE);
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
								pipelineLayout, 0, 1,
								&descriptorSets[currentFrame], 0, nullptr);
		vkCmdDrawIndexed(commandBuffer, (float)indices.size(), 1, 0, 0, 0);
		vkCmdEndRenderPass(commandBuffer);
		res = vkEndCommandBuffer(commandBuffer);
		switch (res) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(std::string(ERROR_END_COMMAND_BUFFER));
		}
	}

	/////////////////////
	// Synchronisation //
	/////////////////////

	void createSyncObjects() {
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo{
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		};
		VkFenceCreateInfo fenceInfo{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT,
		};

		// Create rendering and presentation sync objects
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr,
								  &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(device, &semaphoreInfo, nullptr,
								  &renderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fenceInfo, nullptr,
							  &inFlightFences[i]) != VK_SUCCESS) {
				throw std::runtime_error(std::string(ERROR_CREATE_SYNC));
			}
		}
	}

	void cleanupSyncObjects() {
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}
	}

	/////////////
	// Buffers //
	/////////////

	void createIVBuffer() {
		VkDeviceSize verticesSize = sizeof(Vertex) * vertices.size();
		verticesOffset = 0;
		VkDeviceSize indicesSize = sizeof(index_t) * indices.size();
		indicesOffset = verticesOffset + verticesSize;
		VkDeviceSize bufferSize = verticesSize + indicesSize;

		// Create staging buffer
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
					 stagingBuffer);
		allocateBuffers(
			AllocateBuffersInfo{
				.buffersCount = 1,
				.pBuffers = &stagingBuffer,
				.bufferMemory = &stagingBufferMemory,
				.memoryPropertyflags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
									   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
			},
			nullptr);

		// Copy vertex and index contents
		char *data;
		vkMapMemory(device, stagingBufferMemory, 0, VK_WHOLE_SIZE, 0,
					(void **)&data);
		memcpy(data, vertices.data(), verticesSize);
		memcpy(data + indicesOffset, indices.data(), indicesSize);
		vkUnmapMemory(device, stagingBufferMemory);

		// Create buffer and copy staging buffer data
		createBuffer(bufferSize,
					 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
						 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
						 VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
					 ivBuffer);
		allocateBuffers(
			AllocateBuffersInfo{
				.buffersCount = 1,
				.pBuffers = &ivBuffer,
				.bufferMemory = &ivBufferMemory,
				.memoryPropertyflags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			},
			nullptr);

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, ivBuffer, &memRequirements);

		copyBuffer(stagingBuffer, ivBuffer, bufferSize);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
					  VkBuffer &buffer) {
		VkBufferCreateInfo bufferInfo{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilies[] = {indices.graphicsFamily.value(),
									indices.presentFamily.value()};

		if (indices.graphicsFamily != indices.presentFamily) {
			bufferInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
			bufferInfo.queueFamilyIndexCount = 2;
			bufferInfo.pQueueFamilyIndices = queueFamilies;
		}

		int res = vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);
		switch (res) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(std::string(ERROR_CREATE_BUFFER));
		}
	}

	struct AllocateBuffersInfo {
		uint16_t buffersCount;
		VkBuffer *pBuffers;
		VkDeviceMemory *bufferMemory;
		VkMemoryPropertyFlags memoryPropertyflags;
	};

	struct AllocationInfo {
		std::vector<VkDeviceSize> offsets;
		VkMemoryRequirements requirements;
	};

	void allocateBuffers(AllocateBuffersInfo info, AllocationInfo *ret) {
		std::vector<VkMemoryRequirements> requirements(info.buffersCount);
		for (size_t i = 0; i < info.buffersCount; i++) {
			vkGetBufferMemoryRequirements(device, info.pBuffers[i],
										  &requirements[i]);
		}

		AllocationInfo res = getMemoryRequirements(requirements);

		VkMemoryAllocateInfo allocInfo{
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = res.requirements.size,
			.memoryTypeIndex = findMemoryType(res.requirements.memoryTypeBits,
											  info.memoryPropertyflags),
		};
		int result =
			vkAllocateMemory(device, &allocInfo, nullptr, info.bufferMemory);
		switch (result) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(std::string(ERROR_ALLOCATE_MEMORY_BUFFER));
		}
		for (size_t i = 0; i < info.buffersCount; i++) {
			vkBindBufferMemory(device, info.pBuffers[i], *info.bufferMemory,
							   res.offsets[i]);
		}

		if (ret != nullptr) {
			*ret = res;
		}
	}

	AllocationInfo
	getMemoryRequirements(std::vector<VkMemoryRequirements> requirements) {
		VkMemoryRequirements fullReqs{
			.size = 0,
			.alignment = 1,
			.memoryTypeBits = 0,
		};
		std::for_each(requirements.begin(), requirements.end(), [&](auto req) {
			fullReqs.memoryTypeBits |= req.memoryTypeBits;
			fullReqs.alignment = std::max(req.alignment, fullReqs.alignment);
		});

		// Align buffers to memory
		std::vector<VkDeviceSize> offsets(requirements.size());
		VkDeviceSize currentOffset = 0;
		for (size_t i = 0; i < requirements.size(); i++) {
			offsets[i] = currentOffset;
			fullReqs.size += requirements[i].size;
			if (requirements[i].size % fullReqs.alignment) {
				fullReqs.size += fullReqs.alignment -
								 (requirements[i].size % fullReqs.alignment);
			}
			currentOffset = fullReqs.size;
		}
		return AllocationInfo{
			.offsets = offsets,
			.requirements = fullReqs,
		};
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
		VkBufferCopy copyRegion{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = size,
		};
		runSingleTimeCommand(transferPool, transferQueue,
							 [&](VkCommandBuffer &commandBuffer) {
								 vkCmdCopyBuffer(commandBuffer, srcBuffer,
												 dstBuffer, 1, &copyRegion);
							 });
	}

	uint32_t findMemoryType(uint32_t typeFilter,
							VkMemoryPropertyFlags properties) {

		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if (typeFilter & (1 << i) &&
				(memProperties.memoryTypes[i].propertyFlags & properties) ==
					properties) {
				return i;
			}
		}
		throw std::runtime_error(ERROR_FIND_MEMORY_TYPE_SUITABLE);
	}

	//////////////
	// Uniforms //
	//////////////

	void createDescriptorSetLayout() {
		std::vector<VkDescriptorSetLayoutBinding> bindings = {
			{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
				.pImmutableSamplers = nullptr,

			},
			{
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				.pImmutableSamplers = nullptr,

			}};

		VkDescriptorSetLayoutCreateInfo layoutInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = (uint32_t)bindings.size(),
			.pBindings = bindings.data(),
		};

		int result = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
												 &descriptorSetLayout);
		switch (result) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(ERROR_CREATE_DESCRIPTOR_SET_LAYOUT);
		}
	}

	void createUniformBuffer() {
		VkDeviceSize bufferc = sizeof(UniformBufferObject);
		uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			createBuffer(bufferc, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
						 uniformBuffers[i]);
		}

		AllocationInfo memoryDetails;
		allocateBuffers(
			AllocateBuffersInfo{
				.buffersCount = MAX_FRAMES_IN_FLIGHT,
				.pBuffers = uniformBuffers.data(),
				.bufferMemory = &uniformBufferMemory,
				.memoryPropertyflags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
									   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			},
			&memoryDetails);

		uniformBufferOffsets = memoryDetails.offsets;

		vkMapMemory(device, uniformBufferMemory, 0,
					memoryDetails.requirements.size, 0,
					(void **)&uniformBuffersMap);
	}

	void createDescriptorPool() {
		std::vector<VkDescriptorPoolSize> poolSizes{
			{
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
			},
			{
				.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
			}};

		VkDescriptorPoolCreateInfo poolInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = 0,
			.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
			.poolSizeCount = (uint32_t)poolSizes.size(),
			.pPoolSizes = poolSizes.data(),
		};

		int result =
			vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
		switch (result) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(ERROR_CREATE_DESCRIPTOR_POOL);
		}
	}

	void createDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
												   descriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptorPool,
			.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
			.pSetLayouts = layouts.data(),
		};
		descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		int result =
			vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data());
		switch (result) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(ERROR_ALLOCATE_DESCRIPTOR_SETS);
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			VkDescriptorBufferInfo buffInfo{
				.buffer = uniformBuffers[i],
				.offset = 0,
				.range = sizeof(UniformBufferObject),
			};
			VkDescriptorImageInfo imageInfo{
				.sampler = textureSampler,
				.imageView = textureImageView,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};

			std::vector<VkWriteDescriptorSet> descriptorWrites = {
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = descriptorSets[i],
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &buffInfo,
				},
				{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = descriptorSets[i],
					.dstBinding = 1,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.pImageInfo = &imageInfo,
				}};
			vkUpdateDescriptorSets(device, (uint32_t)descriptorWrites.size(),
								   descriptorWrites.data(), 0, nullptr);
		}
	}

	void updateUniformBuffer(uint32_t frame) {
		static auto startTime = std::chrono::high_resolution_clock::now();

		// Get current time for neat rotation effect
		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(
						 currentTime - startTime)
						 .count();

		// Fill ubo
		UniformBufferObject ubo{
			.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(30.0f),
								 glm::vec3(0.0f, 1.0f, 0.0f)),
			.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f),
								glm::vec3(0.0f, 0.0f, 0.0f),
								glm::vec3(0.0f, 1.0f, 0.0f)),
			.proj = glm::perspective(glm::radians(45.0f),
									 swapChainExtent.width /
										 (float)swapChainExtent.height,
									 0.1f, 10.0f),
		};
		memcpy(&uniformBuffersMap[uniformBufferOffsets[frame]], &ubo,
			   sizeof(ubo));
	}

	//////////////
	// Textures //
	//////////////

	struct AllocateImagesInfo {
		uint16_t imageCount;
		VkImage *pImages;
		VkDeviceMemory *bufferMemory;
		VkMemoryPropertyFlags memoryPropertyFlags;
	};

	void allocateImages(AllocateImagesInfo info, AllocationInfo *ret) {
		std::vector<VkMemoryRequirements> requirements(info.imageCount);
		for (size_t i = 0; i < info.imageCount; i++) {
			vkGetImageMemoryRequirements(device, info.pImages[i],
										 &requirements[i]);
		}

		AllocationInfo res = getMemoryRequirements(requirements);

		VkMemoryAllocateInfo allocInfo{
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = res.requirements.size,
			.memoryTypeIndex = findMemoryType(res.requirements.memoryTypeBits,
											  info.memoryPropertyFlags),
		};
		int result =
			vkAllocateMemory(device, &allocInfo, nullptr, info.bufferMemory);
		switch (result) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(std::string(ERROR_ALLOCATE_MEMORY_BUFFER));
		}
		for (size_t i = 0; i < info.imageCount; i++) {
			vkBindImageMemory(device, info.pImages[i], *info.bufferMemory,
							  res.offsets[i]);
		}

		if (ret != nullptr) {
			*ret = res;
		}
	}

	void transitionImageLayout(VkImage image, VkFormat format,
							   VkImageSubresourceRange subresource,
							   VkImageLayout oldLayout,
							   VkImageLayout newLayout) {

		VkImageMemoryBarrier barrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.oldLayout = oldLayout,
			.newLayout = newLayout,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = image,
			.subresourceRange = subresource,
		};

		// Set appropriate barrier stages from image source and destination
		// layouts. This is based on how the layout relates to their usage.
		VkPipelineStageFlags srcStage, dstStage;
		switch (oldLayout) {
		case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			break;
		case VK_IMAGE_LAYOUT_UNDEFINED:
			barrier.srcAccessMask = 0;
			srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			break;
		default:
			throw std::invalid_argument(ERROR_BARRIER_UNSUPPORTED_SRC_LAYOUT);
		}
		switch (newLayout) {
		case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			break;
		case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
			barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
			dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
			break;
		case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			break;
		default:
			throw std::invalid_argument(ERROR_BARRIER_UNSUPPORTED_DST_LAYOUT);
		}

		runSingleTimeCommand(
			graphicsPool, graphicsQueue, [&](VkCommandBuffer &commandBuffer) {
				vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0,
									 nullptr, 0, nullptr, 1, &barrier);
			});
	}

	void copyBufferToImage(VkBuffer buffer, VkImage image,
						   VkImageSubresourceRange subresource,
						   VkExtent3D extent) {
		VkBufferImageCopy region{
			.bufferOffset = 0,
			.bufferRowLength = 0,
			.bufferImageHeight = 0,
			.imageSubresource =
				{
					.aspectMask = subresource.aspectMask,
					.mipLevel = subresource.baseMipLevel,
					.baseArrayLayer = subresource.baseArrayLayer,
					.layerCount = subresource.layerCount,
				},
			.imageOffset = {0, 0, 0},
			.imageExtent = extent,
		};
		runSingleTimeCommand(
			transferPool, transferQueue, [&](VkCommandBuffer &commandBuffer) {
				vkCmdCopyBufferToImage(commandBuffer, buffer, image,
									   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
									   &region);
			});
	}

	void createImage(VkExtent3D extent, uint32_t mipLevels, VkFormat format,
					 VkImageTiling tiling, VkSampleCountFlagBits numSamples,
					 VkImageUsageFlags flags, VkImage &image) {
		VkImageCreateInfo imageInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.flags = 0,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = format,
			.extent = extent,
			.mipLevels = mipLevels,
			.arrayLayers = 1,
			.samples = numSamples,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = flags,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilies[] = {indices.graphicsFamily.value(),
									indices.transferFamily.value()};
		if (indices.graphicsFamily.value() != indices.transferFamily.value()) {
			imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			imageInfo.queueFamilyIndexCount = 2;
			imageInfo.pQueueFamilyIndices = queueFamilies;
		}

		int result = vkCreateImage(device, &imageInfo, nullptr, &image);
		switch (result) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(ERROR_CREATE_IMAGE);
		}
	}

	void createTextureImageView() {
		textureImageView =
			createImageView(images[0], VK_FORMAT_R8G8B8A8_SRGB,
							{
								.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
								.baseMipLevel = 0,
								.levelCount = mipLevels,
								.baseArrayLayer = 0,
								.layerCount = 1,
							});
	}

	void createTextureSampler() {
		VkPhysicalDeviceProperties properties{};
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);

		VkPhysicalDeviceFeatures features{};
		vkGetPhysicalDeviceFeatures(physicalDevice, &features);

		VkSamplerCreateInfo samplerInfo = DEFAULT_SAMPLER;
		samplerInfo.maxLod = (float)mipLevels;
		if (features.samplerAnisotropy) {
			samplerInfo.anisotropyEnable = VK_TRUE;
			samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
		}
		int result =
			vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler);
		switch (result) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(ERROR_CREATE_SAMPLER);
		}
	}

	void createTextureImage() {
		// Load image data
		int texWidth, texHeight, texChannels;
		stbi_uc *pixels = stbi_load(MODEL_TEX_PATH.c_str(), &texWidth,
									&texHeight, &texChannels, STBI_rgb_alpha);
		if (pixels == NULL) {
			throw std::runtime_error(std::string(ERROR_STBI_LOAD_TEXTURE) +
									 std::string(stbi_failure_reason()));
		}
		mipLevels = getMipLevels(texWidth, texHeight);
		VkDeviceSize imageSize = texWidth * texHeight * STBI_rgb_alpha;

		// Load staging buffer and load stb data to it
		void *data;
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
					 stagingBuffer);
		allocateBuffers(
			AllocateBuffersInfo{
				.buffersCount = 1,
				.pBuffers = &stagingBuffer,
				.bufferMemory = &stagingBufferMemory,
				.memoryPropertyflags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
									   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			},
			nullptr);
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, imageSize);
		vkUnmapMemory(device, stagingBufferMemory);
		stbi_image_free(pixels);

		// Create Vulkan Image
		images.resize(1);
		VkExtent3D imageExtent = {
			.width = (uint32_t)texWidth,
			.height = (uint32_t)texHeight,
			.depth = 1,
		};
		VkImageSubresourceRange textureSubresource{
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = mipLevels,
			.baseArrayLayer = 0,
			.layerCount = 1,
		};
		createImage(imageExtent, mipLevels, VK_FORMAT_R8G8B8A8_SRGB,
					VK_IMAGE_TILING_OPTIMAL, VK_SAMPLE_COUNT_1_BIT,
					VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
						VK_IMAGE_USAGE_TRANSFER_DST_BIT |
						VK_IMAGE_USAGE_SAMPLED_BIT,
					images[0]);

		allocateImages(
			AllocateImagesInfo{
				.imageCount = 1,
				.pImages = images.data(),
				.bufferMemory = &imageMemory,
				.memoryPropertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			},
			nullptr);
		transitionImageLayout(images[0], VK_FORMAT_R8G8B8A8_SRGB,
							  textureSubresource, VK_IMAGE_LAYOUT_UNDEFINED,
							  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copyBufferToImage(stagingBuffer, images[0], textureSubresource,
						  imageExtent);
		generateMipmaps(images[0], VK_FORMAT_R8G8B8A8_SRGB, imageExtent,
						mipLevels);

		// Cleanup
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void generateMipmaps(VkImage image, VkFormat format, VkExtent3D extent,
						 uint32_t mipLevels) {
		VkFormatProperties props;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
		if (!(props.optimalTilingFeatures &
			  VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
			throw std::runtime_error(ERROR_FILTER_FORMAT_FEATURES);
		}

		int32_t mipWidth = extent.width, mipHeight = extent.height;
		VkImageMemoryBarrier barrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			}};
		VkImageBlit blit{
			.srcSubresource =
				{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = 0,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			.srcOffsets = {{0, 0, 0}, {0, 0, 0}},
			.dstSubresource =
				{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			.dstOffsets = {{0, 0, 0}, {0, 0, 0}},
		};
		runSingleTimeCommand(
			graphicsPool, graphicsQueue, [&](VkCommandBuffer &commandBuffer) {
				for (uint32_t i = 1; i < mipLevels; i++) {
					barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
					barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
					barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
					barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
					barrier.subresourceRange.baseMipLevel = i - 1;
					vkCmdPipelineBarrier(commandBuffer,
										 VK_PIPELINE_STAGE_TRANSFER_BIT,
										 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
										 nullptr, 0, nullptr, 1, &barrier);

					blit.srcSubresource.mipLevel = i - 1;
					blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
					blit.dstSubresource.mipLevel = i;
					mipWidth = std::max(mipWidth >> 1, 1);
					mipHeight = std::max(mipHeight >> 1, 1);
					blit.dstOffsets[1] = {mipWidth, mipHeight, 1};
					vkCmdBlitImage(commandBuffer, image,
								   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
								   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
								   &blit, VK_FILTER_LINEAR);

					barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
					barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
					barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
					barrier.newLayout =
						VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					vkCmdPipelineBarrier(
						commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
						VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
						nullptr, 1, &barrier);
				}
				barrier.subresourceRange.baseMipLevel = mipLevels - 1;
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
				barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
				barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				vkCmdPipelineBarrier(commandBuffer,
									 VK_PIPELINE_STAGE_TRANSFER_BIT,
									 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
									 0, nullptr, 0, nullptr, 1, &barrier);
			});
	}

	/////////////////////
	// Depth & Stencil //
	/////////////////////

	VkFormat findSupportedFormat(const std::vector<VkFormat> &candidates,
								 VkImageTiling tiling,
								 VkFormatFeatureFlags features) {

		for (auto &format : candidates) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

			VkFormatFeatureFlags formatFeatures;
			formatFeatures = tiling == VK_IMAGE_TILING_OPTIMAL
								 ? props.optimalTilingFeatures
								 : props.linearTilingFeatures;
			if (features & formatFeatures) {
				return format;
			}
		}
		throw std::runtime_error(ERROR_FIND_FORMAT);
	}

	inline VkFormat findDepthFormat() {
		return findSupportedFormat(
			{VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT,
			 VK_FORMAT_D32_SFLOAT},
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
	}

	inline bool hasStencilComponent(VkFormat format) {
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
			   format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	void createDepthResources() {
		VkFormat format = findDepthFormat();

		VkExtent3D depthExtent{
			.width = swapChainExtent.width,
			.height = swapChainExtent.height,
			.depth = 1,
		};
		createImage(depthExtent, 1, format, VK_IMAGE_TILING_OPTIMAL,
					msaaSamples, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
					depthImage);
		allocateImages(
			AllocateImagesInfo{
				.imageCount = 1,
				.pImages = &depthImage,
				.bufferMemory = &depthImageMemory,
			},
			nullptr);

		VkImageSubresourceRange depthSubresource{
			.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		};
		if (hasStencilComponent(format)) {
			depthSubresource.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
		}
		depthImageView = createImageView(depthImage, format, depthSubresource);
		transitionImageLayout(depthImage, format, depthSubresource,
							  VK_IMAGE_LAYOUT_UNDEFINED,
							  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	}

	////////////
	// Models //
	////////////

	void loadModel() {
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string err;

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err,
							  MODEL_PATH.c_str())) {
			throw std::runtime_error(err);
		}

		std::unordered_map<Vertex, uint32_t> uniqueVertices{};
		for (const auto &shape : shapes) {
			for (const auto &index : shape.mesh.indices) {
				Vertex vert = getVertexFromIndex(attrib, index);
				if (uniqueVertices.count(vert) == 0) {
					uniqueVertices[vert] = (uint32_t)vertices.size();
					vertices.push_back(vert);
				}
				indices.push_back(uniqueVertices[vert]);
			}
		}
	}

	Vertex getVertexFromIndex(tinyobj::attrib_t attrib,
							  tinyobj::index_t index) {
		return {
			.pos = {attrib.vertices[3 * index.vertex_index],
					attrib.vertices[3 * index.vertex_index + 2],
					attrib.vertices[3 * index.vertex_index + 1], 1.0f},
			.color = {1.0f, 1.0f, 1.0f, 1.0f},
			.texCoord = {attrib.texcoords[2 * index.texcoord_index],
						 1.0f - attrib.texcoords[2 * index.texcoord_index + 1]},

		};
	}

	///////////////////
	// Multisampling //
	///////////////////

	void createColorResources() {
		VkFormat colorFormat = swapChainImageFormat;

		createImage({swapChainExtent.width, swapChainExtent.height, 1}, 1,
					colorFormat, VK_IMAGE_TILING_OPTIMAL, msaaSamples,
					VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
						VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
					colorImage);
		allocateImages(
			{
				.imageCount = 1,
				.pImages = &colorImage,
				.bufferMemory = &colorImageMemory,
				.memoryPropertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			},
			nullptr);

		VkImageSubresourceRange subresource{
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		};
		colorImageView = createImageView(colorImage, colorFormat, subresource);
	}

	//////////
	// Loop //
	//////////

	void mainLoop() {
		// static auto lastFrameTime =
		// std::chrono::high_resolution_clock::now();
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			// auto currentTime = std::chrono::high_resolution_clock::now();
			// float time =
			// 	std::chrono::duration<float, std::chrono::milliseconds::period>(
			// 		currentTime - lastFrameTime)
			// 		.count();
			// if (time > FRAMERATE_CAP) {
			drawFrame();
			// 	lastFrameTime = currentTime;
			// }
		}
		vkDeviceWaitIdle(device);
	}

	void drawFrame() {
		// Wait for last frame to finish
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE,
						UINT64_MAX);

		// Get image index from swap chain and device
		uint32_t imageIndex;
		int res = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
										imageAvailableSemaphores[currentFrame],
										VK_NULL_HANDLE, &imageIndex);
		if (res == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapChain();
			return;
		} else if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error(std::string(ERROR_ACQUIRE_NEXT_IMAGE_KHR));
		}
		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		// Resize viewport and scissor to current frame size
		viewport.width = (float)swapChainExtent.width;
		viewport.height = -(float)swapChainExtent.height;
		viewport.y = (float)swapChainExtent.height;
		scissor.extent = swapChainExtent;
		updateUniformBuffer(currentFrame);

		// Reset and record command buffer on image index given.
		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

		// Submit command queue, with semaphores and fences.
		VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
		VkSemaphore signalSemaphores[] = {
			renderFinishedSemaphores[currentFrame]};
		VkPipelineStageFlags waitStages[] = {
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
		VkSubmitInfo submitInfo{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = waitSemaphores,
			.pWaitDstStageMask = waitStages,
			.commandBufferCount = 1,
			.pCommandBuffers = &commandBuffers[currentFrame],
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = signalSemaphores,
		};
		res = vkQueueSubmit(graphicsQueue, 1, &submitInfo,
							inFlightFences[currentFrame]);
		switch (res) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(std::string(ERROR_SUBMIT_QUEUE));
		}
		VkSwapchainKHR swapChains[] = {swapChain};
		VkPresentInfoKHR presentInfo{
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = signalSemaphores,
			.swapchainCount = 1,
			.pSwapchains = swapChains,
			.pImageIndices = &imageIndex,
			.pResults = nullptr,
		};
		res = vkQueuePresentKHR(presentQueue, &presentInfo);
		if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR ||
			framebufferResized) {
			framebufferResized = false;
			recreateSwapChain();
		} else if (res != VK_SUCCESS) {
			throw std::runtime_error(std::string(ERROR_QUEUE_PRESENT_KHR));
		}
		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	/////////////
	// Cleanup //
	/////////////

	void cleanupSwapChain() {
		vkDestroyImageView(device, colorImageView, nullptr);
		vkDestroyImage(device, colorImage, nullptr);
		vkFreeMemory(device, colorImageMemory, nullptr);

		vkDestroyImageView(device, depthImageView, nullptr);
		vkDestroyImage(device, depthImage, nullptr);
		vkFreeMemory(device, depthImageMemory, nullptr);

		for (VkFramebuffer fb : swapChainFramebuffers) {
			vkDestroyFramebuffer(device, fb, nullptr);
		}
		for (VkImageView imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}
		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}

	void recreateSwapChain() {
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);

		// Handle minimization
		if (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}
		vkDeviceWaitIdle(device);
		cleanupSwapChain();

		createSwapChain();
		createSwapChainImageViews();
		createDepthResources();
		createColorResources();
		createFramebuffers();
	}

	void cleanup() {
		cleanupSwapChain();

		//

		// Textures
		vkDestroySampler(device, textureSampler, nullptr);
		vkDestroyImageView(device, textureImageView, nullptr);
		for (auto &image : images) {
			vkDestroyImage(device, image, nullptr);
		}
		vkFreeMemory(device, imageMemory, nullptr);

		// Uniforms
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
		}
		vkFreeMemory(device, uniformBufferMemory, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		// Vertices/ Indices
		vkDestroyBuffer(device, ivBuffer, nullptr);
		vkFreeMemory(device, ivBufferMemory, nullptr);

		// Graphics pipeline
		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);

		cleanupSyncObjects();

		vkDestroyCommandPool(device, graphicsPool, nullptr);
		vkDestroyCommandPool(device, transferPool, nullptr);
		vkDestroyDevice(device, nullptr);

		// Extension/ layers
		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}
		// Window & surface
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
		glfwDestroyWindow(window);
		glfwTerminate();
	}
};

int main() {
	HelloTriangleApplication app;

	try {
		app.run();
	} catch (const std::exception &e) {
		std::cerr << "Oh no! " << e.what() << std::endl;

		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
