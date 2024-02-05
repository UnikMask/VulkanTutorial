#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <ostream>
#include <pthread.h>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <vulkan/vulkan_core.h>

#include "main.h"
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

	// Swap Chains //
	VkSurfaceKHR surface;
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;

	// Graphics Pipeline //
	VkRenderPass renderPass;
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;
	VkViewport viewport;
	VkRect2D scissor;

	// Drawing //
	std::vector<VkFramebuffer> swapChainFramebuffers;
	VkCommandPool commandPool;
	std::vector<VkCommandBuffer> commandBuffers;

	// Synchronisation //
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	uint32_t currentFrame = 0; // Frames-in-flight feature

	////////////////////
	// Initialization //
	////////////////////

	void initVulkan() {
		// Setup
		createInstance();
		setupDebugMessenger();

		// Presentation
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();

		// Graphics Pipeline
		createSwapChain();
		createImageView();
		createRenderPass();
		createGraphicsPipeline();

		// Drawing
		createFramebuffers();
		createCommandPool();
		createCommandBuffers();
		createSyncObjects();
	}

	void initWindow() {
		// if (glfwPlatformSupported(GLFW_PLATFORM_WAYLAND)) {
		//   glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);
		// } else {
		glfwInitHint(GLFW_PLATFORM, GLFW_ANY_PLATFORM);
		// }
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHintString(GLFW_X11_CLASS_NAME, "vulkan_tutorial");
		glfwWindowHintString(GLFW_WAYLAND_APP_ID, WAYLAND_APP_ID);
		glfwWindowHint(GLFW_REFRESH_RATE, 60);
		window = glfwCreateWindow(WIDTH, HEIGHT, WINDOW_NAME, nullptr, nullptr);
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
		std::cerr << "Validation layer: " << pCallbackData->pMessage
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

	struct QueueFamilyIndices {
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;

		bool isComplete() {
			return graphicsFamily.has_value() && presentFamily.has_value();
		}
	};

	QueueFamilyIndices
	selectFamily(std::vector<VkQueueFamilyProperties> queueFamilies,
				 VkPhysicalDevice device, int flags) {
		QueueFamilyIndices res;

		int i = 0;
		for (const auto &qf : queueFamilies) {
			if (qf.queueFlags & flags) {
				res.graphicsFamily = i;
			}
			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface,
												 &presentSupport);
			if (presentSupport) {
				res.presentFamily = i;
			}
			if (res.isComplete()) {
				break;
			}
			i++;
		}
		return res;
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
		QueueFamilyIndices indices;
		uint32_t familyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilies(familyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount,
												 queueFamilies.data());

		return selectFamily(queueFamilies, device, VK_QUEUE_GRAPHICS_BIT);
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
		for (auto device : devices) {
			int score = getDeviceScore(device);
			if (score > 0) {
				candidates.insert(std::make_pair(score, device));
			}
		}
		if (!candidates.empty()) {
			physicalDevice = candidates.rbegin()->second;
			std::cout << "Chosen device score: " << candidates.rbegin()->first
					  << std::endl;
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

	/////////////////////
	// Logical Devices //
	/////////////////////

	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		float queuePriority = 1.0f;

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

		std::set<uint32_t> uniqueQueueFamilies = {
			indices.graphicsFamily.value(), indices.presentFamily.value()};

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
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, surface,
												  &details.capabilities);
		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &formatCount,
											 nullptr);
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
		return presentModes[0];
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
			.queueFamilyIndexCount = 0,
			.pQueueFamilyIndices = nullptr,
			.preTransform = swapChainSupport.capabilities.currentTransform,
			.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			.presentMode = presentMode,
			.clipped = VK_TRUE,
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

	void createImageView() {
		swapChainImageViews.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkImageViewCreateInfo createInfo{
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = swapChainImages[i],
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = swapChainImageFormat,
				.components =
					{
						.r = VK_COMPONENT_SWIZZLE_IDENTITY,
						.g = VK_COMPONENT_SWIZZLE_IDENTITY,
						.b = VK_COMPONENT_SWIZZLE_IDENTITY,
						.a = VK_COMPONENT_SWIZZLE_IDENTITY,
					},
				.subresourceRange =
					{
						.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						.baseMipLevel = 0,
						.levelCount = 1,
						.baseArrayLayer = 0,
						.layerCount = 1,
					},
			};

			int res = vkCreateImageView(device, &createInfo, nullptr,
										&swapChainImageViews[i]);
			switch (res) {
			case VK_SUCCESS:
				break;
			default:
				throw std::runtime_error(ERROR_CREATE_IMAGEVIEW);
			}
		}
	}

	///////////////////////
	// Graphics Pipeline //
	///////////////////////

	void createRenderPass() {
		VkAttachmentDescription colorAttachment{
			.format = swapChainImageFormat,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		};

		VkAttachmentReference colorAttachmentRef{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};
		std::vector<VkAttachmentReference> attachmentRefs = {
			colorAttachmentRef};
		VkSubpassDescription subpass{
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount =
				static_cast<uint32_t>(attachmentRefs.size()),
			.pColorAttachments = attachmentRefs.data(),
		};

		VkSubpassDependency dependency{
			.srcSubpass = VK_SUBPASS_EXTERNAL,
			.dstSubpass = 0,
			.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		};

		VkRenderPassCreateInfo renderPassInfo{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments = &colorAttachment,
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
		// Create shader modules & create shader stages
		auto vertShaderCode = readFile("./tutorial_vert.spv");
		auto fragShaderCode = readFile("./tutorial_frag.spv");

		VkShaderModule vertShader = createShaderModule(vertShaderCode);
		VkShaderModule fragShader = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertShader,
			.pName = "main",
		};

		VkPipelineShaderStageCreateInfo fragCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragShader,
			.pName = "main",
		};

		VkPipelineShaderStageCreateInfo shaderStages[] = {vertCreateInfo,
														  fragCreateInfo};

		// Setup vertex input, and assembly state
		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();
		VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &bindingDescription,
			.vertexAttributeDescriptionCount = 1,
			.pVertexAttributeDescriptions = attributeDescriptions.data(),
		};

		VkPipelineInputAssemblyStateCreateInfo assemblyStateCreateInfo{
			.sType =
				VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			.primitiveRestartEnable = VK_FALSE,
		};

		// Setup viewport and scissor
		viewport = {
			.x = 0.0f,
			.y = 0.0f,
			.width = (float)swapChainExtent.width,
			.height = (float)swapChainExtent.height,
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};
		scissor = {
			.offset = {0, 0},
			.extent = swapChainExtent,
		};

		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR,
		};
		VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
			.pDynamicStates = dynamicStates.data(),
		};

		VkPipelineViewportStateCreateInfo viewportState{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.scissorCount = 1,
		};

		// Setup Rasterizer
		VkPipelineRasterizationStateCreateInfo rasterizer{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.depthClampEnable = VK_FALSE,
			.rasterizerDiscardEnable = VK_FALSE,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.depthBiasEnable = VK_FALSE,
			// optional
			.depthBiasConstantFactor = 0.0f,
			.depthBiasClamp = 0.0f,
			.depthBiasSlopeFactor = 0.0f,
			.lineWidth = 1.0f,
		};

		// TODO: Multisampling - requires GPU feature, disabled rn
		VkPipelineMultisampleStateCreateInfo multisampling{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = VK_FALSE,
			// optional features
		};

		// No depth & stencil testing right now, ignore

		// Color Blending
		VkPipelineColorBlendAttachmentState colorBlendAttachment{
			// Implement ALPHA blending
			.blendEnable = VK_TRUE,
			.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
			.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
			.colorBlendOp = VK_BLEND_OP_ADD,
			.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
			.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
			.alphaBlendOp = VK_BLEND_OP_ADD,
			.colorWriteMask =
				VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
				VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
		};

		VkPipelineColorBlendStateCreateInfo colorBlending{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.logicOpEnable = VK_FALSE,
			.attachmentCount = 1,
			.pAttachments = &colorBlendAttachment,
		};

		// Create empty pipeline layout - useful later on for uniform buffers.
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		};
		int res = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
										 &pipelineLayout);

		switch (res) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(std::string(ERROR_CREATE_PIPELINE_LAYOUT));
		}

		// Create graphics pipeline
		VkGraphicsPipelineCreateInfo pipelineInfo{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = shaderStages,
			.pVertexInputState = &vertexInputCreateInfo,
			.pInputAssemblyState = &assemblyStateCreateInfo,
			.pViewportState = &viewportState,
			.pRasterizationState = &rasterizer,
			.pMultisampleState = &multisampling,
			.pDepthStencilState = nullptr,
			.pColorBlendState = &colorBlending,
			.pDynamicState = &dynamicStateCreateInfo,
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
			VkImageView attachments[] = {swapChainImageViews[i]};
			VkFramebufferCreateInfo framebufferInfo{
				.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				.renderPass = renderPass,
				.attachmentCount = 1,
				.pAttachments = attachments,
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

		VkCommandPoolCreateInfo poolInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(),
		};

		int res = vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
		switch (res) {
		case VK_SUCCESS:
			break;
		default:
			throw std::runtime_error(std::string(ERROR_CREATE_COMMAND_POOL));
		}
	}

	void createCommandBuffers() {
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		VkCommandBufferAllocateInfo allocInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = commandPool,
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

		VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
		VkRenderPassBeginInfo renderPassInfo{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = renderPass,
			.framebuffer = swapChainFramebuffers[imageIndex],
			.renderArea =
				{
					.offset = {0, 0},
					.extent = swapChainExtent,
				},
			.clearValueCount = 1,
			.pClearValues = &clearColor,
		};

		// Run commands
		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
							 VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
						  graphicsPipeline);
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
		vkCmdDraw(commandBuffer, 3, 1, 0, 0);
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
		VkSemaphoreCreateInfo semaphoreInfo{
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		};
		VkFenceCreateInfo fenceInfo{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT,
		};

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

	//////////
	// Loop //
	//////////

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			drawFrame();
		}
		vkDeviceWaitIdle(device);
	}

	void drawFrame() {
		// Wait for last frame to finish
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE,
						UINT64_MAX);
		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		// Get image index from swap chain and device
		uint32_t imageIndex;
		vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
							  imageAvailableSemaphores[currentFrame],
							  VK_NULL_HANDLE, &imageIndex);

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
		int res = vkQueueSubmit(graphicsQueue, 1, &submitInfo,
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
		vkQueuePresentKHR(graphicsQueue, &presentInfo);
		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	/////////////
	// Cleanup //
	/////////////

	void cleanup() {
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}
		vkDestroyCommandPool(device, commandPool, nullptr);
		for (auto framebuffer : swapChainFramebuffers) {
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}
		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);
		for (auto view : swapChainImageViews) {
			vkDestroyImageView(device, view, nullptr);
		}
		vkDestroySwapchainKHR(device, swapChain, nullptr);
		vkDestroyDevice(device, nullptr);

		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}
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
