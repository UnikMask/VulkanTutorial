#include <array>
#include <optional>
#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

// Debug Configuration //
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const std::vector<const char *> validationLayers = {
	"VK_LAYER_KHRONOS_validation"};

// Window information //

#define WINDOW_NAME "Vulkan Tutorial"
#define APP_NAME WINDOW_NAME
#define WAYLAND_APP_ID "vulkan_tutorial"

const uint32_t HEIGHT = 480;
const uint32_t WIDTH = 640;
const int MAX_FRAMES_IN_FLIGHT = 2;

// Application Information //

const VkApplicationInfo APP_INFO{
	.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
	.pApplicationName = "Vulkan Tutorial App",
	.applicationVersion = VK_MAKE_VERSION(0, 0, 1),
	.pEngineName = "No Engine",
	.engineVersion = VK_MAKE_VERSION(0, 0, 1),
	.apiVersion = VK_API_VERSION_1_0,
};

// Graphical extensions //

const std::vector<const char *> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

#define SWAP_SURFACE_TARGET_FORMAT VK_FORMAT_B8G8R8A8_SRGB
#define SWAP_SURFACE_TARGET_COLORSPACE VK_COLOR_SPACE_SRGB_NONLINEAR_KHR

const std::vector<VkPresentModeKHR> presentModeOrder = {
	VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_FIFO_KHR,
	VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_RELAXED_KHR};

// Shader Objects //
struct Vertex {
	glm::vec2 pos;
	glm::vec3 color;

	static VkVertexInputBindingDescription getBindingDescription();
	static std::array<VkVertexInputAttributeDescription, 2>
	getAttributeDescriptions();
};

const std::vector<Vertex> vertices = {{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
									  {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
									  {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
									  {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}};
const std::vector<uint16_t> indices = {
	0, 1, 2, 2, 3, 0,
};

// Queues & Queue Families //

struct QueueFamilyIndices {
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;
	std::optional<uint32_t> transferFamily;

	bool isComplete() {
		return graphicsFamily.has_value() && presentFamily.has_value() &&
			   transferFamily.has_value();
	}
};

struct SelectFamilyInfo {
	int graphicsDo;
	int graphicsDont;
	int transferDo;
	int transferDont;
};

// Errors //

#define VK_CREATE_INSTANCE_ERROR " - Failed to create instance!"
#define VK_ERROR_EXT_NOT_PRESENT_MSG "Vulkan extension not present"
#define VK_GENERAL_ERR_MSG "Something went wrong"

#define ERROR_VALIDATION_LAYERS_MSG                                            \
	"Some validation layer was requested, but unavailable!"
#define ERROR_DEBUG_MESSENGER " - Can't load debug messenger."

#define ERROR_NO_DEVICES "No GPU with Vulkan support present"
#define ERROR_NO_SUITABLE_DEVICES "No suitable GPU found"
#define ERROR_PICK_DEVICES " - Could not pick physical device!"

#define ERROR_CREATE_DEVICE "Failed to create logical device!"

#define ERROR_CREATE_SURFACE "Failed to create surface!"

#define ERROR_CREATE_SWAPCHAIN "- Failed to create swap chain!"
#define ERROR_WINDOW_IN_USE "Surface is already being used"
#define ERROR_DEVICE_LOST "Logical device was lost"

#define ERROR_CREATE_IMAGEVIEW "Failed to create image view!"

#define ERROR_CREATE_SHADERMODULE "Failed to create shader module!"
#define ERROR_CREATE_PIPELINE_LAYOUT "Failed to create pipeline layout!"

#define ERROR_CREATE_RENDERPASS "Failed to create render pass!"

#define ERROR_CREATE_PIPELINE "Failed to create pipeline!"

#define ERROR_CREATE_FRAMEBUFFER "Failed to create framebuffer!"

#define ERROR_CREATE_COMMAND_POOL "Failed to create command pool!"
#define ERROR_CREATE_COMMAND_BUFFER "Failed to create command buffer!"
#define ERROR_BEGIN_COMMAND_BUFFER "Failed to begin command buffer!"
#define ERROR_END_COMMAND_BUFFER "Failed to record command buffer!"

#define ERROR_CREATE_SYNC "Failed to create sync objects!"
#define ERROR_SUBMIT_QUEUE "Failed to submit draw command buffer!"

#define ERROR_ACQUIRE_NEXT_IMAGE_KHR "Failed to acquire swap chain image!"
#define ERROR_QUEUE_PRESENT_KHR "Failed to present swap chain image!"

#define ERROR_CREATE_BUFFER "Failed to create buffer!"
#define ERROR_FIND_MEMORY_TYPE_SUITABLE "Failed to find suitable memory type!"
#define ERROR_ALLOCATE_MEMORY_BUFFER "Failed to allocate buffer memory!"
