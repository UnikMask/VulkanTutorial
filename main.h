#include <array>
#include <optional>
#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdint>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
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
const float FRAMERATE_CAP = 1000 / 20.0f;

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
	VK_KHR_MAINTENANCE_1_EXTENSION_NAME,
};

#define SWAP_SURFACE_TARGET_FORMAT VK_FORMAT_B8G8R8A8_SRGB
#define SWAP_SURFACE_TARGET_COLORSPACE VK_COLOR_SPACE_SRGB_NONLINEAR_KHR

const std::vector<VkPresentModeKHR> presentModeOrder = {
	VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_FIFO_KHR,
	VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_RELAXED_KHR};

// Shader Objects //

struct Vertex {
	glm::vec4 pos;
	glm::vec4 color;
	glm::vec2 texCoord;

	static VkVertexInputBindingDescription getBindingDescription();
	static std::array<VkVertexInputAttributeDescription, 3>
	getAttributeDescriptions();
};

struct UniformBufferObject {
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

const std::vector<Vertex> vertices = {
	{{-0.5f, 0, 0.5f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
	{{0.5f, 0, 0.5f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
	{{0.5f, 0, -0.5f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}, {1.0f, 0.0f}},
	{{-0.5f, 0, -0.5f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}},

	{{-0.5f, -1.0f, 0.5f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
	{{0.5f, -1.0f, 0.5f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
	{{0.5f, -1.0f, -0.5f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}, {1.0f, 0.0f}},
	{{-0.5f, -1.0f, -0.5f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}},

};
const std::vector<uint16_t> indices = {
	0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4,
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

// Default fixed-functions //
const VkPipelineColorBlendAttachmentState DEFAULT_COLOR_BLEND_ATTACHMENT{
	.blendEnable = VK_TRUE,
	.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
	.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
	.colorBlendOp = VK_BLEND_OP_ADD,
	.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
	.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
	.alphaBlendOp = VK_BLEND_OP_ADD,
	.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
					  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,

};
const VkPipelineColorBlendStateCreateInfo DEFAULT_COLOR_BLEND_INFO{
	.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
	.logicOpEnable = VK_FALSE,
	.attachmentCount = 1,
	.pAttachments = &DEFAULT_COLOR_BLEND_ATTACHMENT,
};
const VkPipelineMultisampleStateCreateInfo MULTISAMPLING_STATE_OFF{
	.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
	.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	.sampleShadingEnable = VK_FALSE,
};
// TODO: Antialiasing
const VkPipelineMultisampleStateCreateInfo MULTISAMPLING_STATE_ON{};

const VkPipelineRasterizationStateCreateInfo RASTERIZER_DEPTH_OFF{
	.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
	.depthClampEnable = VK_FALSE,
	.rasterizerDiscardEnable = VK_FALSE,
	.polygonMode = VK_POLYGON_MODE_FILL,
	.cullMode = VK_CULL_MODE_BACK_BIT,
	.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
	.depthBiasEnable = VK_FALSE,
	// optional
	.depthBiasConstantFactor = 0.0f,
	.depthBiasClamp = 0.0f,
	.depthBiasSlopeFactor = 0.0f,
	.lineWidth = 1.0f,
};

const VkPipelineDepthStencilStateCreateInfo DEFAULT_DEPTH_NO_STENCIL{
	.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
	.depthTestEnable = VK_TRUE,
	.depthWriteEnable = VK_TRUE,
	.depthCompareOp = VK_COMPARE_OP_LESS,
	.depthBoundsTestEnable = VK_FALSE,
	.stencilTestEnable = VK_FALSE,
	.minDepthBounds = 0.0f,
	.maxDepthBounds = 1.0f,
};

const VkPipelineDepthStencilStateCreateInfo DEFAULT_DEPTH_STENCIL{
	.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
	.depthTestEnable = VK_TRUE,
	.depthWriteEnable = VK_TRUE,
	.depthCompareOp = VK_COMPARE_OP_LESS,
	.depthBoundsTestEnable = VK_FALSE,
	.stencilTestEnable = VK_FALSE,
	.minDepthBounds = 0.0f,
	.maxDepthBounds = 1.0f,
};

const VkPipelineInputAssemblyStateCreateInfo DEFAULT_INPUT_ASSEMBLY{
	.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
	.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	.primitiveRestartEnable = VK_FALSE,
};

const std::vector<VkDynamicState> DEFAULT_DYNAMIC_STATES = {
	VK_DYNAMIC_STATE_VIEWPORT,
	VK_DYNAMIC_STATE_SCISSOR,
};
const VkPipelineDynamicStateCreateInfo DEFAULT_DYNAMIC_STATE{
	.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
	.dynamicStateCount = (uint32_t)DEFAULT_DYNAMIC_STATES.size(),
	.pDynamicStates = DEFAULT_DYNAMIC_STATES.data(),
};

// Texture Sampling //
const VkSamplerCreateInfo DEFAULT_SAMPLER{
	.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
	.magFilter = VK_FILTER_LINEAR,
	.minFilter = VK_FILTER_LINEAR,
	.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
	.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
	.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
	.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
	.mipLodBias = 0.0f,
	.anisotropyEnable = VK_FALSE,
	.maxAnisotropy = 0.0f,
	.compareEnable = VK_FALSE,
	.compareOp = VK_COMPARE_OP_ALWAYS,
	.minLod = 0.0f,
	.maxLod = 0.0f,
	.borderColor = VK_BORDER_COLOR_INT_OPAQUE_WHITE,
	.unnormalizedCoordinates = VK_FALSE,
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

#define ERROR_CREATE_DESCRIPTOR_SET_LAYOUT                                     \
	"Failed to create descriptor set layout!"
#define ERROR_CREATE_DESCRIPTOR_POOL "Failed to create descriptor pool!"
#define ERROR_ALLOCATE_DESCRIPTOR_SETS "Failed to allocate descriptor sets!"

#define ERROR_STBI_LOAD_TEXTURE "Failed to load texture: "
#define ERROR_CREATE_IMAGE "Failed to create image!"
#define ERROR_CREATE_SAMPLER "Failed to create image sampler!"
#define ERROR_BARRIER_UNSUPPORTED_SRC_LAYOUT                                   \
	"Unsupported source layout in layout transition!"
#define ERROR_BARRIER_UNSUPPORTED_DST_LAYOUT                                   \
	"Unsupported destination layout in layout transition!"

#define ERROR_FIND_FORMAT "Failed to find supported format!"
