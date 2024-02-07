#include "main.h"
#include <array>
#include <cstddef>
#include <vulkan/vulkan_core.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

VkVertexInputBindingDescription Vertex::getBindingDescription() {
	VkVertexInputBindingDescription bindingDescription{
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};
	return bindingDescription;
}

std::array<VkVertexInputAttributeDescription, 2>
Vertex::getAttributeDescriptions() {
	std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
	attributeDescriptions[0] = {
		.location = 0,
		.binding = 0,
		.format = VK_FORMAT_R32G32B32A32_SFLOAT,
		.offset = offsetof(Vertex, pos),
	};
	attributeDescriptions[1] = {
		.location = 1,
		.binding = 0,
		.format = VK_FORMAT_R32G32B32A32_SFLOAT,
		.offset = offsetof(Vertex, color),
	};
	return attributeDescriptions;
}
