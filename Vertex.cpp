#include "glm/fwd.hpp"
#include "main.h"
#include <array>
#include <cstddef>
#include <functional>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

bool Vertex::operator==(const Vertex &other) const {
	return pos == other.pos && color == other.color &&
		   texCoord == other.texCoord;
}

namespace std {
size_t hash<Vertex>::operator()(Vertex const &vertex) const {
	return ((hash<glm::vec4>()(vertex.pos) ^
			 (hash<glm::vec4>()(vertex.color) << 1)) >>
			1) ^
		   (hash<glm::vec2>()(vertex.texCoord) << 1);
}
} // namespace std

VkVertexInputBindingDescription Vertex::getBindingDescription() {
	VkVertexInputBindingDescription bindingDescription{
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};
	return bindingDescription;
}

std::array<VkVertexInputAttributeDescription, 3>
Vertex::getAttributeDescriptions() {
	std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};
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
	attributeDescriptions[2] = {
		.location = 2,
		.binding = 0,
		.format = VK_FORMAT_R32G32_SFLOAT,
		.offset = offsetof(Vertex, texCoord),
	};
	return attributeDescriptions;
}
