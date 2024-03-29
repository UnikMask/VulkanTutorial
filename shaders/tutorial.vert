// vim:ft=glsl
#version 460

layout (binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout (location = 0) in vec4 inPosition;
layout (location = 1) in vec4 inColor;
layout (location = 2) in vec2 inTexCoord;

layout (location = 0) out vec4 fragColor;
layout (location = 1) out vec2 fragTexCoord;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * inPosition;
    fragColor = inColor;
    fragTexCoord = inTexCoord;
}
