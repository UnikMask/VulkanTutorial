// vim:ft=glsl
#version 460

layout (binding = 1) uniform sampler2D texSampler;

layout (location = 0) in vec4 fragColor;
layout (location = 1) in vec2 fragTexCoord;
layout (location = 0) out vec4 outColor;

const int ditherMatrix[16] = int[](0,  8,  2,  10,
                                  12, 4,  14, 6,
                                  3,  11, 1,  9,
                                  15, 7,  13, 5);
const float ditherq = 1.0f;


vec4 dither(vec4 color) {
    float offset = ditherMatrix[int(mod(gl_FragCoord.x, 4)) + int(mod(gl_FragCoord.y, 4)) * 4] / 16.0;
    color += offset * ditherq;
    color -= mod(color, vec4(ditherq));
    return color;
}

void main() {
    outColor = dither(texture(texSampler, fragTexCoord));
}
