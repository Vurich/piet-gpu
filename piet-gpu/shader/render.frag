#version 450
#pragma optimize(on)
#pragma shader_stage(fragment)

layout(location = 0) in vec2 v_TexCoord;

layout(set = 0, binding = 0) uniform texture2D t_Output;
layout(set = 0, binding = 1) uniform sampler s_Color;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(sampler2D(t_Output, s_Color), v_TexCoord);
}
