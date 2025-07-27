#version 430 core
layout(std430, binding=0) buffer Trajectories {
    vec3 positions[];
};

uniform int trailLen;
uniform int head;  // shared head for all particles

void main() {
    int particleID = gl_InstanceID;

    // Wrap using bitmask instead of modulo
    int idx = (head + 1 + gl_VertexID) & (trailLen - 1);

    int base = particleID * trailLen + idx;
    vec3 pos = positions[base];

    gl_Position = vec4(pos, 1.0);
}
