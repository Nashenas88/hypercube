#import math4d::{compute_vertex_geometry}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) depth: f32,
}

@vertex
fn vs_main(
    @location(0) vertex_position: vec3<f32>,
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let geometry = compute_vertex_geometry(instance_index, vertex_position, vertex_index);

    out.clip_position = geometry.clip_position;

    if (!geometry.visible) {
        out.depth = 0.0;
        return out;
    }

    out.depth = geometry.clip_position.z / geometry.clip_position.w;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Inverted so closer objects (smaller z) render brighter.
    let normalized_depth = clamp((1.0 - in.depth) * 0.5, 0.0, 1.0);
    return vec4<f32>(normalized_depth, normalized_depth, normalized_depth, 1.0);
}