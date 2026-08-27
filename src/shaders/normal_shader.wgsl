// Normal visualization shader using instanced rendering
// Displays normal vectors as colors for debugging
#import math4d::{compute_vertex_geometry}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
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
        out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return out;
    }

    out.color = vec4<f32>(geometry.world_normal * 0.5 + 0.5, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}