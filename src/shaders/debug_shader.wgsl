#import math4d::CameraUniform

struct DebugInstance {
    transform: mat4x4<f32>,
    color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<storage, read> debug_instances: array<DebugInstance>;

@vertex
fn vs_main(
    @location(0) vertex_position: vec3<f32>,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let instance = debug_instances[instance_index];
    let world_position = instance.transform * vec4<f32>(vertex_position, 1.0);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_position;
    out.color = instance.color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
