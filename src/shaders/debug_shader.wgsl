// Debug shader for transparent AABB rendering
// Uses instanced rendering with storage buffer for debug instances

// Debug instance data
struct DebugInstance {
    transform: mat4x4<f32>,
    color: vec4<f32>,
}

// Camera uniform
struct CameraUniform {
    view_proj: mat4x4<f32>,
}

// Vertex shader output
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
    // Get debug instance data
    let instance = debug_instances[instance_index];
    
    // Transform vertex position by instance transform
    let world_position = instance.transform * vec4<f32>(vertex_position, 1.0);
    
    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_position;
    out.color = instance.color;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple transparent rendering with instance color
    return in.color;
}