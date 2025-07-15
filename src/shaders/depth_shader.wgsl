// Depth visualization shader
// Displays depth values as grayscale colors for debugging

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) depth: f32,
}

@vertex
fn vs_main(
    @location(0) vertex_position: vec4<f32>,
    @location(1) vertex_color: vec4<f32>,
    @location(2) vertex_normal: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Extract visibility flag from position.w component
    let visibility = vertex_position.w;
    
    // If not visible (culled), move vertex off-screen
    if (visibility < 0.5) {
        out.clip_position = vec4<f32>(0.0, 0.0, -1.0, 1.0); // Behind camera
        out.depth = 0.0;
    } else {
        // Use xyz components for actual position
        let position_3d = vec4<f32>(vertex_position.x, vertex_position.y, vertex_position.z, 1.0);
        out.clip_position = camera.view_proj * position_3d;
        
        // Store depth value in view space (negative z)
        let view_pos = camera.view_proj * position_3d;
        out.depth = view_pos.z / view_pos.w;
    }
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Normalize depth to 0-1 range and display as grayscale
    // Closer objects (smaller z) are brighter, farther objects are darker
    let normalized_depth = clamp((1.0 - in.depth) * 0.5, 0.0, 1.0);
    return vec4<f32>(normalized_depth, normalized_depth, normalized_depth, 1.0);
}