// Depth visualization shader using instanced rendering
// Displays depth values as grayscale colors for debugging

// Transform uniform structure
struct Transform4D {
    rotation_matrix: mat4x4<f32>,
    viewer_distance: f32,
    sticker_scale: f32,
    face_spacing: f32,
    _padding: f32,
}

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

struct FaceDataUniform {
    face_centers: array<vec4<f32>, 8>,
    fixed_dims: array<vec4<u32>, 8>,
}

// Instance data for each sticker
struct StickerInstance {
    position_4d: vec4<f32>,
    color: vec4<f32>,
    face_id: u32,
    _padding: array<u32, 3>,
}

@group(0) @binding(0)
var<uniform> transform: Transform4D;

@group(0) @binding(1)
var<uniform> camera: CameraUniform;

@group(0) @binding(2)
var<uniform> face_data: FaceDataUniform;

@group(0) @binding(3)
var<storage, read> instances: array<StickerInstance>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) depth: f32,
}

// Shared math4d functions


fn project_4d_to_3d(point_4d: vec4<f32>, viewer_distance: f32) -> vec3<f32> {
    let w_distance = viewer_distance - point_4d.w;
    let scale = viewer_distance / w_distance;
    return vec3<f32>(point_4d.x * scale, point_4d.y * scale, point_4d.z * scale);
}

fn is_face_visible(face_center_4d: vec4<f32>, rotation_matrix: mat4x4<f32>, viewer_distance: f32) -> bool {
    let rotated_face_center = rotation_matrix * face_center_4d;
    let viewer_position = vec4<f32>(0.0, 0.0, 0.0, viewer_distance);
    let to_viewer = viewer_position - rotated_face_center;
    let dot_product = dot(rotated_face_center, to_viewer);
    return dot_product < 0.0;
}

fn calculate_sticker_center_4d(sticker_position_4d: vec4<f32>, face_center_4d: vec4<f32>, face_spacing: f32) -> vec4<f32> {
    let sticker_offset_4d = sticker_position_4d - face_center_4d;
    let scaled_face_center = face_center_4d * face_spacing;
    return scaled_face_center + sticker_offset_4d;
}

@vertex
fn vs_main(
    @location(0) vertex_position: vec3<f32>,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Get instance data
    let instance = instances[instance_index];
    
    // Get face center from face_id
    let face_center_4d = face_data.face_centers[instance.face_id];
    let fixed_dim = face_data.fixed_dims[instance.face_id][0];
    
    // Calculate sticker center in 4D
    let sticker_center_4d = calculate_sticker_center_4d(instance.position_4d, face_center_4d, transform.face_spacing);
    
    // Check if this face is visible (4D culling)
    let face_visible = is_face_visible(face_center_4d, transform.rotation_matrix, transform.viewer_distance);
    
    if (!face_visible) {
        // Face is culled - move vertex off-screen
        out.clip_position = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        out.depth = 0.0;
        return out;
    }
    
    // Get cube vertex from vertex attribute
    let local_vertex = vertex_position * transform.sticker_scale;
    
    // Generate vertex in 4D space around sticker center
    var vertex_4d = sticker_center_4d;
    var offset_idx = 0u;
    
    for (var axis = 0u; axis < 4u; axis++) {
        if (axis != fixed_dim) {
            if (offset_idx == 0u) {
                vertex_4d[axis] += local_vertex.x;
            } else if (offset_idx == 1u) {
                vertex_4d[axis] += local_vertex.y;
            } else if (offset_idx == 2u) {
                vertex_4d[axis] += local_vertex.z;
            }
            offset_idx++;
        }
    }
    
    // Apply 4D rotation
    let rotated_vertex_4d = transform.rotation_matrix * vertex_4d;
    
    // Project to 3D
    let vertex_3d = project_4d_to_3d(rotated_vertex_4d, transform.viewer_distance);
    
    // Apply 3D view/projection matrix
    let clip_pos = camera.view_proj * vec4<f32>(vertex_3d, 1.0);
    out.clip_position = clip_pos;
    
    // Store depth value in view space
    out.depth = clip_pos.z / clip_pos.w;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Normalize depth to 0-1 range and display as grayscale
    // Closer objects (smaller z) are brighter, farther objects are darker
    let normalized_depth = clamp((1.0 - in.depth) * 0.5, 0.0, 1.0);
    return vec4<f32>(normalized_depth, normalized_depth, normalized_depth, 1.0);
}