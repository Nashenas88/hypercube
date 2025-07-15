// Compute shader for 4D hypercube transformations and projections

struct Transform4D {
    rotation_matrix: mat4x4<f32>,
    viewer_distance: f32,
    sticker_spacing: f32,
    face_spacing: f32,
    _padding: f32,
}

struct StickerInput {
    position_4d: vec4<f32>,
    color: vec4<f32>,
}

struct InstanceOutput {
    model_matrix: mat4x4<f32>,
    color: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> transform: Transform4D;

@group(0) @binding(1)
var<storage, read> input_stickers: array<StickerInput>;

@group(0) @binding(2)
var<storage, read_write> output_instances: array<InstanceOutput>;

// Projects a 4D point to 3D space using perspective projection
fn project_4d_to_3d(point_4d: vec4<f32>, viewer_distance: f32) -> vec3<f32> {
    let w_distance = viewer_distance - point_4d.w;
    let scale = viewer_distance / w_distance;
    return vec3<f32>(point_4d.x * scale, point_4d.y * scale, point_4d.z * scale);
}

// Creates a translation matrix from a 3D vector
fn translation_matrix(translation: vec3<f32>) -> mat4x4<f32> {
    return mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(translation.x, translation.y, translation.z, 1.0)
    );
}

// Creates a uniform scaling matrix
fn scaling_matrix(scale: f32) -> mat4x4<f32> {
    return mat4x4<f32>(
        vec4<f32>(scale, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, scale, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, scale, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );
}

// Creates a model matrix that could be warped based on 4D projection effects
fn create_warped_model_matrix(position_4d: vec4<f32>, projected_3d: vec3<f32>) -> mat4x4<f32> {
    // Base scale for stickers
    let base_scale = 0.8;
    
    // Calculate warping effects based on W coordinate
    // Objects further in 4D space (higher |w|) appear smaller and more distorted
    let w_distance = abs(position_4d.w);
    let depth_scale = 1.0 / (1.0 + w_distance * 0.2);
    
    // Create perspective-based warping
    let warp_scale = base_scale * depth_scale;
    
    // Combine translation and scaling
    let scale_mat = scaling_matrix(warp_scale);
    let trans_mat = translation_matrix(projected_3d);
    
    return trans_mat * scale_mat;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input_stickers)) {
        return;
    }
    
    let sticker = input_stickers[index];
    
    // Apply 4D rotation
    let rotated_4d = transform.rotation_matrix * (sticker.position_4d * transform.sticker_spacing);
    
    // Project to 3D
    let projected_3d = project_4d_to_3d(rotated_4d, transform.viewer_distance);
    
    // Create warped model matrix based on 4D effects
    let model_matrix = create_warped_model_matrix(rotated_4d, projected_3d);
    
    // Write output
    output_instances[index].model_matrix = model_matrix;
    output_instances[index].color = sticker.color;
}