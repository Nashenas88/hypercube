// Compute shader for 4D hypercube transformations and projections

struct Transform4D {
    rotation_matrix: mat4x4<f32>,
    viewer_distance: f32,
    sticker_scale: f32,
    face_spacing: f32,
    _padding: f32,
}

struct StickerInput {
    position_4d: vec4<f32>,
    color: vec4<f32>,
    face_center_4d: vec4<f32>,
}

struct VertexOutput {
    position: vec4<f32>,  // 4th component unused but needed for alignment
    color: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> transform: Transform4D;

@group(0) @binding(1)
var<storage, read> input_stickers: array<StickerInput>;

@group(0) @binding(2)
var<storage, read_write> output_vertices: array<VertexOutput>;

// Projects a 4D point to 3D space using perspective projection
fn project_4d_to_3d(point_4d: vec4<f32>, viewer_distance: f32) -> vec3<f32> {
    let w_distance = viewer_distance - point_4d.w;
    let scale = viewer_distance / w_distance;
    return vec3<f32>(point_4d.x * scale, point_4d.y * scale, point_4d.z * scale);
}

// Determine which axis is fixed based on face center (which coordinate is ±1.0)
fn get_fixed_axis(face_center: vec4<f32>) -> u32 {
    if (abs(face_center.x) > 0.5) { return 0u; }  // X axis fixed
    if (abs(face_center.y) > 0.5) { return 1u; }  // Y axis fixed  
    if (abs(face_center.z) > 0.5) { return 2u; }  // Z axis fixed
    return 3u;  // W axis fixed
}

// Generate cube vertex offsets based on which axis is fixed
fn generate_cube_vertex_4d(vertex_index: u32, fixed_axis: u32, center_4d: vec4<f32>, cube_scale: f32) -> vec4<f32> {
    // Get the vertex offset in the exact order expected by INDICES (matches original VERTICES array)
    var offset: vec3<f32>;
    switch vertex_index {
        case 0u: { offset = vec3<f32>(-0.333333, -0.333333, -0.333333); }
        case 1u: { offset = vec3<f32>( 0.333333, -0.333333, -0.333333); }
        case 2u: { offset = vec3<f32>( 0.333333,  0.333333, -0.333333); }
        case 3u: { offset = vec3<f32>(-0.333333,  0.333333, -0.333333); }
        case 4u: { offset = vec3<f32>(-0.333333, -0.333333,  0.333333); }
        case 5u: { offset = vec3<f32>( 0.333333, -0.333333,  0.333333); }
        case 6u: { offset = vec3<f32>( 0.333333,  0.333333,  0.333333); }
        case 7u: { offset = vec3<f32>(-0.333333,  0.333333,  0.333333); }
        default: { offset = vec3<f32>(0.0, 0.0, 0.0); }
    }
    
    offset = offset * cube_scale;
    
    // Apply offset to the 3 free axes (not the fixed axis)
    var vertex_4d = center_4d;
    var offset_idx = 0u;
    
    for (var axis = 0u; axis < 4u; axis++) {
        if (axis != fixed_axis) {
            if (offset_idx == 0u) {
                vertex_4d[axis] += offset.x;
            } else if (offset_idx == 1u) {
                vertex_4d[axis] += offset.y;
            } else if (offset_idx == 2u) {
                vertex_4d[axis] += offset.z;
            }
            offset_idx++;
        }
    }
    
    return vertex_4d;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input_stickers)) {
        return;
    }
    
    let sticker = input_stickers[index];
    
    // Get sticker offset from its face center
    let sticker_offset_4d = sticker.position_4d - sticker.face_center_4d;
    
    // Scale face centers and keep original sticker positions
    let scaled_face_center = sticker.face_center_4d * transform.face_spacing;
    let sticker_center_4d = scaled_face_center + sticker_offset_4d;
    
    // Determine which axis is fixed for this face
    let fixed_axis = get_fixed_axis(sticker.face_center_4d);
    
    // Generate all 8 vertices of the cube in 4D space
    for (var i = 0u; i < 8u; i++) {
        // Generate cube vertex in 4D space around sticker center
        let vertex_4d = generate_cube_vertex_4d(i, fixed_axis, sticker_center_4d, transform.sticker_scale);
        
        // Apply 4D rotation to the vertex
        let rotated_vertex_4d = transform.rotation_matrix * vertex_4d;
        
        // Project this vertex to 3D
        let projected_vertex = project_4d_to_3d(rotated_vertex_4d, transform.viewer_distance);
        
        // Store the projected vertex in flat array
        let vertex_output_index = index * 8u + i;
        output_vertices[vertex_output_index].position = vec4<f32>(projected_vertex.x, projected_vertex.y, projected_vertex.z, 1.0);
        output_vertices[vertex_output_index].color = sticker.color;
    }
}