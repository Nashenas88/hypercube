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
    position: vec4<f32>,  // 4th component stores visibility flag (1.0 = visible, 0.0 = culled)
    color: vec4<f32>,
    normal: vec4<f32>,    // Normal vector for this vertex (w component unused)
}

struct NormalOutput {
    normal: vec4<f32>,    // 3D normal + padding
}

@group(0) @binding(0)
var<uniform> transform: Transform4D;

@group(0) @binding(1)
var<storage, read> input_stickers: array<StickerInput>;

@group(0) @binding(2)
var<storage, read_write> output_vertices: array<VertexOutput>;

@group(0) @binding(3)
var<storage, read_write> output_normals: array<NormalOutput>;

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
    
    // Generate all 8 vertices of the cube in 4D space and project them to 3D
    var projected_vertices: array<vec3<f32>, 8>;
    for (var i = 0u; i < 8u; i++) {
        // Generate cube vertex in 4D space around sticker center
        let vertex_4d = generate_cube_vertex_4d(i, fixed_axis, sticker_center_4d, transform.sticker_scale);
        
        // Apply 4D rotation to the vertex
        let rotated_vertex_4d = transform.rotation_matrix * vertex_4d;
        
        // Project this vertex to 3D
        projected_vertices[i] = project_4d_to_3d(rotated_vertex_4d, transform.viewer_distance);
    }
    
    // Calculate actual face normals from the projected vertices
    // Define which vertices form each face (using first 3 vertices of each face for normal calculation)
    var face_vertex_indices: array<array<u32, 3>, 6>;
    face_vertex_indices[0] = array<u32, 3>(0u, 1u, 2u); // Front
    face_vertex_indices[1] = array<u32, 3>(1u, 5u, 6u); // Right
    face_vertex_indices[2] = array<u32, 3>(5u, 4u, 7u); // Back
    face_vertex_indices[3] = array<u32, 3>(4u, 0u, 3u); // Left
    face_vertex_indices[4] = array<u32, 3>(3u, 2u, 6u); // Top
    face_vertex_indices[5] = array<u32, 3>(4u, 5u, 1u); // Bottom
    
    // Calculate the center of the projected cube for normal orientation check
    var cube_center = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < 8u; i++) {
        cube_center += projected_vertices[i];
    }
    cube_center = cube_center / 8.0;

    // Calculate normals from actual projected vertices
    var face_normals: array<vec3<f32>, 6>;
    for (var face_3d = 0u; face_3d < 6u; face_3d++) {
        let v0 = projected_vertices[face_vertex_indices[face_3d][0]];
        let v1 = projected_vertices[face_vertex_indices[face_3d][1]];
        let v2 = projected_vertices[face_vertex_indices[face_3d][2]];
        
        // Calculate normal using cross product
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        var normal = normalize(cross(edge1, edge2));
        
        // Use 4D face orientation to determine correct normal direction
        // Calculate the 4D face normal direction before projection
        let face_center_4d = sticker.face_center_4d;
        let rotated_face_center_4d = transform.rotation_matrix * face_center_4d;
        
        // Project the 4D face normal to 3D
        let face_normal_4d_direction = rotated_face_center_4d.xyz;
        
        // Ensure normal points in the same general direction as the 4D face normal
        if (dot(normal, normalize(face_normal_4d_direction)) < 0.0) {
            normal = -normal;
        }
        
        face_normals[face_3d] = normal;
    }
    
    // Test if this face should be visible based on 4D orientation
    let rotated_face_center = transform.rotation_matrix * sticker.face_center_4d;
    
    // Vector from face center to viewer (viewer is at positive W looking toward negative W)
    let viewer_position = vec4<f32>(0.0, 0.0, 0.0, transform.viewer_distance);
    let to_viewer = viewer_position - rotated_face_center;
    
    // Face is visible if it's facing toward the viewer
    let dot_product = dot(rotated_face_center, to_viewer);
    let visibility = select(0.0, 1.0, dot_product < 0.0);
    
    // Store 36 vertices (6 faces × 6 vertices per face)
    // Each triangle gets its own vertices with correct normals baked in
    var vertex_output_index = index * 36u;
    
    // Define face triangle vertex indices (same as INDICES pattern)
    var face_triangles: array<array<u32, 6>, 6>;
    face_triangles[0] = array<u32, 6>(0u, 1u, 2u, 2u, 3u, 0u); // front
    face_triangles[1] = array<u32, 6>(1u, 5u, 6u, 6u, 2u, 1u); // right
    face_triangles[2] = array<u32, 6>(5u, 4u, 7u, 7u, 6u, 5u); // back
    face_triangles[3] = array<u32, 6>(4u, 0u, 3u, 3u, 7u, 4u); // left
    face_triangles[4] = array<u32, 6>(3u, 2u, 6u, 6u, 7u, 3u); // top
    face_triangles[5] = array<u32, 6>(4u, 5u, 1u, 1u, 0u, 4u); // bottom
    
    // Generate vertices for each face
    for (var face_id = 0u; face_id < 6u; face_id++) {
        let face_normal = face_normals[face_id];
        
        // Generate 6 vertices for this face (2 triangles)
        for (var vert_in_face = 0u; vert_in_face < 6u; vert_in_face++) {
            let cube_vertex_index = face_triangles[face_id][vert_in_face];
            let vertex_pos = projected_vertices[cube_vertex_index];
            
            output_vertices[vertex_output_index].position = vec4<f32>(
                vertex_pos.x,
                vertex_pos.y, 
                vertex_pos.z,
                visibility
            );
            output_vertices[vertex_output_index].color = sticker.color;
            output_vertices[vertex_output_index].normal = vec4<f32>(face_normal, 0.0);
            
            vertex_output_index++;
        }
    }
    
    // Store 6 calculated normals for the cube faces
    for (var face_3d = 0u; face_3d < 6u; face_3d++) {
        let normal_output_index = index * 6u + face_3d;
        output_normals[normal_output_index].normal = vec4<f32>(face_normals[face_3d], 0.0);
    }
}