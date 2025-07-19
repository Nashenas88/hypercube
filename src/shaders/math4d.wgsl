// Shared 4D mathematics functions for hypercube rendering
// Contains common transformations and projections used across all shaders

// Transform uniform structure shared across shaders
struct Transform4D {
    rotation_matrix: mat4x4<f32>,
    viewer_distance: f32,
    sticker_scale: f32,
    face_spacing: f32,
    _padding: f32,
}

// Static cube geometry: 36 vertices (6 faces × 6 vertices per face)
// Vertices are ordered by face: front, right, back, left, top, bottom
// Each face uses 2 triangles with proper winding order
const CUBE_VERTICES: array<vec3<f32>, 36> = array<vec3<f32>, 36>(
    // Front face (2 triangles: 0,1,2 and 2,3,0)
    vec3<f32>(-0.333333, -0.333333, -0.333333), // 0
    vec3<f32>( 0.333333, -0.333333, -0.333333), // 1
    vec3<f32>( 0.333333,  0.333333, -0.333333), // 2
    vec3<f32>( 0.333333,  0.333333, -0.333333), // 2
    vec3<f32>(-0.333333,  0.333333, -0.333333), // 3
    vec3<f32>(-0.333333, -0.333333, -0.333333), // 0
    
    // Right face (2 triangles: 1,5,6 and 6,2,1)
    vec3<f32>( 0.333333, -0.333333, -0.333333), // 1
    vec3<f32>( 0.333333, -0.333333,  0.333333), // 5
    vec3<f32>( 0.333333,  0.333333,  0.333333), // 6
    vec3<f32>( 0.333333,  0.333333,  0.333333), // 6
    vec3<f32>( 0.333333,  0.333333, -0.333333), // 2
    vec3<f32>( 0.333333, -0.333333, -0.333333), // 1
    
    // Back face (2 triangles: 5,4,7 and 7,6,5)
    vec3<f32>( 0.333333, -0.333333,  0.333333), // 5
    vec3<f32>(-0.333333, -0.333333,  0.333333), // 4
    vec3<f32>(-0.333333,  0.333333,  0.333333), // 7
    vec3<f32>(-0.333333,  0.333333,  0.333333), // 7
    vec3<f32>( 0.333333,  0.333333,  0.333333), // 6
    vec3<f32>( 0.333333, -0.333333,  0.333333), // 5
    
    // Left face (2 triangles: 4,0,3 and 3,7,4)
    vec3<f32>(-0.333333, -0.333333,  0.333333), // 4
    vec3<f32>(-0.333333, -0.333333, -0.333333), // 0
    vec3<f32>(-0.333333,  0.333333, -0.333333), // 3
    vec3<f32>(-0.333333,  0.333333, -0.333333), // 3
    vec3<f32>(-0.333333,  0.333333,  0.333333), // 7
    vec3<f32>(-0.333333, -0.333333,  0.333333), // 4
    
    // Top face (2 triangles: 3,2,6 and 6,7,3)
    vec3<f32>(-0.333333,  0.333333, -0.333333), // 3
    vec3<f32>( 0.333333,  0.333333, -0.333333), // 2
    vec3<f32>( 0.333333,  0.333333,  0.333333), // 6
    vec3<f32>( 0.333333,  0.333333,  0.333333), // 6
    vec3<f32>(-0.333333,  0.333333,  0.333333), // 7
    vec3<f32>(-0.333333,  0.333333, -0.333333), // 3
    
    // Bottom face (2 triangles: 4,5,1 and 1,0,4)
    vec3<f32>(-0.333333, -0.333333,  0.333333), // 4
    vec3<f32>( 0.333333, -0.333333,  0.333333), // 5
    vec3<f32>( 0.333333, -0.333333, -0.333333), // 1
    vec3<f32>( 0.333333, -0.333333, -0.333333), // 1
    vec3<f32>(-0.333333, -0.333333, -0.333333), // 0
    vec3<f32>(-0.333333, -0.333333,  0.333333), // 4
);

// Pre-calculated normals for each vertex (matches CUBE_VERTICES order)
const CUBE_NORMALS: array<vec3<f32>, 36> = array<vec3<f32>, 36>(
    // Front face - all vertices have normal pointing in -Z direction
    vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 0.0, -1.0),
    vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 0.0, -1.0),
    
    // Right face - all vertices have normal pointing in +X direction
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
    
    // Back face - all vertices have normal pointing in +Z direction
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0),
    
    // Left face - all vertices have normal pointing in -X direction
    vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0),
    vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0),
    
    // Top face - all vertices have normal pointing in +Y direction
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
    
    // Bottom face - all vertices have normal pointing in -Y direction
    vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, -1.0, 0.0),
    vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, -1.0, 0.0),
);

// Get face center and fixed dimension from face_id (matches cube.rs tesseract geometry)
fn get_face_center_and_fixed_dim(face_id: u32) -> vec4<f32> {
    switch face_id {
        case 0u: { return vec4<f32>(0.0, 0.0, 0.0, -1.0); } // W = -1
        case 1u: { return vec4<f32>(0.0, 0.0, -1.0, 0.0); } // Z = -1
        case 2u: { return vec4<f32>(0.0, -1.0, 0.0, 0.0); } // Y = -1
        case 3u: { return vec4<f32>(-1.0, 0.0, 0.0, 0.0); } // X = -1
        case 4u: { return vec4<f32>(1.0, 0.0, 0.0, 0.0); }  // X = +1
        case 5u: { return vec4<f32>(0.0, 1.0, 0.0, 0.0); }  // Y = +1
        case 6u: { return vec4<f32>(0.0, 0.0, 1.0, 0.0); }  // Z = +1
        case 7u: { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }  // W = +1
        default: { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
    }
}

// Get fixed dimension from face_id
fn get_fixed_dim(face_id: u32) -> u32 {
    switch face_id {
        case 0u: { return 3u; } // W axis fixed
        case 7u: { return 3u; } // W axis fixed
        case 1u: { return 2u; } // Z axis fixed  
        case 6u: { return 2u; } // Z axis fixed  
        case 2u: { return 1u; } // Y axis fixed
        case 5u: { return 1u; } // Y axis fixed
        case 3u: { return 0u; } // X axis fixed
        case 4u: { return 0u; } // X axis fixed
        default: { return 0u; }
    }
}

// Projects a 4D point to 3D space using perspective projection
fn project_4d_to_3d(point_4d: vec4<f32>, viewer_distance: f32) -> vec3<f32> {
    let w_distance = viewer_distance - point_4d.w;
    let scale = viewer_distance / w_distance;
    return vec3<f32>(point_4d.x * scale, point_4d.y * scale, point_4d.z * scale);
}

// Generate cube vertex in 4D space around sticker center
fn generate_cube_vertex_4d(vertex_index: u32, fixed_dim: u32, center_4d: vec4<f32>, cube_scale: f32) -> vec4<f32> {
    // Get the local cube vertex position (before scaling)
    let local_vertex = CUBE_VERTICES[vertex_index];
    let offset = local_vertex * cube_scale;
    
    // Apply offset to the 3 free axes (not the fixed axis)
    var vertex_4d = center_4d;
    var offset_idx = 0u;
    
    for (var axis = 0u; axis < 4u; axis++) {
        if (axis != fixed_dim) {
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

// Get the 4D transformation matrix for a face
// This matrix transforms from the face's local 3D coordinate system to 4D space
fn get_face_transform_matrix(face_id: u32) -> mat4x4<f32> {
    switch face_id {
        case 0u: { // W = -1, local XYZ -> 4D XYZ, invert Z for inward normal
            return mat4x4<f32>(
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, -1.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            );
        }
        case 1u: { // Z = -1, local XYZ -> 4D XYW, invert W for inward normal
            return mat4x4<f32>(
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, -1.0,
                0.0, 0.0, 0.0, 0.0
            );
        }
        case 2u: { // Y = -1, local XYZ -> 4D XZW, invert Z for inward normal  
            return mat4x4<f32>(
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, -1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0
            );
        }
        case 3u: { // X = -1, local XYZ -> 4D YZW, invert Y for inward normal
            return mat4x4<f32>(
                0.0, -1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0
            );
        }
        case 4u: { // X = +1, local XYZ -> 4D YZW, outward normal
            return mat4x4<f32>(
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0
            );
        }
        case 5u: { // Y = +1, local XYZ -> 4D XZW, outward normal
            return mat4x4<f32>(
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0
            );
        }
        case 6u: { // Z = +1, local XYZ -> 4D XYW, outward normal
            return mat4x4<f32>(
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0
            );
        }
        case 7u: { // W = +1, local XYZ -> 4D XYZ, outward normal
            return mat4x4<f32>(
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            );
        }
        default: {
            return mat4x4<f32>(
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            );
        }
    }
}

// Map a 3D normal to 4D space using proper transformation matrix
fn map_normal_to_4d(normal_3d: vec3<f32>, face_id: u32) -> vec4<f32> {
    let transform = get_face_transform_matrix(face_id);
    let normal_4d_homogeneous = vec4<f32>(normal_3d, 0.0);
    return transform * normal_4d_homogeneous;
}

// Transform a 4D normal vector through 4D rotation
fn transform_normal_4d(normal_4d: vec4<f32>, rotation_matrix: mat4x4<f32>) -> vec4<f32> {
    // Apply full 4D rotation matrix to the 4D normal
    let rotated_normal = rotation_matrix * normal_4d;
    return normalize(rotated_normal);
}


// Test if a 4D face should be visible based on orientation
fn is_face_visible(face_center_4d: vec4<f32>, rotation_matrix: mat4x4<f32>, viewer_distance: f32) -> bool {
    let rotated_face_center = rotation_matrix * face_center_4d;
    
    // Vector from face center to viewer (viewer is at positive W looking toward negative W)
    let viewer_position = vec4<f32>(0.0, 0.0, 0.0, viewer_distance);
    let to_viewer = viewer_position - rotated_face_center;
    
    // Face is visible if it's facing toward the viewer
    let dot_product = dot(rotated_face_center, to_viewer);
    return dot_product < 0.0;
}

// Calculate sticker's 4D center from face center and sticker offset
fn calculate_sticker_center_4d(sticker_position_4d: vec4<f32>, face_center_4d: vec4<f32>, face_spacing: f32) -> vec4<f32> {
    // Get sticker offset from its face center
    let sticker_offset_4d = sticker_position_4d - face_center_4d;
    
    // Scale face centers and keep original sticker positions
    let scaled_face_center = face_center_4d * face_spacing;
    return scaled_face_center + sticker_offset_4d;
}