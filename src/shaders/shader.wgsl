// Vertex shader using instanced rendering with static cube geometry
// Imports shared 4D math functions

// Import shared 4D math functions
// Note: WGSL doesn't have a standard import mechanism, so we'll include the content directly
// TODO: Replace with proper import when WGSL supports it

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

struct LightUniform {
    direction: vec3<f32>,
    _padding1: f32,
    color: vec3<f32>,
    _padding2: f32,
    ambient: vec3<f32>,
    _padding3: f32,
};

struct FaceDataUniform {
    face_centers: array<vec4<f32>, 8>,
    fixed_dims: array<vec4<u32>, 8>,
}

struct NormalsUniform {
    normals: array<vec3<f32>, 48>,  // 8 faces × 6 normals each
};

struct HighlightingUniform {
    hovered_sticker_index: u32,
    highlight_intensity: f32,
    _padding1: vec2<f32>,
    highlight_color: vec3<f32>,
    _padding2: f32,
};

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
var<uniform> light: LightUniform;

@group(0) @binding(3)
var<uniform> face_data: FaceDataUniform;

@group(0) @binding(4)
var<uniform> normals: NormalsUniform;

@group(0) @binding(5)
var<storage, read> instances: array<StickerInstance>;

@group(0) @binding(6)
var<uniform> highlighting: HighlightingUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) instance_index: u32,
}

// Include math4d functions directly (until WGSL supports imports)
// Static cube geometry: 36 vertices (6 faces × 6 vertices per face)

// Math4D functions

fn project_4d_to_3d(point_4d: vec4<f32>, viewer_distance: f32) -> vec3<f32> {
    let w_distance = viewer_distance - point_4d.w;
    let scale = viewer_distance / w_distance;
    return vec3<f32>(point_4d.x * scale, point_4d.y * scale, point_4d.z * scale);
}

 

fn transform_normal_4d(normal_4d: vec4<f32>, rotation_matrix: mat4x4<f32>) -> vec4<f32> {
    // Apply full 4D rotation matrix to the 4D normal
    let rotated_normal = rotation_matrix * normal_4d;
    return normalize(rotated_normal);
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
    @builtin(vertex_index) vertex_index: u32,
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
        out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.world_position = vec3<f32>(0.0, 0.0, 0.0);
        out.world_normal = vec3<f32>(0.0, 0.0, 1.0);
        return out;
    }
    
    // Get cube vertex from vertex attributes
    let local_vertex = vertex_position * transform.sticker_scale;
    
    // Calculate normal index: which cube face this vertex belongs to (0-5)
    let face_3d = vertex_index / 6u;
    
    // Get the pre-calculated normal for this face and vertex
    let global_normal_index = instance.face_id * 6u + face_3d;
    let world_normal = normals.normals[global_normal_index];

    var corrected_vertex = local_vertex;
    let face_id = instance.face_id;

    // The axis permutations for faces 1, 3, 4, 6 result in a reflection
    // (a change of handedness). We must flip one axis of the local vertex
    // to counteract this and preserve the correct winding order.
    // if (face_id == 0u || face_id == 1u || face_id == 3u || face_id == 4u || face_id == 6u) {
    //     corrected_vertex.x = -corrected_vertex.x;
    // }
    
    // Generate vertex in 4D space around sticker center
    var vertex_4d = sticker_center_4d;
    var offset_idx = 0u;
    
    for (var axis = 0u; axis < 4u; axis++) {
        if (axis != fixed_dim) {
            if (offset_idx == 0u) {
                vertex_4d[axis] += corrected_vertex.x;
            } else if (offset_idx == 1u) {
                vertex_4d[axis] += corrected_vertex.y;
            } else if (offset_idx == 2u) {
                vertex_4d[axis] += corrected_vertex.z;
            }
            offset_idx++;
        }
    }
    
    // Apply 4D rotation
    let rotated_vertex_4d = transform.rotation_matrix * vertex_4d;
    
    // Project to 3D
    let vertex_3d = project_4d_to_3d(rotated_vertex_4d, transform.viewer_distance);
    
    // Apply 3D view/projection matrix
    out.clip_position = camera.view_proj * vec4<f32>(vertex_3d, 1.0);
    out.color = instance.color;
    out.world_position = vertex_3d;
    out.world_normal = world_normal;
    out.instance_index = instance_index;
    
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Normalize the normal vector
    let normal = normalize(in.world_normal);
    
    // Calculate light direction (directional light)
    let light_dir = normalize(-light.direction);
    
    // Calculate view direction (camera position is at origin in view space)
    let view_dir = normalize(-in.world_position);
    
    // Ambient lighting
    let ambient = light.ambient * in.color.rgb;
    
    // Diffuse lighting (Lambertian)
    let diffuse_strength = max(dot(normal, light_dir), 0.0);
    let diffuse = diffuse_strength * light.color * in.color.rgb;
    
    // Specular lighting (Blinn-Phong)
    let half_dir = normalize(light_dir + view_dir);
    let specular_strength = pow(max(dot(normal, half_dir), 0.0), 32.0);
    let specular = specular_strength * light.color * 0.3; // Reduced specular intensity
    
    // Combine all lighting components
    var final_color = ambient + diffuse + specular;
    
    // Apply highlighting if this sticker is hovered
    if (in.instance_index == highlighting.hovered_sticker_index) {
        // Mix the final color with the highlight color
        final_color = mix(final_color, highlighting.highlight_color, highlighting.highlight_intensity);
    }
    
    return vec4<f32>(final_color, in.color.a);
}

// Skybox shaders
@group(0) @binding(0)
var<uniform> sky_camera: CameraUniform;

@group(0) @binding(1)
var sky_texture: texture_cube<f32>;

@group(0) @binding(2)
var sky_sampler: sampler;

struct SkyboxVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
}

// Skybox vertex shader
@vertex
fn vs_sky(@location(0) position: vec2<f32>) -> SkyboxVertexOutput {
    var out: SkyboxVertexOutput;
    
    // Use the vertex position from the vertex buffer
    let x = position.x;
    let y = position.y;
    
    out.clip_position = vec4<f32>(x, y, 1.0, 1.0);
    
    // Convert screen position back to world direction for cubemap sampling
    // Inverse of view-projection matrix to get world space direction
    let inverse_view_proj = transpose(sky_camera.view_proj);
    let world_pos = inverse_view_proj * vec4<f32>(x, y, 1.0, 1.0);
    out.world_position = normalize(world_pos.xyz / world_pos.w);
    
    return out;
}

// Skybox fragment shader
@fragment
fn fs_sky(in: SkyboxVertexOutput) -> @location(0) vec4<f32> {
    return textureSample(sky_texture, sky_sampler, in.world_position);
}