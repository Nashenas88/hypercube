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
    view_proj_inv: mat4x4<f32>,
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
}

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
    basis: array<vec4<f32>, 3>,
    face_id: u32,
    _padding: array<u32, 3>,
    face_normal_4d: vec4<f32>,
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
var<storage, read> instances: array<StickerInstance>;

@group(0) @binding(5)
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

 

// Derives the world-space outward normal for local cube face `face_3d`
// (0..5) directly from the instance's own basis vectors, the same data used
// to place the vertex itself - so the normal always matches the mesh, both
// static and mid-rotation. `k` is the face's normal axis in `basis`, `s` its
// local sign, and `i`/`j` its two tangent axes; both are derived once here
// via projected finite offsets along `basis`, mirroring the cross-product-
// of-projected-edges technique, and the k/s offset is used purely to decide
// which of the two cross-product directions is outward.
fn compute_world_normal(
    sticker_center_4d: vec4<f32>,
    basis: array<vec4<f32>, 3>,
    face_3d: u32,
    rotation_matrix: mat4x4<f32>,
    viewer_distance: f32,
) -> vec3<f32> {
    var i: u32;
    var j: u32;
    var k: u32;
    var s: f32;
    switch (face_3d) {
        case 0u: { k = 2u; s = -1.0; i = 0u; j = 1u; }
        case 1u: { k = 0u; s = 1.0; i = 1u; j = 2u; }
        case 2u: { k = 2u; s = 1.0; i = 0u; j = 1u; }
        case 3u: { k = 0u; s = -1.0; i = 1u; j = 2u; }
        case 4u: { k = 1u; s = 1.0; i = 0u; j = 2u; }
        default: { k = 1u; s = -1.0; i = 0u; j = 2u; }
    }

    let p0 = project_4d_to_3d(rotation_matrix * sticker_center_4d, viewer_distance);
    let pi = project_4d_to_3d(rotation_matrix * (sticker_center_4d + basis[i]), viewer_distance);
    let pj = project_4d_to_3d(rotation_matrix * (sticker_center_4d + basis[j]), viewer_distance);
    let pk = project_4d_to_3d(rotation_matrix * (sticker_center_4d + s * basis[k]), viewer_distance);

    var n = normalize(cross(pi - p0, pj - p0));
    if (dot(n, pk - p0) < 0.0) {
        n = -n;
    }
    return n;
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

    // Calculate sticker center in 4D
    let sticker_center_4d = calculate_sticker_center_4d(instance.position_4d, face_center_4d, transform.face_spacing);
    
    // Check if this face is visible (4D culling)
    let face_visible = is_face_visible(instance.face_normal_4d, transform.rotation_matrix, transform.viewer_distance);
    
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
    
    // Which of the 6 local cube faces this vertex belongs to (0-5)
    let face_3d = vertex_index / 6u;

    // Derive the normal from the instance's own basis, so it always matches
    // this instance's actual (possibly mid-rotation) orientation.
    let world_normal = compute_world_normal(
        sticker_center_4d,
        instance.basis,
        face_3d,
        transform.rotation_matrix,
        transform.viewer_distance,
    );

    var corrected_vertex = local_vertex;
    let face_id = instance.face_id;

    // The axis permutations for faces 1, 3, 4, 6 result in a reflection
    // (a change of handedness). We must flip one axis of the local vertex
    // to counteract this and preserve the correct winding order.
    // if (face_id == 0u || face_id == 1u || face_id == 3u || face_id == 4u || face_id == 6u) {
    //     corrected_vertex.x = -corrected_vertex.x;
    // }
    
    // Generate vertex in 4D space around sticker center by embedding the
    // local cube offset along the instance's own (possibly mid-rotation)
    // basis vectors, rather than a fixed world axis.
    var vertex_4d = sticker_center_4d
        + corrected_vertex.x * instance.basis[0]
        + corrected_vertex.y * instance.basis[1]
        + corrected_vertex.z * instance.basis[2];

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

    // Convert screen position back to world direction for cubemap sampling using
    // the translation-free inverse view-projection matrix. Leaving the result
    // un-normalized keeps it affine in (x, y), so linear interpolation across the
    // quad's four corners lands on the exact per-pixel direction.
    let world_pos = sky_camera.view_proj_inv * vec4<f32>(x, y, 1.0, 1.0);
    out.world_position = world_pos.xyz / world_pos.w;

    return out;
}

// Skybox fragment shader
@fragment
fn fs_sky(in: SkyboxVertexOutput) -> @location(0) vec4<f32> {
    return textureSample(sky_texture, sky_sampler, normalize(in.world_position));
}