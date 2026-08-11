// Vertex shader using instanced rendering with static cube geometry
#import math4d::{Transform4D, CameraUniform, StickerInstance, project_4d_to_3d, is_face_visible, compute_world_normal, face_push_offset_3d}

struct LightUniform {
    direction: vec3<f32>,
    _padding1: f32,
    color: vec3<f32>,
    _padding2: f32,
    ambient: vec3<f32>,
    _padding3: f32,
};

struct HighlightingUniform {
    hovered_sticker_index: u32,
    highlight_intensity: f32,
    _padding1: vec2<f32>,
    highlight_color: vec3<f32>,
    _padding2: f32,
};

@group(0) @binding(0)
var<uniform> transform: Transform4D;

@group(0) @binding(1)
var<uniform> camera: CameraUniform;

@group(0) @binding(2)
var<uniform> light: LightUniform;

@group(0) @binding(3)
var<storage, read> instances: array<StickerInstance>;

@group(0) @binding(4)
var<uniform> highlighting: HighlightingUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) instance_index: u32,
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
    let sticker_center_4d = instance.position_4d;

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

    // Generate vertex in 4D space around sticker center by embedding the
    // local cube offset along the instance's own (possibly mid-rotation)
    // basis vectors, rather than a fixed world axis.
    var vertex_4d = sticker_center_4d
        + local_vertex.x * instance.basis[0]
        + local_vertex.y * instance.basis[1]
        + local_vertex.z * instance.basis[2];

    // Apply 4D rotation
    let rotated_vertex_4d = transform.rotation_matrix * vertex_4d;

    // Project to 3D, then push outward along the face's own current
    // direction by a constant 3D distance - keeps the sticker rigid (no
    // warp) at any gap size, since it's a uniform per-instance offset
    // applied after the perspective divide rather than before it.
    let push = face_push_offset_3d(instance.face_normal_4d, transform.rotation_matrix, transform.viewer_distance) * transform.face_gap;
    let vertex_3d = project_4d_to_3d(rotated_vertex_4d, transform.viewer_distance) + push;

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