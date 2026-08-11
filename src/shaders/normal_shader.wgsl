// Normal visualization shader using instanced rendering
// Displays normal vectors as colors for debugging
#import math4d::{Transform4D, CameraUniform, StickerInstance, project_4d_to_3d, is_face_visible, compute_world_normal, face_push_offset_3d}

@group(0) @binding(0)
var<uniform> transform: Transform4D;

@group(0) @binding(1)
var<uniform> camera: CameraUniform;

@group(0) @binding(2)
var<storage, read> instances: array<StickerInstance>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
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
    // direction by a constant 3D distance - see face_push_offset_3d.
    let push = face_push_offset_3d(instance.face_normal_4d, transform.rotation_matrix, transform.viewer_distance) * transform.face_gap;
    let vertex_3d = project_4d_to_3d(rotated_vertex_4d, transform.viewer_distance) + push;

    // Apply 3D view/projection matrix
    out.clip_position = camera.view_proj * vec4<f32>(vertex_3d, 1.0);

    // Convert normal vector to color (normalize to 0-1 range)
    out.color = vec4<f32>(world_normal * 0.5 + 0.5, 1.0);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}