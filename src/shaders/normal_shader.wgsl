// Normal visualization shader using instanced rendering
// Displays normal vectors as colors for debugging
#import math4d::{Transform4D, CameraUniform, StickerInstance, project_4d_to_3d, is_face_visible, compute_world_normal}

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

    // Rotate the face normal once; it's reused for both the visibility
    // test and the face-gap push below, instead of rotating it twice.
    let rotated_face_normal = transform.rotation_matrix * instance.face_normal_4d;

    // Check if this face is visible (4D culling)
    let face_visible = is_face_visible(rotated_face_normal, transform.viewer_distance);

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

    // Rotate the sticker center and basis vectors once; by linearity
    // R*(center + basis[i]) == R*center + R*basis[i], so these same four
    // rotated vectors serve both the normal and the vertex position below,
    // instead of each being re-rotated separately.
    let rc = transform.rotation_matrix * sticker_center_4d;
    let rb0 = transform.rotation_matrix * instance.basis[0];
    let rb1 = transform.rotation_matrix * instance.basis[1];
    let rb2 = transform.rotation_matrix * instance.basis[2];

    // Derive the normal from the instance's own basis, so it always matches
    // this instance's actual (possibly mid-rotation) orientation.
    let world_normal = compute_world_normal(
        rc,
        rb0,
        rb1,
        rb2,
        face_3d,
        transform.viewer_distance,
    );

    // Generate the vertex in 4D space by embedding the local cube offset
    // along the instance's own (possibly mid-rotation) rotated basis
    // vectors, rather than a fixed world axis.
    let rotated_vertex_4d = rc
        + local_vertex.x * rb0
        + local_vertex.y * rb1
        + local_vertex.z * rb2;

    // Project to 3D, then push outward along the face's own current
    // direction by a constant 3D distance. Deliberately not normalized to a
    // fixed length: the face normal is always a 4D unit vector, so this
    // projected vector's own length already shrinks smoothly toward zero
    // exactly when a face's piece renders near the center of the screen -
    // normalizing would force that near-zero (direction-unstable) vector
    // back up to full length, snapping the piece to a full-size
    // displacement in a swinging direction instead of tapering out
    // smoothly.
    let push = project_4d_to_3d(rotated_face_normal, transform.viewer_distance) * transform.face_gap;
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