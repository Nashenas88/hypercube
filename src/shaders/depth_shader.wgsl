// Depth visualization shader using instanced rendering
// Displays depth values as grayscale colors for debugging
#import math4d::{Transform4D, CameraUniform, StickerInstance, project_4d_to_3d, is_face_visible}

@group(0) @binding(0)
var<uniform> transform: Transform4D;

@group(0) @binding(1)
var<uniform> camera: CameraUniform;

@group(0) @binding(2)
var<storage, read> instances: array<StickerInstance>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) depth: f32,
}

@vertex
fn vs_main(
    @location(0) vertex_position: vec3<f32>,
    @builtin(instance_index) instance_index: u32,
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
        out.depth = 0.0;
        return out;
    }

    // Get cube vertex from vertex attribute
    let local_vertex = vertex_position * transform.sticker_scale;

    // Rotate the sticker center and basis vectors once; by linearity
    // R*(center + basis[i]) == R*center + R*basis[i].
    let rc = transform.rotation_matrix * sticker_center_4d;
    let rb0 = transform.rotation_matrix * instance.basis[0];
    let rb1 = transform.rotation_matrix * instance.basis[1];
    let rb2 = transform.rotation_matrix * instance.basis[2];

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