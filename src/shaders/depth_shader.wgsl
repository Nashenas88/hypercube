// Depth visualization shader using instanced rendering
// Displays depth values as grayscale colors for debugging

// Transform uniform structure
struct Transform4D {
    rotation_matrix: mat4x4<f32>,
    viewer_distance: f32,
    sticker_scale: f32,
    face_gap: f32,
    _padding: f32,
}

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

// Instance data for each sticker
struct StickerInstance {
    position_4d: vec4<f32>,
    color: vec4<f32>,
    basis: array<vec4<f32>, 3>,
    face_normal_4d: vec4<f32>,
}

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

// Shared math4d functions


fn project_4d_to_3d(point_4d: vec4<f32>, viewer_distance: f32) -> vec3<f32> {
    let w_distance = viewer_distance - point_4d.w;
    let scale = viewer_distance / w_distance;
    return vec3<f32>(point_4d.x * scale, point_4d.y * scale, point_4d.z * scale);
}

fn is_face_visible(face_center_4d: vec4<f32>, rotation_matrix: mat4x4<f32>, viewer_distance: f32) -> bool {
    let rotated_face_center = rotation_matrix * face_center_4d;
    let viewer_position = vec4<f32>(0.0, 0.0, 0.0, viewer_distance);
    let to_viewer = viewer_position - rotated_face_center;
    let dot_product = dot(rotated_face_center, to_viewer);
    return dot_product < 0.0;
}

// The outward push for a face, projected into 3D: `face_normal_4d` rotated
// and projected like any other point. Pushing already-projected geometry
// by this offset (rather than scaling a 4D anchor before projection) can
// never cross the perspective divide's `viewer_distance - w = 0`
// singularity. Deliberately not normalized to a fixed length:
// `face_normal_4d` is always a 4D unit vector, so this projected vector's
// own length already shrinks smoothly toward zero exactly when a face's
// piece renders near the center of the screen - normalizing would force
// that near-zero (direction-unstable) vector back up to full length,
// snapping the piece to a full-size displacement in a swinging direction
// instead of tapering out smoothly.
fn face_push_offset_3d(face_normal_4d: vec4<f32>, rotation_matrix: mat4x4<f32>, viewer_distance: f32) -> vec3<f32> {
    return project_4d_to_3d(rotation_matrix * face_normal_4d, viewer_distance);
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

    // Check if this face is visible (4D culling)
    let face_visible = is_face_visible(instance.face_normal_4d, transform.rotation_matrix, transform.viewer_distance);
    
    if (!face_visible) {
        // Face is culled - move vertex off-screen
        out.clip_position = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        out.depth = 0.0;
        return out;
    }
    
    // Get cube vertex from vertex attribute
    let local_vertex = vertex_position * transform.sticker_scale;
    
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