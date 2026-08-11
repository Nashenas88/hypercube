// Shared 4D mathematics structs and functions for hypercube rendering.
// Imported into each pipeline shader via naga_oil's #import (see renderer.rs).
#define_import_path math4d

// Transform uniform structure shared across shaders
struct Transform4D {
    rotation_matrix: mat4x4<f32>,
    viewer_distance: f32,
    sticker_scale: f32,
    face_gap: f32,
    _padding: f32,
}

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_proj_inv: mat4x4<f32>,
}

// Instance data for each sticker
struct StickerInstance {
    position_4d: vec4<f32>,
    color: vec4<f32>,
    basis: array<vec4<f32>, 3>,
    face_normal_4d: vec4<f32>,
}

// Projects a 4D point to 3D space using perspective projection
fn project_4d_to_3d(point_4d: vec4<f32>, viewer_distance: f32) -> vec3<f32> {
    let w_distance = viewer_distance - point_4d.w;
    let scale = viewer_distance / w_distance;
    return vec3<f32>(point_4d.x * scale, point_4d.y * scale, point_4d.z * scale);
}

// Test if a 4D face should be visible based on orientation
fn is_face_visible(face_center_4d: vec4<f32>, rotation_matrix: mat4x4<f32>, viewer_distance: f32) -> bool {
    let rotated_face_center = rotation_matrix * face_center_4d;
    let viewer_position = vec4<f32>(0.0, 0.0, 0.0, viewer_distance);
    let to_viewer = viewer_position - rotated_face_center;
    let dot_product = dot(rotated_face_center, to_viewer);
    return dot_product < 0.0;
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
