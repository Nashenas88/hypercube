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

// Test if a 4D face should be visible based on orientation. `rotated_face_normal`
// is the face normal already rotated into world space.
fn is_face_visible(rotated_face_normal: vec4<f32>, viewer_distance: f32) -> bool {
    let viewer_position = vec4<f32>(0.0, 0.0, 0.0, viewer_distance);
    let to_viewer = viewer_position - rotated_face_normal;
    let dot_product = dot(rotated_face_normal, to_viewer);
    return dot_product < 0.0;
}

// Derives the world-space outward normal for local cube face `face_3d`
// (0..5) directly from the instance's own basis vectors, the same data used
// to place the vertex itself - so the normal always matches the mesh, both
// static and mid-rotation. `rc`/`rb0`/`rb1`/`rb2` are the sticker center and
// the three basis vectors, already rotated into world space. The switch
// selects `vi`/`vj`/`vk` directly among the three hoisted `rb*` locals per
// face rather than indexing `array<vec4<f32>, 3>` with a runtime index,
// which forces the array into scratch memory on most drivers; `vk`'s sign
// picks which of the two cross-product directions is outward for that
// face. `pi`/`pj`/`pk` are those same points projected to 3D, mirroring the
// cross-product-of-projected-edges technique.
fn compute_world_normal(
    rc: vec4<f32>,
    rb0: vec4<f32>,
    rb1: vec4<f32>,
    rb2: vec4<f32>,
    face_3d: u32,
    viewer_distance: f32,
) -> vec3<f32> {
    var vi: vec4<f32>;
    var vj: vec4<f32>;
    var vk: vec4<f32>;
    switch (face_3d) {
        case 0u: {
            vi = rc + rb0;
            vj = rc + rb1;
            vk = rc - rb2;
        }
        case 1u: {
            vi = rc + rb1;
            vj = rc + rb2;
            vk = rc + rb0;
        }
        case 2u: {
            vi = rc + rb0;
            vj = rc + rb1;
            vk = rc + rb2;
        }
        case 3u: {
            vi = rc + rb1;
            vj = rc + rb2;
            vk = rc - rb0;
        }
        case 4u: {
            vi = rc + rb0;
            vj = rc + rb2;
            vk = rc + rb1;
        }
        default: {
            vi = rc + rb0;
            vj = rc + rb2;
            vk = rc - rb1;
        }
    }

    let p0 = project_4d_to_3d(rc, viewer_distance);
    let pi = project_4d_to_3d(vi, viewer_distance);
    let pj = project_4d_to_3d(vj, viewer_distance);
    let pk = project_4d_to_3d(vk, viewer_distance);

    var n = normalize(cross(pi - p0, pj - p0));
    if (dot(n, pk - p0) < 0.0) {
        n = -n;
    }
    return n;
}
