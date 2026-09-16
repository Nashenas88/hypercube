// Shared 4D mathematics structs and functions for hypercube rendering.
// Imported into each pipeline shader via naga_oil's #import (see renderer.rs).
#define_import_path math4d

// Transform uniform structure shared across shaders
struct Transform4D {
    rotation_matrix: mat4x4<f32>,
    viewer_distance: f32,
    sticker_scale: f32,
    face_gap: f32,
    face_gap_4d: f32,
    _padding: vec3<f32>,
    // Wall-clock seconds since the app started, wrapped modulo 3600, driving
    // fragment-shader animation.
    elapsed_seconds: f32,
}

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_proj_inv: mat4x4<f32>,
    // World-space camera position; `w` is padding.
    eye_position: vec4<f32>,
}

// Instance data for each sticker
struct StickerInstance {
    position_4d: vec4<f32>,
    basis: array<vec4<f32>, 3>,
    face_normal_4d: vec4<f32>,
    kind: u32,
    // How many stickers the piece this facet belongs to has (2..=4) - drives
    // the first-run tutorial's flash-the-target-pieces highlight.
    facet_count: u32,
    _padding: array<u32, 2>,
}

@group(0) @binding(0)
var<uniform> transform: Transform4D;

@group(0) @binding(1)
var<uniform> camera: CameraUniform;

@group(0) @binding(2)
var<storage, read> instances: array<StickerInstance>;

// Inverse of a 3x3 matrix, by cofactor expansion. A singular `m` yields
// infinities rather than an error, so callers whose matrix can degenerate
// have to test for that themselves.
fn inverse3(m: mat3x3<f32>) -> mat3x3<f32> {
    let a = m[0];
    let b = m[1];
    let c = m[2];

    // Rows of the adjugate, which are the cofactors of `m`'s columns.
    let r0 = cross(b, c);
    let r1 = cross(c, a);
    let r2 = cross(a, b);

    let inv_determinant = 1.0 / dot(a, r0);

    return mat3x3<f32>(
        vec3<f32>(r0.x, r1.x, r2.x),
        vec3<f32>(r0.y, r1.y, r2.y),
        vec3<f32>(r0.z, r1.z, r2.z),
    ) * inv_determinant;
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

// World-space frame of one sticker, for effects that sit off its surface
// rather than on it. `edge_x`/`edge_y`/`edge_z` are the projected cube's
// half-edge vectors, one per local mesh axis: unequal in length and no longer
// mutually perpendicular once the 4D perspective divide warps the cube, so
// mapping a direction through them stretches it exactly as the sticker is
// stretched, instead of applying one scale to every axis.
struct StickerAnchor {
    world_center: vec3<f32>,
    edge_x: vec3<f32>,
    edge_y: vec3<f32>,
    edge_z: vec3<f32>,
    visible: bool,
}

// Places a sticker instance's center and its warped local frame in world
// space, for a cube whose local half-extent is `half_extent`. Evaluates the
// same rotate, push and project path as `compute_vertex_geometry`, at the
// sticker's center and along its own basis rather than at a mesh vertex, so
// both stay consistent while a move animation sweeps that basis. `visible` is
// false when the 4D face is culled; the other fields then hold arbitrary but
// valid defaults.
fn compute_sticker_anchor(instance_index: u32, half_extent: f32) -> StickerAnchor {
    var out: StickerAnchor;

    let instance = instances[instance_index];
    let rotated_face_normal = transform.rotation_matrix * instance.face_normal_4d;

    if (!is_face_visible(rotated_face_normal, transform.viewer_distance)) {
        out.world_center = vec3<f32>(0.0, 0.0, 0.0);
        out.edge_x = vec3<f32>(1.0, 0.0, 0.0);
        out.edge_y = vec3<f32>(0.0, 1.0, 0.0);
        out.edge_z = vec3<f32>(0.0, 0.0, 1.0);
        out.visible = false;
        return out;
    }

    let depth_preserving_push = vec4<f32>(rotated_face_normal.xyz, 0.0) * (transform.face_gap_4d - 1.0);
    let rc = transform.rotation_matrix * instance.position_4d + depth_preserving_push;

    let rb0 = transform.rotation_matrix * instance.basis[0];
    let rb1 = transform.rotation_matrix * instance.basis[1];
    let rb2 = transform.rotation_matrix * instance.basis[2];

    // Each edge is a difference of projected points rather than a projected
    // difference, since the 4D perspective divide is not linear - that
    // difference is the whole reason the three edges come out unequal.
    let center_3d = project_4d_to_3d(rc, transform.viewer_distance);
    out.edge_x = project_4d_to_3d(rc + rb0 * half_extent, transform.viewer_distance) - center_3d;
    out.edge_y = project_4d_to_3d(rc + rb1 * half_extent, transform.viewer_distance) - center_3d;
    out.edge_z = project_4d_to_3d(rc + rb2 * half_extent, transform.viewer_distance) - center_3d;

    // The face-gap push shifts the whole sticker equally, so it belongs on
    // the center and cancels out of the edges above.
    out.world_center = center_3d
        + project_4d_to_3d(rotated_face_normal, transform.viewer_distance) * transform.face_gap;
    out.visible = true;

    return out;
}

// Geometry-only placement of one sticker-instance vertex in clip space.
// Excludes per-shader material fields (e.g. `kind`, `piece_slot`).
struct VertexGeometry {
    clip_position: vec4<f32>,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    visible: bool,
}

// Rotates a sticker instance's center and basis vectors, derives the world
// normal, embeds the local cube offset along the rotated basis, then
// projects to 3D and pushes outward along the face's current direction.
// `visible` is false when the 4D face is culled; the other fields then hold
// arbitrary but valid off-screen defaults.
fn compute_vertex_geometry(
    instance_index: u32,
    vertex_position: vec3<f32>,
    vertex_index: u32,
) -> VertexGeometry {
    var out: VertexGeometry;

    let instance = instances[instance_index];
    let sticker_center_4d = instance.position_4d;

    // Rotate the face normal once; it's reused for both the visibility
    // test and the face-gap push below, instead of rotating it twice.
    let rotated_face_normal = transform.rotation_matrix * instance.face_normal_4d;

    let face_visible = is_face_visible(rotated_face_normal, transform.viewer_distance);
    if (!face_visible) {
        out.clip_position = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        out.world_position = vec3<f32>(0.0, 0.0, 0.0);
        out.world_normal = vec3<f32>(0.0, 0.0, 1.0);
        out.visible = false;
        return out;
    }

    let local_vertex = vertex_position * transform.sticker_scale;

    // Which of the 6 local cube faces this vertex belongs to (0-5)
    let face_3d = vertex_index / 6u;

    // Rotate the sticker center and basis vectors once; by linearity
    // R*(center + basis[i]) == R*center + R*basis[i], so these same four
    // rotated vectors serve both the normal and the vertex position below,
    // instead of each being re-rotated separately.
    //
    // `depth_preserving_push` is the rotated face normal with its
    // w-component discarded: adding it shifts x/y/z without changing the w
    // the perspective divide below uses, so a sticker's apparent size never
    // depends on the push magnitude.
    let depth_preserving_push = vec4<f32>(rotated_face_normal.xyz, 0.0) * (transform.face_gap_4d - 1.0);
    let rc = transform.rotation_matrix * sticker_center_4d + depth_preserving_push;
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

    out.clip_position = camera.view_proj * vec4<f32>(vertex_3d, 1.0);
    out.world_position = vertex_3d;
    out.world_normal = world_normal;
    out.visible = true;

    return out;
}
