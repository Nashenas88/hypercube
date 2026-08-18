//! 4D mathematics utilities for hypercube visualization.
//!
//! This module provides CPU-side 4D rotation calculations, 4D-to-3D projections,
//! and shared transformation logic to eliminate code duplication.

use nalgebra::{Matrix3, Matrix4, Point3, Quaternion, Rotation3, UnitQuaternion, Vector3, Vector4};

use crate::geometry::{BASE_CUBE_VERTICES, FACE_CENTERS, FIXED_DIMS};

/// Mouse sensitivity for 4D rotation controls
const MOUSE_SENSITIVITY: f32 = 0.5;

/// 4D viewer distance for perspective projection
pub(crate) const VIEWER_DISTANCE: f32 = 3.0;

/// The maximum size of a cube dimension that the sticker can occupy
pub(crate) const BASE_STICKER_SIZE: f32 = 1.0 / 3.0;

/// Half-width of the 3x3x3 sticker grid positioning pattern
/// Stickers are positioned at coordinates {-2/3, 0, +2/3} on free axes
pub(crate) const GRID_EXTENT: f32 = 2.0 / 3.0;

/// Creates a 4D rotation matrix around the XW plane.
///
/// This rotation affects the X and W coordinates while leaving Y and Z unchanged.
/// In 4D space, there are 6 possible rotation planes; this is one of them.
/// Only used as the hand-written reference `plane_rotation_matches_xw_special_case`
/// checks the general `create_4d_plane_rotation` formula against.
///
/// # Arguments
/// * `angle` - Rotation angle in radians
///
/// # Returns
/// A 4x4 rotation matrix for the XW plane
#[cfg(test)]
fn create_4d_rotation_xw(angle: f32) -> Matrix4<f32> {
    let cos_x = angle.cos();
    let sin_x = angle.sin();
    Matrix4::new(
        cos_x, 0.0, 0.0, -sin_x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, sin_x, 0.0, 0.0, cos_x,
    )
}

/// Processes mouse input to create incremental 4D rotation.
///
/// Converts mouse movement into 4D rotation by rotating the planes spanned by
/// the camera's current right/up directions (lifted into 4D with `w = 0`) and
/// the W axis, so a drag rotates whatever the camera currently sees into or
/// out of the hidden 4th dimension - rather than always rotating the same
/// fixed world-space planes regardless of the camera's current orientation.
/// The rotations are applied incrementally to the existing rotation matrix.
///
/// # Arguments
/// * `current_rotation` - The current 4D rotation matrix
/// * `delta_x` - Horizontal mouse movement delta
/// * `delta_y` - Vertical mouse movement delta
/// * `camera_right` - World-space right direction of the current camera view
/// * `camera_up` - World-space up direction of the current camera view
///
/// # Returns
/// Updated 4D rotation matrix incorporating the mouse movement
pub(crate) fn process_4d_rotation(
    current_rotation: &Matrix4<f32>,
    delta_x: f32,
    delta_y: f32,
    camera_right: Vector3<f32>,
    camera_up: Vector3<f32>,
) -> Matrix4<f32> {
    // Screen Y grows downward, so "drag up" is a negative delta_y that must
    // flip sign to read as a positive rotation; screen X already grows
    // rightward in the same sense as `camera_right`, so it doesn't.
    let angle_x = delta_x * MOUSE_SENSITIVITY * 0.01;
    let angle_y = -delta_y * MOUSE_SENSITIVITY * 0.01;

    let w_axis = Vector4::new(0.0, 0.0, 0.0, 1.0);
    let right_4d = Vector4::new(camera_right.x, camera_right.y, camera_right.z, 0.0);
    let up_4d = Vector4::new(camera_up.x, camera_up.y, camera_up.z, 0.0);

    let rotation_h = create_4d_plane_rotation(right_4d, w_axis, angle_x);
    let rotation_v = create_4d_plane_rotation(up_4d, w_axis, angle_y);

    rotation_v * rotation_h * current_rotation
}

/// Transform a 4D position to 3D world space using perspective projection.
///
/// This is the core transformation used throughout the application for
/// projecting 4D coordinates to visible 3D space. Replaces duplicate logic
/// in ray_casting.rs and shader_widget.rs.
///
/// # Arguments
/// * `position_4d` - 4D position to transform
/// * `rotation_4d` - 4D rotation matrix
/// * `viewer_distance` - Distance of 4D viewer from W=0 plane
///
/// # Returns
/// Projected 3D position
pub(crate) fn project_4d_to_3d(
    position_4d: Vector4<f32>,
    rotation_4d: &Matrix4<f32>,
    viewer_distance: f32,
) -> Point3<f32> {
    // Apply 4D rotation
    let rotated_4d = rotation_4d * position_4d;

    // Project to 3D using perspective projection
    let w_distance = viewer_distance - rotated_4d.w;
    let scale = viewer_distance / w_distance;

    Point3::new(
        rotated_4d.x * scale,
        rotated_4d.y * scale,
        rotated_4d.z * scale,
    )
}

/// The outward push for a face, projected into 3D: the same
/// `face_normal_4d` used for 4D culling, rotated and projected through
/// `project_4d_to_3d` like any other point. Used to push already-projected
/// sticker geometry outward by a 3D offset instead of scaling a 4D anchor
/// before projection (which drives the perspective divide toward its
/// `viewer_distance - w = 0` singularity as the push grows).
///
/// Deliberately *not* normalized to a fixed length: `face_normal_4d` is
/// always a 4D unit vector, so this projected vector's own length already
/// shrinks smoothly toward zero exactly when a face's outward direction is
/// nearly aligned with the 4D depth axis - which is also exactly when that
/// face's piece renders near the center of the screen. Normalizing here
/// would force a near-zero (and therefore direction-unstable) vector back
/// up to full length, snapping the piece to a full-size displacement in a
/// swinging direction as the puzzle rotates through that zone; using the
/// natural length instead lets the push taper out smoothly there.
pub(crate) fn face_push_offset_3d(
    face_normal_4d: Vector4<f32>,
    rotation_4d: &Matrix4<f32>,
    viewer_distance: f32,
) -> Vector3<f32> {
    let projected = project_4d_to_3d(face_normal_4d, rotation_4d, viewer_distance);
    Vector3::new(projected.x, projected.y, projected.z)
}

/// Transform all vertices of a sticker cube to 3D space.
///
/// Replaces the duplicate vertex transformation logic in both
/// ray_casting.rs and shader_widget.rs.
///
/// # Arguments
/// * `sticker_position_4d` - 4D position of the sticker (nominal, unpushed)
/// * `face_id` - Face ID (0-7) to determine face center and fixed dimension
/// * `rotation_4d` - 4D rotation matrix
/// * `sticker_scale` - Scale factor for individual stickers
/// * `gap_distance` - 3D distance to push the sticker outward along its
///   face's outward direction, applied after projection
/// * `viewer_distance` - Distance of 4D viewer from W=0 plane
///
/// # Returns
/// Vector of 36 transformed 3D vertices (one complete cube)
pub(crate) fn transform_sticker_vertices_to_3d(
    sticker_position_4d: Vector4<f32>,
    face_id: usize,
    rotation_4d: &Matrix4<f32>,
    sticker_scale: f32,
    gap_distance: f32,
    viewer_distance: f32,
) -> Vec<Point3<f32>> {
    let fixed_dim = FIXED_DIMS[face_id];
    let push =
        face_push_offset_3d(FACE_CENTERS[face_id], rotation_4d, viewer_distance) * gap_distance;

    // Transform each cube vertex exactly like the shader does
    let mut world_vertices = Vec::with_capacity(36);
    for vertex in &BASE_CUBE_VERTICES {
        let local_vertex =
            Vector3::new(vertex[0], vertex[1], vertex[2]) * BASE_STICKER_SIZE * sticker_scale;
        let projected = project_cube_point(
            local_vertex,
            sticker_position_4d,
            fixed_dim,
            rotation_4d,
            viewer_distance,
        );
        world_vertices.push(projected + push);
    }

    world_vertices
}

pub(crate) fn project_cube_point(
    local_vertex: Vector3<f32>,
    center_vertex: Vector4<f32>,
    fixed_dim: usize,
    rotation_4d: &Matrix4<f32>,
    viewer_distance: f32,
) -> Point3<f32> {
    // Generate vertex in 4D space around sticker center (matching shader logic)
    let mut vertex_4d = center_vertex;
    let mut offset_idx = 0;

    for axis in 0..4 {
        if axis != fixed_dim {
            match offset_idx {
                0 => vertex_4d[axis] += local_vertex.x,
                1 => vertex_4d[axis] += local_vertex.y,
                2 => vertex_4d[axis] += local_vertex.z,
                _ => {}
            }
            offset_idx += 1;
        }
    }

    project_4d_to_3d(vertex_4d, rotation_4d, viewer_distance)
}

/// Rotates within the plane spanned by orthonormal `u` and `v` by `angle`,
/// leaving their orthogonal complement fixed. `create_4d_rotation_xw` is the
/// special case where `u`, `v` are standard basis vectors.
pub(crate) fn create_4d_plane_rotation(
    u: Vector4<f32>,
    v: Vector4<f32>,
    angle: f32,
) -> Matrix4<f32> {
    let (sin, cos) = angle.sin_cos();
    let outer_uu = u * u.transpose();
    let outer_vv = v * v.transpose();
    let outer_vu = v * u.transpose();
    let outer_uv = u * v.transpose();
    Matrix4::identity() + (outer_uu + outer_vv) * (cos - 1.0) + (outer_vu - outer_uv) * sin
}

/// Finds the orthonormal plane and signed angle such that rotating `from` by
/// `create_4d_plane_rotation(from, v, angle)` carries it onto unit vector
/// `to` (shortest arc). Falls back to an arbitrary orthogonal plane when
/// `from`/`to` are anti-parallel, since no unique plane exists in that case.
pub(crate) fn shortest_arc_plane(
    from: Vector4<f32>,
    to: Vector4<f32>,
) -> (Vector4<f32>, Vector4<f32>, f32) {
    let dot = from.dot(&to).clamp(-1.0, 1.0);
    if dot > 0.999_999 {
        return (from, to, 0.0);
    }
    let v = if dot < -0.999_999 {
        let fallback = if from.x.abs() < 0.9 {
            Vector4::new(1.0, 0.0, 0.0, 0.0)
        } else {
            Vector4::new(0.0, 1.0, 0.0, 0.0)
        };
        (fallback - from * from.dot(&fallback)).normalize()
    } else {
        (to - from * dot).normalize()
    };
    (from, v, dot.acos())
}

/// Wraps a `Vector4<f32>` interpreted in `nalgebra`'s native quaternion
/// coordinate order, `(i, j, k, w)`, as a unit quaternion. Only valid when
/// `coords` is already known to be unit length, e.g. a column of an
/// orthogonal matrix.
fn quat_from_unit_coords(coords: Vector4<f32>) -> UnitQuaternion<f32> {
    UnitQuaternion::new_unchecked(Quaternion::from(coords))
}

fn quat_one() -> Vector4<f32> {
    Vector4::new(0.0, 0.0, 0.0, 1.0)
}
fn quat_i() -> Vector4<f32> {
    Vector4::new(1.0, 0.0, 0.0, 0.0)
}
fn quat_j() -> Vector4<f32> {
    Vector4::new(0.0, 1.0, 0.0, 0.0)
}
fn quat_k() -> Vector4<f32> {
    Vector4::new(0.0, 0.0, 1.0, 0.0)
}

/// Spherical interpolation from `a` to `b` that lands exactly on `b` at
/// `t = 1`, even when that means going "the long way" around.
///
/// This deliberately does *not* use `UnitQuaternion::slerp`: that method
/// treats a unit quaternion as a stand-in for a 3D rotation, where `q` and
/// `-q` represent the identical rotation, so it silently swaps in `-b` for
/// `b` whenever that's less than 90 degrees away - harmless for a single
/// rotation, but wrong for one half of an isoclinic pair (see
/// [`decompose_so4`]): swapping only one half without the other changes
/// which `SO(4)` matrix the pair composes to.
pub(crate) fn quat_slerp_exact(
    a: UnitQuaternion<f32>,
    b: UnitQuaternion<f32>,
    t: f32,
) -> UnitQuaternion<f32> {
    let a = a.coords;
    let b = b.coords;
    let dot = a.dot(&b).clamp(-1.0, 1.0);
    if dot > 0.9995 {
        // Too close together for the general formula below (dividing by a
        // near-zero sin(theta)) to stay numerically stable; linear
        // interpolation is an indistinguishable approximation over such a
        // small arc.
        return quat_from_unit_coords((a + (b - a) * t).normalize());
    }
    let theta = dot.acos();
    let sin_theta = theta.sin();
    let coeff_a = ((1.0 - t) * theta).sin() / sin_theta;
    let coeff_b = (t * theta).sin() / sin_theta;
    quat_from_unit_coords((a * coeff_a + b * coeff_b).normalize())
}

/// Picks whichever of the two double-cover-equivalent representations
/// `(p, q)` / `(-p, -q)` of the same `SO(4)` rotation is jointly closer to
/// `(identity, identity)`, so that slerping both toward identity travels the
/// shorter combined arc. Flipping only one of the pair would represent a
/// different rotation, so the sign choice must be made for both at once.
fn canonicalize_pair(
    p: UnitQuaternion<f32>,
    q: UnitQuaternion<f32>,
) -> (UnitQuaternion<f32>, UnitQuaternion<f32>) {
    if p.coords.w + q.coords.w < 0.0 {
        (
            UnitQuaternion::new_unchecked(-p.into_inner()),
            UnitQuaternion::new_unchecked(-q.into_inner()),
        )
    } else {
        (p, q)
    }
}

/// Decomposes an `SO(4)` rotation matrix into a pair of unit quaternions
/// `(p, q)` such that `compose_so4(p, q) == m`, where `m` acts on a point
/// `x` (reinterpreted as a quaternion) as `p * x * q⁻¹`.
///
/// A unit quaternion's "conjugate" is its inverse: `q⁻¹ = conjugate(q)`.
/// Conjugating a pure-imaginary quaternion by a unit quaternion,
/// `q * v * q⁻¹`, is exactly a 3D rotation of `v` - the same double-cover
/// trick used to spin 3D objects with quaternions, just applied here to
/// recover `q` itself.
///
/// This pairing is unique only up to `(p, q) ↔ (-p, -q)` (both represent
/// the same matrix), resolved by [`canonicalize_pair`].
///
/// Unlike a single plane rotation (see [`shortest_arc_plane`]), which can
/// only align one vector, this pair lets an entire rotation be
/// interpolated toward another (e.g. identity) by slerping `p` and `q`
/// independently - the true geodesic in `SO(4)`.
pub(crate) fn decompose_so4(m: &Matrix4<f32>) -> (UnitQuaternion<f32>, UnitQuaternion<f32>) {
    // a = m applied to the quaternion `1` = p * 1 * q⁻¹ = p * q⁻¹.
    let a = quat_from_unit_coords(m * quat_one());
    let mi = quat_from_unit_coords(m * quat_i());
    let mj = quat_from_unit_coords(m * quat_j());

    // Left-multiplying by a⁻¹ cancels the shared p factor:
    // a⁻¹ * (p * i * q⁻¹) = q * i * q⁻¹, a pure rotation of i by q.
    // Same for j. Being pure rotations of i and j, both results are
    // pure-imaginary (zero scalar part).
    let a_conj = a.conjugate();
    let r = (a_conj * mi).into_inner().coords;
    let s = (a_conj * mj).into_inner().coords;
    let r_vec = Vector3::new(r.x, r.y, r.z);
    let s_vec = Vector3::new(s.x, s.y, s.z);
    // q * k * q⁻¹ = (q * i * q⁻¹) x (q * j * q⁻¹): a unit-quaternion
    // conjugation is an orientation-preserving 3D rotation, so it commutes
    // with the cross product - no need to compute it directly.
    let t_vec = r_vec.cross(&s_vec);

    // r_vec, s_vec, t_vec are q's rotation of the 3D basis vectors i, j, k -
    // i.e. the columns of q's 3x3 rotation matrix. Recover q from it.
    let rot3 = Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[r_vec, s_vec, t_vec]));
    let q = UnitQuaternion::from_rotation_matrix(&rot3);
    let p = a * q; // a = p * q⁻¹  =>  p = a * q

    canonicalize_pair(p, q)
}

/// Rebuilds the `SO(4)` rotation matrix `M(x) = p * x * q⁻¹` from an
/// isoclinic quaternion pair. Inverse of [`decompose_so4`].
pub(crate) fn compose_so4(p: UnitQuaternion<f32>, q: UnitQuaternion<f32>) -> Matrix4<f32> {
    let q_inv = q.conjugate();
    let columns = [quat_i(), quat_j(), quat_k(), quat_one()].map(|coords| {
        (p * quat_from_unit_coords(coords) * q_inv)
            .into_inner()
            .coords
    });
    Matrix4::from_columns(&columns)
}

/// Check if a 4D face is visible from the viewer position.
///
/// Replaces the duplicate implementation in ray_casting.rs is_face_visible().
///
/// # Arguments
/// * `face_id` - Face ID (0-7) to check visibility for
/// * `rotation_4d` - 4D rotation matrix
/// * `viewer_distance` - Distance of 4D viewer from W=0 plane
///
/// # Returns
/// true if the face is visible, false if it should be culled
pub(crate) fn is_face_visible(
    face_id: usize,
    rotation_4d: &Matrix4<f32>,
    viewer_distance: f32,
) -> bool {
    let face_center_4d = FACE_CENTERS[face_id];
    let rotated_face_center = rotation_4d * face_center_4d;
    let viewer_position = Vector4::new(0.0, 0.0, 0.0, viewer_distance);
    let to_viewer = viewer_position - rotated_face_center;
    let dot_product = rotated_face_center.dot(&to_viewer);
    dot_product < 0.0
}

/// Which of the 8 4D faces (`FACE_CENTERS` indices) are currently facing
/// the viewer.
pub(crate) fn visible_faces(rotation_4d: &Matrix4<f32>, viewer_distance: f32) -> [bool; 8] {
    std::array::from_fn(|face_id| is_face_visible(face_id, rotation_4d, viewer_distance))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::FACE_CENTERS;
    use std::f32::consts::FRAC_PI_2;

    const EPSILON: f32 = 1e-4;

    fn assert_vector4_close(a: Vector4<f32>, b: Vector4<f32>) {
        assert!(
            (a - b).norm() < EPSILON,
            "expected {b:?} to be close to {a:?}"
        );
    }

    /// Reference implementation kept only to check `create_4d_plane_rotation`
    /// against, now that `process_4d_rotation` derives its planes from the
    /// camera instead of always using world Y/W.
    fn create_4d_rotation_yw(angle: f32) -> Matrix4<f32> {
        let cos_y = angle.cos();
        let sin_y = angle.sin();
        Matrix4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, cos_y, 0.0, -sin_y, 0.0, 0.0, 1.0, 0.0, 0.0, sin_y, 0.0, cos_y,
        )
    }

    #[test]
    fn plane_rotation_matches_xw_special_case() {
        let x = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let w = Vector4::new(0.0, 0.0, 0.0, 1.0);
        for angle in [0.3, FRAC_PI_2, -1.1] {
            let general = create_4d_plane_rotation(x, w, angle);
            let special = create_4d_rotation_xw(angle);
            assert!(
                (general - special).norm() < EPSILON,
                "angle {angle}: {general:?} != {special:?}"
            );
        }
    }

    #[test]
    fn process_4d_rotation_matches_xw_yw_for_world_aligned_camera() {
        let current = Matrix4::identity();
        let right = Vector3::new(1.0, 0.0, 0.0);
        let up = Vector3::new(0.0, 1.0, 0.0);
        let (delta_x, delta_y) = (12.0, -7.0);

        let actual = process_4d_rotation(&current, delta_x, delta_y, right, up);

        let angle_x = delta_x * MOUSE_SENSITIVITY * 0.01;
        let angle_y = -delta_y * MOUSE_SENSITIVITY * 0.01;
        let expected = create_4d_rotation_yw(angle_y) * create_4d_rotation_xw(angle_x) * current;

        assert!((actual - expected).norm() < EPSILON);
    }

    #[test]
    fn process_4d_rotation_follows_camera_basis() {
        // A camera basis rotated 90 degrees around Y: "right" is world -Z
        // instead of world +X. A horizontal drag should now rotate the ZW
        // plane rather than XW.
        let current = Matrix4::identity();
        let right = Vector3::new(0.0, 0.0, -1.0);
        let up = Vector3::new(0.0, 1.0, 0.0);

        let actual = process_4d_rotation(&current, 12.0, 0.0, right, up);

        // XW should be untouched; ZW should carry the rotation instead.
        assert!((actual[(0, 3)]).abs() < EPSILON);
        assert!((actual[(2, 3)]).abs() > EPSILON);
    }

    #[test]
    fn shortest_arc_plane_is_noop_for_identical_vectors() {
        let (_, _, angle) = shortest_arc_plane(FACE_CENTERS[4], FACE_CENTERS[4]);
        assert!(angle.abs() < EPSILON);
    }

    #[test]
    fn shortest_arc_plane_round_trips_for_orthogonal_axes() {
        for &from in &FACE_CENTERS {
            for &to in &FACE_CENTERS {
                let (u, v, angle) = shortest_arc_plane(from, to);
                let rotation = create_4d_plane_rotation(u, v, angle);
                assert_vector4_close(rotation * from, to);
            }
        }
    }

    #[test]
    fn shortest_arc_plane_round_trips_for_antiparallel_axes() {
        // Face 7 (W=+1) and face 0 (W=-1) are exact opposites - the
        // fallback-plane branch, since `from`/`to` alone don't determine a
        // unique rotation plane.
        let (u, v, angle) = shortest_arc_plane(FACE_CENTERS[7], FACE_CENTERS[0]);
        let rotation = create_4d_plane_rotation(u, v, angle);
        assert_vector4_close(rotation * FACE_CENTERS[7], FACE_CENTERS[0]);
    }

    #[test]
    fn plane_rotation_is_orthogonal() {
        let u = Vector4::new(0.0, 1.0, 0.0, 0.0);
        let v = Vector4::new(0.0, 0.0, 0.0, 1.0);
        let rotation = create_4d_plane_rotation(u, v, 0.7);
        let identity = rotation * rotation.transpose();
        assert!((identity - Matrix4::identity()).norm() < EPSILON);
    }

    #[test]
    fn visible_faces_matches_is_face_visible_per_face() {
        let rotation = create_4d_rotation_xw(0.7);
        let result = visible_faces(&rotation, VIEWER_DISTANCE);
        for (face_id, &visible) in result.iter().enumerate() {
            assert_eq!(
                visible,
                is_face_visible(face_id, &rotation, VIEWER_DISTANCE)
            );
        }
    }

    #[test]
    fn visible_faces_at_identity_matches_known_visibility() {
        let result = visible_faces(&Matrix4::identity(), VIEWER_DISTANCE);
        assert!(result[0], "face 0 (W=-1) should be visible");
        assert!(!result[7], "face 7 (W=+1) should be culled");
    }

    fn assert_matrix4_close(a: Matrix4<f32>, b: Matrix4<f32>) {
        assert!(
            (a - b).norm() < EPSILON,
            "expected {b:?} to be close to {a:?}"
        );
    }

    /// A rotation combining unequal angles in two orthogonal, non-interacting
    /// planes (`xw` and `yz` share no axis) - a "double rotation". Its
    /// isoclinic pair (see `decompose_so4`) ends up with unequal angles too,
    /// unlike a single-plane rotation, whose pair splits the angle evenly
    /// between `p` and `q`.
    fn double_rotation(xw_angle: f32, yz_angle: f32) -> Matrix4<f32> {
        let x = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let y = Vector4::new(0.0, 1.0, 0.0, 0.0);
        let z = Vector4::new(0.0, 0.0, 1.0, 0.0);
        let w = Vector4::new(0.0, 0.0, 0.0, 1.0);
        create_4d_plane_rotation(x, w, xw_angle) * create_4d_plane_rotation(y, z, yz_angle)
    }

    fn sample_so4_matrices() -> Vec<Matrix4<f32>> {
        let x = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let y = Vector4::new(0.0, 1.0, 0.0, 0.0);
        let w = Vector4::new(0.0, 0.0, 0.0, 1.0);
        vec![
            Matrix4::identity(),
            create_4d_plane_rotation(x, w, 0.7),
            create_4d_plane_rotation(y, w, -1.3),
            create_4d_plane_rotation(x, w, 0.4) * create_4d_plane_rotation(y, w, 0.9),
            process_4d_rotation(
                &Matrix4::identity(),
                37.0,
                -21.0,
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ),
            process_4d_rotation(
                &create_4d_plane_rotation(x, w, 2.1),
                14.0,
                50.0,
                Vector3::new(0.0, 0.0, -1.0),
                Vector3::new(0.0, 1.0, 0.0),
            ),
            double_rotation(2.5, 1.0),
        ]
    }

    #[test]
    fn decompose_so4_round_trips() {
        for m in sample_so4_matrices() {
            let (p, q) = decompose_so4(&m);
            assert_matrix4_close(compose_so4(p, q), m);
        }
    }

    #[test]
    fn decompose_so4_of_identity_is_identity_pair() {
        let (p, q) = decompose_so4(&Matrix4::identity());
        assert!((p.coords - UnitQuaternion::identity().coords).norm() < EPSILON);
        assert!((q.coords - UnitQuaternion::identity().coords).norm() < EPSILON);
    }

    #[test]
    fn slerping_isoclinic_pair_to_identity_composes_to_identity() {
        for m in sample_so4_matrices() {
            let (p, q) = decompose_so4(&m);
            let identity = UnitQuaternion::identity();
            let p_end = quat_slerp_exact(p, identity, 1.0);
            let q_end = quat_slerp_exact(q, identity, 1.0);
            assert_matrix4_close(compose_so4(p_end, q_end), Matrix4::identity());
        }
    }

    /// `double_rotation(2.5, 1.0)`'s isoclinic pair has one quaternion more
    /// than 90 degrees from `identity`, exercising the `dot < 0` case where
    /// `quat_slerp_exact` must diverge from `UnitQuaternion::slerp`.
    #[test]
    fn quat_slerp_exact_reaches_the_exact_target_even_when_far_away() {
        let (p, q) = decompose_so4(&double_rotation(2.5, 1.0));
        let identity = UnitQuaternion::identity();
        assert!(
            p.coords.w < 0.0 || q.coords.w < 0.0,
            "test setup must exercise the dot < 0 branch"
        );

        for &(a, name) in &[(p, "p"), (q, "q")] {
            let end = quat_slerp_exact(a, identity, 1.0);
            assert!(
                (end.coords - identity.coords).norm() < EPSILON,
                "{name}: expected exact identity at t=1, got {end:?}"
            );
        }
    }
}
