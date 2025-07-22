//! 4D mathematics utilities for hypercube visualization.
//!
//! This module provides CPU-side 4D rotation calculations, 4D-to-3D projections,
//! and shared transformation logic to eliminate code duplication.

use nalgebra::{Matrix4, Point3, Vector3, Vector4};

use crate::cube::{BASE_CUBE_VERTICES, FACE_CENTERS, FIXED_DIMS};

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
///
/// # Arguments
/// * `angle` - Rotation angle in radians
///
/// # Returns
/// A 4x4 rotation matrix for the XW plane
pub(crate) fn create_4d_rotation_xw(angle: f32) -> Matrix4<f32> {
    let cos_x = angle.cos();
    let sin_x = angle.sin();
    Matrix4::new(
        cos_x, 0.0, 0.0, -sin_x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, sin_x, 0.0, 0.0, cos_x,
    )
}

/// Creates a 4D rotation matrix around the YW plane.
///
/// This rotation affects the Y and W coordinates while leaving X and Z unchanged.
/// Combined with XW rotation, this allows intuitive 4D navigation.
///
/// # Arguments
/// * `angle` - Rotation angle in radians
pub(crate) fn create_4d_rotation_yw(angle: f32) -> Matrix4<f32> {
    let cos_y = angle.cos();
    let sin_y = angle.sin();
    Matrix4::new(
        1.0, 0.0, 0.0, 0.0, 0.0, cos_y, 0.0, -sin_y, 0.0, 0.0, 1.0, 0.0, 0.0, sin_y, 0.0, cos_y,
    )
}

/// Processes mouse input to create incremental 4D rotation.
///
/// Converts mouse movement into 4D rotation by combining XW and YW plane rotations.
/// The rotations are applied incrementally to the existing rotation matrix.
///
/// # Arguments
/// * `current_rotation` - The current 4D rotation matrix
/// * `delta_x` - Horizontal mouse movement delta
/// * `delta_y` - Vertical mouse movement delta
///
/// # Returns
/// Updated 4D rotation matrix incorporating the mouse movement
pub(crate) fn process_4d_rotation(
    current_rotation: &Matrix4<f32>,
    delta_x: f32,
    delta_y: f32,
) -> Matrix4<f32> {
    let angle_x = -delta_x * MOUSE_SENSITIVITY * 0.01;
    let angle_y = -delta_y * MOUSE_SENSITIVITY * 0.01;

    let rotation_xw = create_4d_rotation_xw(angle_x);
    let rotation_yw = create_4d_rotation_yw(angle_y);

    rotation_yw * rotation_xw * current_rotation
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

/// Transform a 4D sticker position to 3D world space.
///
/// Combines sticker offset calculation with 4D rotation and 3D projection.
/// Replaces duplicate logic in ray_casting.rs transform_sticker_to_3d().
///
/// # Arguments
/// * `sticker_position_4d` - 4D position of the sticker
/// * `face_id` - Face ID (0-7) to determine face center and fixed dimension
/// * `rotation_4d` - 4D rotation matrix
/// * `face_spacing` - Spacing multiplier for face separation
/// * `viewer_distance` - Distance of 4D viewer from W=0 plane
///
/// # Returns
/// 3D world position of the transformed sticker
pub(crate) fn calc_sticker_center(
    sticker_position_4d: Vector4<f32>,
    face_id: usize,
    face_spacing: f32,
) -> Vector4<f32> {
    // Get face information
    let face_center_4d = FACE_CENTERS[face_id];

    // Calculate sticker center in 4D (matching shader logic)
    let sticker_offset_4d = sticker_position_4d - face_center_4d;
    let scaled_face_center = face_center_4d * face_spacing;
    scaled_face_center + sticker_offset_4d
}

/// Transform all vertices of a sticker cube to 3D space.
///
/// Replaces the duplicate vertex transformation logic in both
/// ray_casting.rs and shader_widget.rs.
///
/// # Arguments
/// * `sticker_position_4d` - 4D position of the sticker
/// * `face_id` - Face ID (0-7) to determine face center and fixed dimension  
/// * `rotation_4d` - 4D rotation matrix
/// * `sticker_scale` - Scale factor for individual stickers
/// * `face_spacing` - Spacing multiplier for face separation
/// * `viewer_distance` - Distance of 4D viewer from W=0 plane
///
/// # Returns
/// Vector of 36 transformed 3D vertices (one complete cube)
pub(crate) fn transform_sticker_vertices_to_3d(
    sticker_center_4d: Vector4<f32>,
    face_id: usize,
    rotation_4d: &Matrix4<f32>,
    sticker_scale: f32,
    viewer_distance: f32,
) -> Vec<Point3<f32>> {
    let fixed_dim = FIXED_DIMS[face_id];

    // Transform each cube vertex exactly like the shader does
    let mut world_vertices = Vec::with_capacity(36);
    for vertex in &BASE_CUBE_VERTICES {
        let local_vertex =
            Vector3::new(vertex[0], vertex[1], vertex[2]) * BASE_STICKER_SIZE * sticker_scale;
        world_vertices.push(project_cube_point(
            local_vertex,
            sticker_center_4d,
            fixed_dim,
            rotation_4d,
            viewer_distance,
        ));
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
