//! 4D mathematics and projection utilities for hypercube visualization.
//!
//! This module provides the core mathematical operations for 4D geometry,
//! including 4D-to-3D projection, 4D rotations, and instance generation.

use crate::cube::Hypercube;
use crate::renderer::Instance;
use nalgebra::{Vector3, Vector4};

/// Spacing between individual stickers within each 3D side
const STICKER_SPACING: f32 = 1.2;
/// Distance of the 4D viewer from the W=0 hyperplane for projection
const VIEWER_DISTANCE_4D: f32 = 3.0;
/// Mouse sensitivity for 4D rotation controls
const MOUSE_SENSITIVITY: f32 = 0.5;

/// Projects a 4D point to 3D space using perspective projection.
///
/// Similar to how 3D points are projected to 2D screens, this function projects
/// 4D coordinates to 3D space by dividing by the W distance from the viewer.
///
/// # Arguments
/// * `point_4d` - The 4D point to project (x, y, z, w)
/// * `viewer_distance` - Distance of the 4D viewer from the W=0 hyperplane
///
/// # Returns
/// The projected 3D point (x, y, z)
pub(crate) fn project_4d_to_3d(point_4d: Vector4<f32>, viewer_distance: f32) -> Vector3<f32> {
    let w_distance = viewer_distance - point_4d.w;
    let scale = viewer_distance / w_distance;

    Vector3::new(point_4d.x * scale, point_4d.y * scale, point_4d.z * scale)
}

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
pub(crate) fn create_4d_rotation_xw(angle: f32) -> nalgebra::Matrix4<f32> {
    let cos_x = angle.cos();
    let sin_x = angle.sin();
    nalgebra::Matrix4::new(
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
pub(crate) fn create_4d_rotation_yw(angle: f32) -> nalgebra::Matrix4<f32> {
    let cos_y = angle.cos();
    let sin_y = angle.sin();
    nalgebra::Matrix4::new(
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
    current_rotation: &nalgebra::Matrix4<f32>,
    delta_x: f32,
    delta_y: f32,
) -> nalgebra::Matrix4<f32> {
    let angle_x = delta_x * MOUSE_SENSITIVITY * 0.01;
    let angle_y = delta_y * MOUSE_SENSITIVITY * 0.01;

    let rotation_xw = create_4d_rotation_xw(angle_x);
    let rotation_yw = create_4d_rotation_yw(angle_y);

    rotation_yw * rotation_xw * current_rotation
}

/// Generates render instances for all hypercube stickers.
///
/// Applies 4D rotation, then projects each sticker to 3D space for rendering.
/// Creates one instance per sticker with position and color data.
///
/// # Arguments
/// * `hypercube` - The hypercube containing all sticker data
/// * `rotation_4d` - Current 4D rotation matrix to apply
///
/// # Returns
/// Vector of render instances ready for GPU upload
pub(crate) fn generate_instances(
    hypercube: &Hypercube,
    rotation_4d: &nalgebra::Matrix4<f32>,
) -> Vec<Instance> {
    let mut instances = Vec::new();

    for side in &hypercube.sides {
        for sticker in &side.stickers {
            let rotated_4d = rotation_4d * sticker.position.scale(STICKER_SPACING);
            let projected_3d = project_4d_to_3d(rotated_4d, VIEWER_DISTANCE_4D);

            instances.push(Instance {
                position: projected_3d,
                color: nalgebra::Vector4::from(sticker.color),
            });
        }
    }

    instances
}
