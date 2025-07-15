//! 4D mathematics utilities for hypercube visualization.
//!
//! This module provides CPU-side 4D rotation calculations and mouse input processing.
//! The heavy 4D-to-3D projection and instance generation is now handled by compute shaders.

use nalgebra::Matrix4;

/// Mouse sensitivity for 4D rotation controls
const MOUSE_SENSITIVITY: f32 = 0.5;


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
    let angle_x = delta_x * MOUSE_SENSITIVITY * 0.01;
    let angle_y = delta_y * MOUSE_SENSITIVITY * 0.01;

    let rotation_xw = create_4d_rotation_xw(angle_x);
    let rotation_yw = create_4d_rotation_yw(angle_y);

    rotation_yw * rotation_xw * current_rotation
}

