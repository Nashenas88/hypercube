//! Camera system for 3D navigation around the 4D hypercube.
//!
//! This module provides an orbital camera system that allows users to rotate around
//! the hypercube origin and zoom in/out for better viewing angles.

use nalgebra::{Matrix4, Point3, Vector3};

/// Mouse rotation sensitivity for camera controls
const MOUSE_SENSITIVITY: f32 = 0.5;
/// Mouse wheel zoom sensitivity
const ZOOM_SENSITIVITY: f32 = 1.0;
/// Minimum camera distance from target
const MIN_DISTANCE: f32 = 5.0;
/// Maximum camera distance from target
const MAX_DISTANCE: f32 = 50.0;

/// 3D camera representing the viewer's position and orientation in space.
///
/// Uses a standard look-at camera model with eye position, target point, and up vector.
#[derive(Debug, Clone)]
pub(crate) struct Camera {
    /// Camera position in 3D space
    pub(crate) eye: Point3<f32>,
    /// Point the camera is looking at (typically the origin)
    pub(crate) target: Point3<f32>,
    /// Up direction vector for camera orientation
    pub(crate) up: Vector3<f32>,
}

impl Camera {
    /// Builds the view matrix for transforming world coordinates to camera space.
    ///
    /// Uses right-handed coordinate system with the camera looking down the negative Z axis.
    pub(crate) fn build_view_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(&self.eye, &self.target, &self.up)
    }

    /// Builds a rotation-only view matrix (translation stripped).
    ///
    /// Shares the same orientation as [`Camera::build_view_matrix`] but keeps the
    /// eye fixed at the origin, so the result carries no eye-position translation.
    fn build_rotation_only_view_matrix(&self) -> Matrix4<f32> {
        let mut view = self.build_view_matrix();
        view[(0, 3)] = 0.0;
        view[(1, 3)] = 0.0;
        view[(2, 3)] = 0.0;
        view
    }
}

/// Orbital camera controller for smooth navigation around a target point.
///
/// Provides mouse-controlled rotation around the target with distance-based zoom.
/// Uses spherical coordinates (yaw/pitch) for intuitive orbital movement.
pub(crate) struct CameraController {
    /// Distance from camera to target point
    pub(crate) distance: f32,
    /// Horizontal rotation angle in degrees
    pub(crate) yaw: f32,
    /// Vertical rotation angle in degrees (clamped to prevent flipping)
    pub(crate) pitch: f32,
}

impl CameraController {
    /// Creates a new camera controller at the specified distance from origin.
    ///
    /// # Arguments
    /// * `distance` - Initial distance from the camera to the target point
    pub(crate) fn new(distance: f32) -> Self {
        Self {
            distance,
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    /// Updates the camera position based on current yaw, pitch, and distance.
    ///
    /// Converts spherical coordinates to Cartesian position around the origin.
    ///
    /// # Arguments
    /// * `camera` - The camera to update with new position and orientation
    pub(crate) fn update_camera(&self, camera: &mut Camera) {
        let yaw_rad = self.yaw.to_radians();
        let pitch_rad = self.pitch.to_radians();

        let x = self.distance * pitch_rad.cos() * yaw_rad.sin();
        let y = self.distance * pitch_rad.sin();
        let z = self.distance * pitch_rad.cos() * yaw_rad.cos();

        camera.eye = Point3::new(x, y, z);
        camera.target = Point3::new(0.0, 0.0, 0.0);
        camera.up = Vector3::new(0.0, 1.0, 0.0);
    }

    /// Processes mouse movement for camera rotation.
    ///
    /// Updates yaw and pitch based on mouse delta, with pitch clamping to prevent camera flipping.
    ///
    /// # Arguments
    /// * `delta_x` - Horizontal mouse movement delta
    /// * `delta_y` - Vertical mouse movement delta
    pub(crate) fn process_mouse_motion(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw -= delta_x * MOUSE_SENSITIVITY;
        self.pitch += delta_y * MOUSE_SENSITIVITY;

        self.pitch = self.pitch.clamp(-89.0, 89.0);
    }

    /// Processes mouse scroll input for camera zoom.
    ///
    /// Adjusts camera distance with bounds checking to maintain reasonable viewing range.
    ///
    /// # Arguments
    /// * `delta` - Scroll wheel delta (positive = zoom in, negative = zoom out)
    pub(crate) fn process_scroll(&mut self, delta: f32) {
        self.distance -= delta * ZOOM_SENSITIVITY;
        self.distance = self.distance.clamp(MIN_DISTANCE, MAX_DISTANCE);
    }
}

/// 3D perspective projection parameters for rendering.
///
/// Defines the viewing frustum and field of view for the camera.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Projection {
    /// Aspect ratio (width/height) of the viewport
    pub(crate) aspect: f32,
    /// Vertical field of view in degrees
    pub(crate) fovy: f32,
    /// Near clipping plane distance
    pub(crate) znear: f32,
    /// Far clipping plane distance
    pub(crate) zfar: f32,
}

impl Projection {
    /// Builds the perspective projection matrix for 3D rendering.
    ///
    /// Creates a standard perspective projection with the current parameters.
    ///
    /// # Returns
    /// A 4x4 projection matrix for transforming camera space to clip space
    pub(crate) fn build_projection_matrix(&self) -> Matrix4<f32> {
        nalgebra::Matrix4::new_perspective(self.aspect, self.fovy, self.znear, self.zfar)
    }
}

/// GPU uniform buffer data for camera transforms.
///
/// Contains the combined view-projection matrix for vertex shader transformation.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CameraUniform {
    /// Combined view-projection matrix as 4x4 array
    pub(crate) view_proj: [[f32; 4]; 4],
    /// Inverse of the translation-free view-projection matrix, used by the skybox
    /// to reproject screen position back to a world-space direction.
    pub(crate) view_proj_inv: [[f32; 4]; 4],
}

impl CameraUniform {
    /// Creates a new camera uniform with identity matrices.
    pub(crate) fn new() -> Self {
        Self {
            view_proj: nalgebra::Matrix4::identity().into(),
            view_proj_inv: nalgebra::Matrix4::identity().into(),
        }
    }

    /// Updates the uniform with current camera and projection matrices.
    ///
    /// Combines the projection and view matrices for efficient GPU transformation,
    /// and derives the translation-free inverse used by the skybox pass.
    ///
    /// # Arguments
    /// * `camera` - Current camera state for view matrix
    /// * `projection` - Current projection parameters
    pub(crate) fn update_view_proj(&mut self, camera: &Camera, projection: &Projection) {
        let proj = projection.build_projection_matrix();
        self.view_proj = (proj * camera.build_view_matrix()).into();

        let rotation_only_view_proj = proj * camera.build_rotation_only_view_matrix();
        self.view_proj_inv = rotation_only_view_proj
            .try_inverse()
            .unwrap_or_else(Matrix4::identity)
            .into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera_at(distance: f32, yaw: f32, pitch: f32) -> Camera {
        let controller = CameraController {
            distance,
            yaw,
            pitch,
        };
        let mut camera = Camera {
            eye: Point3::new(0.0, 0.0, 0.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
        };
        controller.update_camera(&mut camera);
        camera
    }

    fn test_projection() -> Projection {
        Projection {
            aspect: 16.0 / 9.0,
            fovy: 45.0_f32.to_radians(),
            znear: 0.1,
            zfar: 100.0,
        }
    }

    /// Reproduces the skybox vertex shader's screen-corner-to-world-direction math
    /// (`vs_sky` in shader.wgsl) for the four NDC screen corners.
    fn corner_world_directions(camera: &Camera, projection: &Projection) -> [Vector3<f32>; 4] {
        let mut uniform = CameraUniform::new();
        uniform.update_view_proj(camera, projection);
        let inv = Matrix4::from(uniform.view_proj_inv);

        [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)].map(|(x, y)| {
            let world_pos = inv * nalgebra::Vector4::new(x, y, 1.0, 1.0);
            Vector3::new(world_pos.x, world_pos.y, world_pos.z) / world_pos.w
        })
    }

    fn pairwise_angle_cosines(dirs: &[Vector3<f32>; 4]) -> Vec<f32> {
        let mut cosines = Vec::new();
        for i in 0..4 {
            for j in (i + 1)..4 {
                cosines.push(dirs[i].normalize().dot(&dirs[j].normalize()));
            }
        }
        cosines
    }

    #[test]
    fn skybox_directions_preserve_angles_under_yaw_rotation() {
        let projection = test_projection();
        let dirs_a = corner_world_directions(&camera_at(20.0, 0.0, 89.0), &projection);
        let dirs_b = corner_world_directions(&camera_at(20.0, 90.0, 89.0), &projection);

        let cosines_a = pairwise_angle_cosines(&dirs_a);
        let cosines_b = pairwise_angle_cosines(&dirs_b);

        for (a, b) in cosines_a.iter().zip(cosines_b.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "corner angle changed under yaw rotation at extreme pitch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn skybox_directions_are_independent_of_camera_distance() {
        let projection = test_projection();
        let dirs_near = corner_world_directions(&camera_at(5.0, 30.0, 89.0), &projection);
        let dirs_far = corner_world_directions(&camera_at(50.0, 30.0, 89.0), &projection);

        for (near, far) in dirs_near.iter().zip(dirs_far.iter()) {
            let cos_angle = near.normalize().dot(&far.normalize());
            assert!(
                cos_angle > 1.0 - 1e-4,
                "skybox direction depends on camera distance: cos_angle = {cos_angle}"
            );
        }
    }
}
