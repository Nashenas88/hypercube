//! Camera system for 3D navigation around the 4D hypercube.
//! 
//! This module provides an orbital camera system that allows users to rotate around
//! the hypercube origin and zoom in/out for better viewing angles.

use nalgebra::{Matrix4, Point3, Vector3};
use winit::event::{MouseButton, ElementState};

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
pub struct Camera {
    /// Camera position in 3D space
    pub eye: Point3<f32>,
    /// Point the camera is looking at (typically the origin)
    pub target: Point3<f32>,
    /// Up direction vector for camera orientation
    pub up: Vector3<f32>,
}

impl Camera {
    /// Builds the view matrix for transforming world coordinates to camera space.
    /// 
    /// Uses right-handed coordinate system with the camera looking down the negative Z axis.
    pub fn build_view_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(&self.eye, &self.target, &self.up)
    }
}

/// Orbital camera controller for smooth navigation around a target point.
/// 
/// Provides mouse-controlled rotation around the target with distance-based zoom.
/// Uses spherical coordinates (yaw/pitch) for intuitive orbital movement.
pub struct CameraController {
    /// Distance from camera to target point
    pub distance: f32,
    /// Horizontal rotation angle in degrees
    pub yaw: f32,
    /// Vertical rotation angle in degrees (clamped to prevent flipping)
    pub pitch: f32,
    /// Last recorded mouse position for delta calculations
    pub last_mouse_pos: Option<(f32, f32)>,
}

impl CameraController {
    /// Creates a new camera controller at the specified distance from origin.
    /// 
    /// # Arguments
    /// * `distance` - Initial distance from the camera to the target point
    pub fn new(distance: f32) -> Self {
        Self {
            distance,
            yaw: 0.0,
            pitch: 0.0,
            last_mouse_pos: None,
        }
    }

    /// Updates the camera position based on current yaw, pitch, and distance.
    /// 
    /// Converts spherical coordinates to Cartesian position around the origin.
    /// 
    /// # Arguments
    /// * `camera` - The camera to update with new position and orientation
    pub fn update_camera(&self, camera: &mut Camera) {
        let yaw_rad = self.yaw.to_radians();
        let pitch_rad = self.pitch.to_radians();
        
        let x = self.distance * pitch_rad.cos() * yaw_rad.sin();
        let y = self.distance * pitch_rad.sin();
        let z = self.distance * pitch_rad.cos() * yaw_rad.cos();
        
        camera.eye = Point3::new(x, y, z);
        camera.target = Point3::new(0.0, 0.0, 0.0);
        camera.up = Vector3::new(0.0, 1.0, 0.0);
    }

    /// Processes mouse button input for camera control.
    /// 
    /// Tracks right mouse button state for enabling/disabling camera rotation.
    /// 
    /// # Arguments
    /// * `button` - The mouse button that was pressed/released
    /// * `state` - Whether the button was pressed or released
    pub fn process_mouse_input(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Right {
            if state == ElementState::Released {
                self.last_mouse_pos = None;
            }
        }
    }

    /// Processes mouse movement for camera rotation.
    /// 
    /// Updates yaw and pitch based on mouse delta, with pitch clamping to prevent camera flipping.
    /// 
    /// # Arguments
    /// * `delta_x` - Horizontal mouse movement delta
    /// * `delta_y` - Vertical mouse movement delta
    pub fn process_mouse_motion(&mut self, delta_x: f32, delta_y: f32) {
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
    pub fn process_scroll(&mut self, delta: f32) {
        self.distance -= delta * ZOOM_SENSITIVITY;
        self.distance = self.distance.clamp(MIN_DISTANCE, MAX_DISTANCE);
    }
}

/// 3D perspective projection parameters for rendering.
/// 
/// Defines the viewing frustum and field of view for the camera.
pub struct Projection {
    /// Aspect ratio (width/height) of the viewport
    pub aspect: f32,
    /// Vertical field of view in degrees
    pub fovy: f32,
    /// Near clipping plane distance
    pub znear: f32,
    /// Far clipping plane distance
    pub zfar: f32,
}

impl Projection {
    /// Builds the perspective projection matrix for 3D rendering.
    /// 
    /// Creates a standard perspective projection with the current parameters.
    /// 
    /// # Returns
    /// A 4x4 projection matrix for transforming camera space to clip space
    pub fn build_projection_matrix(&self) -> Matrix4<f32> {
        nalgebra::Matrix4::new_perspective(
            self.aspect,
            self.fovy,
            self.znear,
            self.zfar,
        )
    }
}

/// GPU uniform buffer data for camera transforms.
/// 
/// Contains the combined view-projection matrix for vertex shader transformation.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// Combined view-projection matrix as 4x4 array
    pub view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    /// Creates a new camera uniform with identity matrix.
    pub fn new() -> Self {
        Self {
            view_proj: nalgebra::Matrix4::identity().into(),
        }
    }

    /// Updates the uniform with current camera and projection matrices.
    /// 
    /// Combines the projection and view matrices for efficient GPU transformation.
    /// 
    /// # Arguments
    /// * `camera` - Current camera state for view matrix
    /// * `projection` - Current projection parameters
    pub fn update_view_proj(&mut self, camera: &Camera, projection: &Projection) {
        self.view_proj = (projection.build_projection_matrix() * camera.build_view_matrix()).into();
    }
}