use nalgebra::{Matrix4, Point3, Vector3};
use winit::event::{MouseButton, ElementState};

const MOUSE_SENSITIVITY: f32 = 0.5;
const ZOOM_SENSITIVITY: f32 = 1.0;
const MIN_DISTANCE: f32 = 5.0;
const MAX_DISTANCE: f32 = 50.0;

pub struct Camera {
    pub eye: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,
}

impl Camera {
    pub fn build_view_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(&self.eye, &self.target, &self.up)
    }
}

pub struct CameraController {
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub last_mouse_pos: Option<(f32, f32)>,
}

impl CameraController {
    pub fn new(distance: f32) -> Self {
        Self {
            distance,
            yaw: 0.0,
            pitch: 0.0,
            last_mouse_pos: None,
        }
    }

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

    pub fn process_mouse_input(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Right {
            if state == ElementState::Released {
                self.last_mouse_pos = None;
            }
        }
    }

    pub fn process_mouse_motion(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw -= delta_x * MOUSE_SENSITIVITY;
        self.pitch += delta_y * MOUSE_SENSITIVITY;
        
        self.pitch = self.pitch.clamp(-89.0, 89.0);
    }

    pub fn process_scroll(&mut self, delta: f32) {
        self.distance -= delta * ZOOM_SENSITIVITY;
        self.distance = self.distance.clamp(MIN_DISTANCE, MAX_DISTANCE);
    }
}

pub struct Projection {
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Projection {
    pub fn build_projection_matrix(&self) -> Matrix4<f32> {
        nalgebra::Matrix4::new_perspective(
            self.aspect,
            self.fovy,
            self.znear,
            self.zfar,
        )
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: nalgebra::Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera, projection: &Projection) {
        self.view_proj = (projection.build_projection_matrix() * camera.build_view_matrix()).into();
    }
}