//! Main application state and logic for the 4D hypercube visualization.
//! 
//! This module coordinates all application components including the hypercube state,
//! camera system, input handling, and 4D rotation processing.

use winit::event::{WindowEvent, DeviceEvent};
use winit::keyboard::ModifiersState;

use crate::camera::{Camera, CameraController, Projection};
use crate::cube::Hypercube;
use crate::input::{InputHandler, InputState};
use crate::math::process_4d_rotation;

/// Field of view for the 3D perspective projection in degrees
const PROJECTION_FOVY: f32 = 45.0;

/// Main application state containing all components for hypercube visualization.
/// 
/// Coordinates the hypercube data, camera system, input handling, and 4D transformations
/// to provide an interactive 4D Rubik's cube experience.
pub struct App {
    /// The 4D hypercube being visualized
    pub hypercube: Hypercube,
    /// 3D camera for viewing the projected hypercube
    pub camera: Camera,
    /// Controller for handling camera movement
    pub camera_controller: CameraController,
    /// 3D perspective projection parameters
    pub projection: Projection,
    /// Current 4D rotation matrix applied to the hypercube
    pub rotation_4d: nalgebra::Matrix4<f32>,
    /// Tracks current input device states
    input_state: InputState,
}

impl App {
    /// Creates a new application with default initial state.
    /// 
    /// Sets up the hypercube, camera, and projection based on window dimensions.
    /// 
    /// # Arguments
    /// * `window_width` - Initial window width in pixels
    /// * `window_height` - Initial window height in pixels
    pub fn new(window_width: u32, window_height: u32) -> Self {
        let hypercube = Hypercube::new();
        
        let mut camera = Camera {
            eye: nalgebra::Point3::new(0.0, 0.0, 15.0),
            target: nalgebra::Point3::new(0.0, 0.0, 0.0),
            up: nalgebra::Vector3::new(0.0, 1.0, 0.0),
        };
        
        let camera_controller = CameraController::new(15.0);
        camera_controller.update_camera(&mut camera);

        let projection = Projection {
            aspect: window_width as f32 / window_height as f32,
            fovy: PROJECTION_FOVY,
            znear: 0.1,
            zfar: 100.0,
        };

        Self {
            hypercube,
            camera,
            camera_controller,
            projection,
            rotation_4d: nalgebra::Matrix4::identity(),
            input_state: InputState::new(),
        }
    }

    /// Handles window resize events by updating projection aspect ratio.
    /// 
    /// # Arguments
    /// * `new_size` - New window dimensions in pixels
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.projection.aspect = new_size.width as f32 / new_size.height as f32;
        }
    }

    /// Updates the application state for the current frame.
    /// 
    /// Currently updates the camera position based on controller state.
    pub fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
    }
}

impl InputHandler for App {
    /// Handles window-level input events like mouse clicks and scrolling.
    /// 
    /// Processes mouse input for camera control and updates internal input state.
    /// 
    /// # Arguments
    /// * `event` - The window event to process
    /// 
    /// # Returns
    /// `true` if the event was handled, `false` otherwise
    fn handle_window_event(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::MouseInput { button, state, .. } => {
                self.input_state.update_mouse_state(*button, *state);
                self.camera_controller.process_mouse_input(*button, *state);
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_delta = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                self.camera_controller.process_scroll(scroll_delta);
                true
            }
            _ => false,
        }
    }

    /// Handles device-level input events like mouse movement.
    /// 
    /// Processes mouse motion for either camera rotation or 4D hypercube rotation
    /// based on modifier key state.
    /// 
    /// # Arguments
    /// * `event` - The device event to process
    /// * `modifiers` - Current modifier key state (Shift, Ctrl, etc.)
    /// 
    /// # Returns
    /// `true` if the event was handled, `false` otherwise
    fn handle_device_event(&mut self, event: &DeviceEvent, modifiers: &ModifiersState) -> bool {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                if self.input_state.is_right_mouse_pressed {
                    if modifiers.shift_key() {
                        self.rotation_4d = process_4d_rotation(&self.rotation_4d, delta.0 as f32, delta.1 as f32);
                    } else {
                        self.camera_controller.process_mouse_motion(delta.0 as f32, delta.1 as f32);
                    }
                }
                true
            }
            _ => false,
        }
    }
}