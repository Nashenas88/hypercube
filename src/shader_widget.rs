//! Custom shader widget for 4D hypercube rendering.
//!
//! This module implements the shader widget that encapsulates all 3D rendering
//! logic, camera controls, and 4D transformations. It follows Option C architecture
//! where the shader widget manages its own state independently.

use iced::widget::shader::{self, wgpu};
use iced::{Point, Rectangle, event, mouse};
use nalgebra::Matrix4;

use crate::{Message, RenderMode};
use crate::camera::{Camera, CameraController, Projection};
use crate::cube::Hypercube;
use crate::math::process_4d_rotation;
use crate::renderer::Renderer;

/// Custom primitive for rendering our 4D hypercube
#[derive(Debug, Clone)]
pub(crate) struct HypercubePrimitive {
    pub(crate) hypercube: Hypercube,
    pub(crate) camera: Camera,
    pub(crate) projection: Projection,
    pub(crate) rotation_4d: Matrix4<f32>,
    pub(crate) sticker_scale: f32,
    pub(crate) face_scale: f32,
    pub(crate) render_mode: RenderMode,
}

impl shader::Primitive for HypercubePrimitive {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        storage: &mut shader::Storage,
        bounds: &Rectangle,
        viewport: &shader::Viewport,
    ) {
        if !storage.has::<Renderer>() {
            let renderer = pollster::block_on(Renderer::new(
                device,
                format,
                *bounds,
                viewport.physical_size(),
                &self.hypercube,
                self.sticker_scale,
                self.face_scale,
                self.render_mode,
            ));
            storage.store(renderer);
        }
        let renderer = storage.get_mut::<Renderer>().unwrap();
        renderer.resize(device, *bounds, viewport.physical_size());
        renderer.update_instances(
            queue,
            &self.rotation_4d,
            self.sticker_scale,
            self.face_scale,
        );
        renderer.update_camera(queue, &self.camera, &self.projection);
        renderer.set_render_mode(self.render_mode);
    }

    fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        storage: &shader::Storage,
        target: &wgpu::TextureView,
        _clip_bounds: &Rectangle<u32>,
    ) {
        let renderer = storage.get::<Renderer>().unwrap();
        renderer.compute_instances(encoder);
        renderer.render(encoder, target);
    }
}

/// Internal state managed by the shader widget
pub(crate) struct HypercubeShaderState {
    hypercube: Hypercube,
    camera: Camera,
    camera_controller: CameraController,
    projection: Projection,
    rotation_4d: nalgebra::Matrix4<f32>,
    mouse_pressed: bool,
    last_mouse_pos: Option<Point>,
    shift_pressed: bool,
}

/// The shader program that handles 4D hypercube rendering
pub(crate) struct HypercubeShaderProgram {
    sticker_scale: f32,
    face_scale: f32,
    render_mode: RenderMode,
}

impl HypercubeShaderProgram {
    /// Create a new shader program with the given parameters
    pub(crate) fn new(sticker_scale: f32, face_scale: f32, render_mode: RenderMode) -> Self {
        Self {
            sticker_scale,
            face_scale,
            render_mode,
        }
    }
}

impl shader::Program<Message> for HypercubeShaderProgram {
    type State = HypercubeShaderState;
    type Primitive = HypercubePrimitive;

    fn update(
        &self,
        state: &mut Self::State,
        event: shader::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
        _shell: &mut iced::advanced::Shell<'_, Message>,
    ) -> (event::Status, Option<Message>) {
        // Update camera each frame
        state.camera_controller.update_camera(&mut state.camera);

        // Update viewport size if bounds changed
        if bounds.width > 0.0 && bounds.height > 0.0 {
            state.projection.aspect = bounds.width / bounds.height;
        }

        let status = match event {
            shader::Event::Mouse(mouse_event) => {
                self.handle_mouse_event(state, mouse_event, bounds, cursor)
            }
            shader::Event::Keyboard(keyboard_event) => {
                self.handle_keyboard_event(state, keyboard_event)
            }
            _ => event::Status::Ignored,
        };

        (status, None)
    }

    fn draw(
        &self,
        state: &Self::State,
        _cursor: mouse::Cursor,
        _bounds: Rectangle,
    ) -> Self::Primitive {
        HypercubePrimitive {
            hypercube: state.hypercube.clone(),
            camera: state.camera.clone(),
            projection: state.projection,
            rotation_4d: state.rotation_4d,
            sticker_scale: self.sticker_scale,
            face_scale: self.face_scale,
            render_mode: self.render_mode,
        }
    }
}

impl HypercubeShaderProgram {
    /// Handle mouse events for 3D navigation and 4D rotation
    fn handle_mouse_event(
        &self,
        state: &mut HypercubeShaderState,
        mouse_event: mouse::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> event::Status {
        match mouse_event {
            mouse::Event::CursorMoved { .. } => {
                let Some(position) = cursor.position_in(bounds) else {
                    return event::Status::Ignored;
                };
                // Calculate mouse delta for camera movement
                if let Some(last_pos) = state.last_mouse_pos {
                    let delta_x = position.x - last_pos.x;
                    let delta_y = position.y - last_pos.y;

                    // Apply mouse movement to camera or 4D rotation
                    if state.mouse_pressed {
                        if state.shift_pressed {
                            // 4D rotation
                            state.rotation_4d =
                                process_4d_rotation(&state.rotation_4d, delta_x, delta_y);
                        } else {
                            // 3D camera rotation
                            state
                                .camera_controller
                                .process_mouse_motion(delta_x, delta_y);
                        }
                    }
                }
                state.last_mouse_pos = Some(position);
                return event::Status::Captured;
            }
            mouse::Event::ButtonPressed(button) => {
                if cursor.position_in(bounds).is_some() && button == mouse::Button::Right {
                    state.mouse_pressed = true;
                    state.camera_controller.process_mouse_press(button);
                    return event::Status::Captured;
                }
            }
            mouse::Event::ButtonReleased(button) => {
                if button == mouse::Button::Right {
                    state.mouse_pressed = false;
                    state.camera_controller.process_mouse_release(button);
                    return event::Status::Captured;
                }
            }
            mouse::Event::WheelScrolled { delta } => {
                if cursor.position_in(bounds).is_some() {
                    let scroll_delta = match delta {
                        mouse::ScrollDelta::Lines { y, .. } => y,
                        mouse::ScrollDelta::Pixels { y, .. } => y * 0.01,
                    };
                    state.camera_controller.process_scroll(scroll_delta);
                    return event::Status::Captured;
                }
            }
            mouse::Event::CursorEntered | mouse::Event::CursorLeft => {
                // Handle cursor enter/leave if needed
            }
        }

        event::Status::Ignored
    }

    /// Handle keyboard events for additional controls
    fn handle_keyboard_event(
        &self,
        state: &mut HypercubeShaderState,
        keyboard_event: iced::keyboard::Event,
    ) -> event::Status {
        use iced::keyboard::Event;
        use iced::keyboard::{Key, key};
        match keyboard_event {
            Event::KeyPressed {
                key: Key::Named(key::Named::Shift),
                ..
            } => {
                state.shift_pressed = true;
                return event::Status::Captured;
            }
            Event::KeyReleased {
                key: Key::Named(key::Named::Shift),
                ..
            } => {
                state.shift_pressed = false;
                return event::Status::Captured;
            }
            _ => {}
        }

        event::Status::Ignored
    }
}

impl Default for HypercubeShaderState {
    fn default() -> Self {
        let hypercube = Hypercube::new();

        let mut camera = Camera {
            eye: nalgebra::Point3::new(0.0, 0.0, 15.0),
            target: nalgebra::Point3::new(0.0, 0.0, 0.0),
            up: nalgebra::Vector3::new(0.0, 1.0, 0.0),
        };

        let camera_controller = CameraController::new(15.0);
        camera_controller.update_camera(&mut camera);

        let projection = Projection {
            aspect: 800.0 / 600.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        Self {
            hypercube,
            camera,
            camera_controller,
            projection,
            rotation_4d: nalgebra::Matrix4::identity(),
            mouse_pressed: false,
            last_mouse_pos: None,
            shift_pressed: false,
        }
    }
}
