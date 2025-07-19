//! Custom shader widget for 4D hypercube rendering.
//!
//! This module implements the shader widget that encapsulates all 3D rendering
//! logic, camera controls, and 4D transformations. It follows Option C architecture
//! where the shader widget manages its own state independently.

use iced::widget::shader::{self, wgpu};
use iced::{Point, Rectangle, event, mouse};
use nalgebra::{Matrix4, Vector3};

use crate::camera::{Camera, CameraController, Projection};
use crate::cube::{BASE_INDICES, CUBE_VERTICES, Hypercube};
use crate::math::process_4d_rotation;
use crate::renderer::Renderer;
use crate::{Message, RenderMode};

/// Parameters controlled from the ui.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UiControls {
    pub(crate) sticker_scale: f32,
    pub(crate) face_scale: f32,
    pub(crate) render_mode: RenderMode,
}

/// Custom primitive for rendering our 4D hypercube
#[derive(Debug, Clone)]
pub(crate) struct HypercubePrimitive {
    pub(crate) hypercube: Hypercube,
    pub(crate) camera: Camera,
    pub(crate) projection: Projection,
    pub(crate) rotation_4d: Matrix4<f32>,
    pub(crate) ui_controls: UiControls,
    pub(crate) cached_indices: Vec<u16>,
    pub(crate) cached_normals: Vec<Vector3<f32>>,
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
                self.ui_controls,
            ));
            storage.store(renderer);
        }
        let renderer = storage.get_mut::<Renderer>().unwrap();
        renderer.resize(device, *bounds, viewport.physical_size());
        renderer.update_instances(
            queue,
            &self.rotation_4d,
            self.ui_controls.sticker_scale,
            self.ui_controls.face_scale,
        );
        renderer.update_camera(queue, &self.camera, &self.projection);
        renderer.update_normals(queue, &self.cached_normals);
        renderer.update_indices(queue, &self.cached_indices);
        renderer.set_render_mode(self.ui_controls.render_mode);
    }

    fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        storage: &shader::Storage,
        target: &wgpu::TextureView,
        _clip_bounds: &Rectangle<u32>,
    ) {
        let renderer = storage.get::<Renderer>().unwrap();
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
    cached_indices: Vec<u16>,
    cached_normals: Vec<Vector3<f32>>,
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

        // Check if 4D rotation changed and recalculate normals
        let mut rotation_changed = false;

        let status = match event {
            shader::Event::Mouse(mouse_event) => {
                let old_rotation = state.rotation_4d;
                let result = self.handle_mouse_event(state, mouse_event, bounds, cursor);
                if state.rotation_4d != old_rotation {
                    rotation_changed = true;
                }
                result
            }
            shader::Event::Keyboard(keyboard_event) => {
                self.handle_keyboard_event(state, keyboard_event)
            }
            _ => event::Status::Ignored,
        };

        // Recalculate normals if rotation changed
        if rotation_changed {
            (state.cached_normals, state.cached_indices) =
                Self::calculate_normals_and_indices(&state.rotation_4d);
        }

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
            ui_controls: UiControls {
                sticker_scale: self.sticker_scale,
                face_scale: self.face_scale,
                render_mode: self.render_mode,
            },
            cached_indices: state.cached_indices.clone(),
            cached_normals: state.cached_normals.clone(),
        }
    }
}

const FACE_CENTERS: [nalgebra::Vector4<f32>; 8] = [
    nalgebra::Vector4::new(0.0, 0.0, 0.0, -1.0), // Face 0: W = -1
    nalgebra::Vector4::new(0.0, 0.0, -1.0, 0.0), // Face 1: Z = -1
    nalgebra::Vector4::new(0.0, -1.0, 0.0, 0.0), // Face 2: Y = -1
    nalgebra::Vector4::new(-1.0, 0.0, 0.0, 0.0), // Face 3: X = -1
    nalgebra::Vector4::new(1.0, 0.0, 0.0, 0.0),  // Face 4: X = +1
    nalgebra::Vector4::new(0.0, 1.0, 0.0, 0.0),  // Face 5: Y = +1
    nalgebra::Vector4::new(0.0, 0.0, 1.0, 0.0),  // Face 6: Z = +1
    nalgebra::Vector4::new(0.0, 0.0, 0.0, 1.0),  // Face 7: W = +1
];

const FIXED_DIMENSIONS: [usize; 8] = [3, 2, 1, 0, 0, 1, 2, 3];

const VIEWER_DISTANCE: f32 = 3.0;

impl HypercubeShaderProgram {
    /// Calculate normals for all cube faces after 4D transformation and 3D projection
    fn calculate_normals_and_indices(
        rotation_4d: &nalgebra::Matrix4<f32>,
    ) -> (Vec<Vector3<f32>>, Vec<u16>) {
        let mut normals = Vec::with_capacity(48); // 8 faces × 6 normals each
        let mut indices = Vec::with_capacity(288); // 36 indices * 8 4d faces

        for (face_idx, (face_center_4d, fixed_dim)) in
            FACE_CENTERS.iter().zip(FIXED_DIMENSIONS.iter()).enumerate()
        {
            // Transform 8 cube vertices to 3D
            let mut transformed_vertices = Vec::with_capacity(36);

            for (vertex_idx, vertex) in CUBE_VERTICES.iter().enumerate() {
                let local_vertex = Vector3::new(vertex[0], vertex[1], vertex[2]);

                // Generate vertex in 4D space around face center (matching shader logic)
                let mut vertex_4d = *face_center_4d;
                let mut offset_idx = 0;

                for axis in 0..4 {
                    if axis != *fixed_dim {
                        match offset_idx {
                            0 => vertex_4d[axis] += local_vertex.x,
                            1 => vertex_4d[axis] += local_vertex.y,
                            2 => vertex_4d[axis] += local_vertex.z,
                            _ => {}
                        }
                        offset_idx += 1;
                    }
                }
                log::debug!(
                    "Face {face_idx}, vertex {vertex_idx} Mapping {local_vertex:?} to {vertex_4d:?}"
                );

                // Apply 4D rotation
                let rotated_vertex_4d = rotation_4d * vertex_4d;

                // Project to 3D (matching shader logic)
                let w_distance = VIEWER_DISTANCE - rotated_vertex_4d.w;
                let scale = VIEWER_DISTANCE / w_distance;
                let vertex_3d = Vector3::new(
                    rotated_vertex_4d.x * scale,
                    rotated_vertex_4d.y * scale,
                    rotated_vertex_4d.z * scale,
                );

                log::debug!(
                    "{face_idx} * 8 + {vertex_idx} = {}",
                    face_idx * 8 + vertex_idx
                );
                transformed_vertices.push(vertex_3d);
            }

            // Calculate one normal per cube face (6 faces)
            for (triangle_idx, mut triangle_indices) in
                BASE_INDICES.as_chunks::<3>().0.iter().copied().enumerate()
            {
                let v0 = transformed_vertices[triangle_indices[0] as usize];
                let v1 = transformed_vertices[triangle_indices[1] as usize];
                let v2 = transformed_vertices[triangle_indices[2] as usize];

                // Calculate triangle normal using cross product
                let edge1 = v1 - v0;
                let edge2 = v2 - v0;
                let mut normal = edge1.cross(&edge2);

                // Normalize and check for degenerate triangles
                let length = normal.norm();
                if length > 1e-6 {
                    normal /= length;
                } else {
                    // Degenerate triangle, use a default normal
                    log::warn!(
                        "Degenerate triangle detected for 4D face {face_idx} triangle {triangle_idx}: vertices {v0:?}, {v1:?}, {v2:?}"
                    );
                    normal = Vector3::new(0.0, 0.0, 1.0);
                }

                // Check winding order: normal should point outward from cube center
                let centroid = transformed_vertices.iter().sum::<Vector3<f32>>() / 8.0;
                if normal.dot(&centroid) < 0.0 {
                    log::debug!(
                        "Bad winding order detected for 4D face {face_idx} cube face {triangle_idx}: normal {normal:?} points inward, flipping"
                    );
                    triangle_indices.swap(1, 2);
                }

                if triangle_idx % 2 == 0 {
                    log::debug!(
                        "normal: {normal:?} for face {}, {face_idx}",
                        triangle_idx / 2
                    );
                    // Add this normal for all 6 vertices of this cube face (2 triangles × 3 vertices)
                    normals.push(normal);
                }

                indices.extend(triangle_indices.into_iter());
            }
        }

        (normals, indices)
    }

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
            up: Vector3::new(0.0, 1.0, 0.0),
        };

        let camera_controller = CameraController::new(15.0);
        camera_controller.update_camera(&mut camera);

        let projection = Projection {
            aspect: 800.0 / 600.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let rotation_4d = nalgebra::Matrix4::identity();
        let (cached_normals, cached_indices) =
            HypercubeShaderProgram::calculate_normals_and_indices(&rotation_4d);

        Self {
            hypercube,
            camera,
            camera_controller,
            projection,
            rotation_4d,
            mouse_pressed: false,
            last_mouse_pos: None,
            shift_pressed: false,
            cached_indices,
            cached_normals,
        }
    }
}
