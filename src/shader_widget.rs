//! Custom shader widget for 4D hypercube rendering.
//!
//! This module implements the shader widget that encapsulates all 3D rendering
//! logic, camera controls, and 4D transformations. It follows Option C architecture
//! where the shader widget manages its own state independently.

use std::time::{Duration, Instant};

use iced::wgpu;
use iced::widget::{Action, shader};
use iced::{Event, Point, Rectangle, event, mouse};
use nalgebra::{Matrix4, Vector3};

use crate::camera::{Camera, CameraController, Projection};
use crate::geometry::{
    BASE_CUBE_VERTICES, FACE_CENTERS, FIXED_DIMS, NORMAL_TO_BASE_INDICES, VERTEX_NORMAL_INDICES,
};
use crate::math::{GRID_EXTENT, VIEWER_DISTANCE, process_4d_rotation, project_cube_point};
use crate::moves::{base_angle, rotate_local_position};
use crate::piece::{
    FACET_TABLE, Hypercube, Piece, StickerInstance, free_axes, generate_sticker_instances,
};
use crate::ray_casting::{calculate_mouse_ray, find_intersected_sticker};
use crate::renderer::{DebugInstanceWithDistance, Renderer};
use crate::settings::RotateButton;
use crate::{AABBMode, Message, RenderMode};

/// An in-progress move's animation: piece state has already been committed
/// atomically by `apply_move`; this only drives the visual sweep from the
/// pre-move snapshot toward the (already-final) post-move positions.
struct AnimatingMove {
    side_axis: usize,
    side_sign: i8,
    local_coords: [i8; 3],
    /// Signed target angle (its sign already encodes direction).
    angle: f32,
    pre_move_pieces: Vec<Piece>,
    elapsed: Duration,
    duration: Duration,
}

/// Smoothstep ease: slow-fast-slow, applied to the normalized [0,1] progress.
fn ease(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Builds the GPU instance list for the current frame. Piece state is
/// already final (`apply_move` commits atomically) - while a move is
/// animating, the 27 affected facets are instead swept from their pre-move
/// position/color toward that already-committed final position, using the
/// exact same rotation formula `apply_move` used, so the last animated
/// frame always lines up perfectly with the static post-move render it
/// hands off to.
fn sticker_instances_for_render(state: &HypercubeShaderState) -> Vec<StickerInstance> {
    let Some(animating) = &state.animating_move else {
        return generate_sticker_instances(&state.hypercube);
    };

    let t = if animating.duration.is_zero() {
        1.0
    } else {
        (animating.elapsed.as_secs_f32() / animating.duration.as_secs_f32()).clamp(0.0, 1.0)
    };
    let partial_angle = animating.angle * ease(t);
    let axes = free_axes(animating.side_axis);

    FACET_TABLE
        .iter()
        .map(|facet| {
            let pre_move_piece = &animating.pre_move_pieces[facet.piece_slot];
            let color = pre_move_piece.colors[facet.axis]
                .expect("FACET_TABLE entries are only built where colors[axis] is Some");

            let position_4d = if pre_move_piece.position[animating.side_axis] == animating.side_sign
            {
                let local_position = [
                    pre_move_piece.position[axes[0]] as f32,
                    pre_move_piece.position[axes[1]] as f32,
                    pre_move_piece.position[axes[2]] as f32,
                ];
                let rotated =
                    rotate_local_position(animating.local_coords, partial_angle, local_position);

                let mut unscaled = [0.0f32; 4];
                unscaled[animating.side_axis] = pre_move_piece.position[animating.side_axis] as f32;
                for i in 0..3 {
                    unscaled[axes[i]] = rotated[i];
                }

                // Matches `facet_position_4d`'s static convention: a facet's
                // own axis is unscaled, the other 3 are GRID_EXTENT-scaled -
                // which axis that is depends on this facet, not on the move.
                let mut position_4d = [0.0f32; 4];
                for axis in 0..4 {
                    position_4d[axis] = if axis == facet.axis {
                        unscaled[axis]
                    } else {
                        unscaled[axis] * GRID_EXTENT
                    };
                }
                position_4d
            } else {
                facet.position_4d
            };

            StickerInstance {
                position_4d,
                color: nalgebra::Vector4::from(color).into(),
                face_id: facet.face_id as u32,
                _padding: [0; 3],
            }
        })
        .collect()
}

/// Parameters controlled from the ui.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UiControls {
    pub(crate) sticker_scale: f32,
    pub(crate) face_scale: f32,
    pub(crate) render_mode: RenderMode,
}

fn scale_bounds(bounds: &Rectangle, scale: f32) -> Rectangle {
    Rectangle {
        x: bounds.x * scale,
        y: bounds.y * scale,
        width: bounds.width * scale,
        height: bounds.height * scale,
    }
}

/// Custom primitive for rendering our 4D hypercube
#[derive(Debug, Clone)]
pub(crate) struct HypercubePrimitive {
    pub(crate) camera: Camera,
    pub(crate) projection: Projection,
    pub(crate) rotation_4d: Matrix4<f32>,
    pub(crate) ui_controls: UiControls,
    pub(crate) cached_indices: Vec<u16>,
    pub(crate) cached_normals: Vec<Vector3<f32>>,
    pub(crate) hovered_sticker: Option<usize>,
    pub(crate) debug_instances: Vec<DebugInstanceWithDistance>,
    pub(crate) sticker_instances: Vec<StickerInstance>,
}

impl shader::Primitive for HypercubePrimitive {
    type Pipeline = Renderer;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        viewport: &shader::Viewport,
    ) {
        let scale = viewport.scale_factor();
        let physical_bounds = scale_bounds(bounds, scale);
        pipeline.resize(device, physical_bounds, viewport.physical_size());
        pipeline.update_instances(
            queue,
            &self.rotation_4d,
            self.ui_controls.sticker_scale,
            self.ui_controls.face_scale,
        );
        pipeline.update_camera(queue, &self.camera, &self.projection);
        pipeline.update_normals(queue, &self.cached_normals);
        pipeline.update_indices(queue, &self.cached_indices);
        pipeline.update_highlighting(queue, self.hovered_sticker);
        pipeline.update_debug_instances(queue, &self.debug_instances);
        pipeline.update_sticker_instances(queue, &self.sticker_instances);
        pipeline.set_render_mode(self.ui_controls.render_mode);
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        _clip_bounds: &Rectangle<u32>,
    ) {
        pipeline.render(encoder, target);

        // Render transparent debug AABBs
        pipeline.render_debug_aabb(encoder, target, self.debug_instances.len() as u32);
    }
}

/// Internal state managed by the shader widget
pub(crate) struct HypercubeShaderState {
    pub(crate) camera: Camera,
    camera_controller: CameraController,
    projection: Projection,
    pub(crate) rotation_4d: nalgebra::Matrix4<f32>,
    mouse_pressed: bool,
    last_mouse_pos: Option<Point>,
    shift_pressed: bool,
    cached_indices: Vec<u16>,
    cached_normals: Vec<Vector3<f32>>,
    hovered_sticker: Option<usize>,
    debug_instances: Vec<DebugInstanceWithDistance>,
    hypercube: Hypercube,
    animating_move: Option<AnimatingMove>,
    last_redraw_instant: Option<Instant>,
}

/// The shader program that handles 4D hypercube rendering
pub(crate) struct HypercubeShaderProgram {
    sticker_scale: f32,
    face_scale: f32,
    render_mode: RenderMode,
    aabb_mode: AABBMode,
    rotate_button: RotateButton,
    animation_duration_ms: u32,
}

impl HypercubeShaderProgram {
    /// Create a new shader program with the given parameters
    pub(crate) fn new(
        sticker_scale: f32,
        face_scale: f32,
        render_mode: RenderMode,
        aabb_mode: AABBMode,
        rotate_button: RotateButton,
        animation_duration_ms: u32,
    ) -> Self {
        Self {
            sticker_scale,
            face_scale,
            render_mode,
            aabb_mode,
            rotate_button,
            animation_duration_ms,
        }
    }
}

impl shader::Program<Message> for HypercubeShaderProgram {
    type State = HypercubeShaderState;
    type Primitive = HypercubePrimitive;

    fn update(
        &self,
        state: &mut Self::State,
        event: &Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<Action<Message>> {
        // Update camera each frame
        state.camera_controller.update_camera(&mut state.camera);

        // Update viewport size if bounds changed
        if bounds.width > 0.0 && bounds.height > 0.0 {
            state.projection.aspect = bounds.width / bounds.height;
        }

        // Check if 4D rotation changed and recalculate normals
        let mut rotation_changed = false;

        let status = match event {
            Event::Mouse(mouse_event) => {
                let old_rotation = state.rotation_4d;
                let result = self.handle_mouse_event(state, mouse_event, bounds, cursor);
                if state.rotation_4d != old_rotation {
                    rotation_changed = true;
                }
                result
            }
            Event::Keyboard(keyboard_event) => self.handle_keyboard_event(state, keyboard_event),
            Event::Window(iced::window::Event::RedrawRequested(now)) => {
                Self::advance_animation(state, *now)
            }
            _ => event::Status::Ignored,
        };

        // Recalculate normals if rotation changed
        if rotation_changed {
            (state.cached_normals, state.cached_indices) =
                Self::calculate_normals_and_indices(&state.rotation_4d);
        }

        match status {
            event::Status::Captured => Some(Action::request_redraw()),
            event::Status::Ignored => None,
        }
    }

    fn draw(
        &self,
        state: &Self::State,
        _cursor: mouse::Cursor,
        _bounds: Rectangle,
    ) -> Self::Primitive {
        HypercubePrimitive {
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
            hovered_sticker: state.hovered_sticker,
            debug_instances: state.debug_instances.clone(),
            sticker_instances: sticker_instances_for_render(state),
        }
    }
}

impl HypercubeShaderProgram {
    /// Calculate normals for all cube faces after 4D transformation and 3D projection
    fn calculate_normals_and_indices(
        rotation_4d: &nalgebra::Matrix4<f32>,
    ) -> (Vec<Vector3<f32>>, Vec<u16>) {
        let mut normals = Vec::with_capacity(48); // 8 faces × 6 normals each
        let mut indices = Vec::with_capacity(288); // 36 indices * 8 4d faces

        for (face_idx, (face_center_4d, fixed_dim)) in
            FACE_CENTERS.iter().zip(FIXED_DIMS.iter()).enumerate()
        {
            // Transform 8 cube vertices to 3D
            let mut transformed_vertices = Vec::with_capacity(8);

            for (vertex_idx, vertex) in BASE_CUBE_VERTICES.iter().enumerate() {
                let local_vertex = Vector3::new(vertex[0], vertex[1], vertex[2]);
                let vertex_3d = project_cube_point(
                    local_vertex,
                    *face_center_4d,
                    *fixed_dim,
                    rotation_4d,
                    VIEWER_DISTANCE,
                )
                .coords;

                log::debug!(
                    "{face_idx} * 8 + {vertex_idx} = {}",
                    face_idx * 8 + vertex_idx
                );
                transformed_vertices.push(vertex_3d);
            }

            // Calculate one normal per cube face (6 faces)
            for (triangle_idx, mut triangle_indices) in VERTEX_NORMAL_INDICES
                .as_chunks::<3>()
                .0
                .iter()
                .copied()
                .enumerate()
            {
                let v0 = transformed_vertices[NORMAL_TO_BASE_INDICES[triangle_indices[0] as usize]];
                let v1 = transformed_vertices[NORMAL_TO_BASE_INDICES[triangle_indices[1] as usize]];
                let v2 = transformed_vertices[NORMAL_TO_BASE_INDICES[triangle_indices[2] as usize]];

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

                indices.extend(triangle_indices);
            }
        }

        (normals, indices)
    }

    /// Handle mouse events for 3D navigation and 4D rotation
    fn handle_mouse_event(
        &self,
        state: &mut HypercubeShaderState,
        mouse_event: &mouse::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> event::Status {
        match mouse_event {
            mouse::Event::CursorMoved { .. } => {
                let Some(position) = cursor.position_in(bounds) else {
                    state.hovered_sticker = None;
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

                // Perform ray casting for sticker hover detection (only when not
                // dragging or mid-animation, since state has already moved past
                // what's currently rendering)
                if !state.mouse_pressed && state.animating_move.is_none() {
                    let mouse_ray =
                        calculate_mouse_ray(position, bounds, &state.camera, &state.projection);

                    let (hovered_sticker, debug_instances) = find_intersected_sticker(
                        &mouse_ray,
                        state,
                        self.sticker_scale,
                        self.face_scale,
                        VIEWER_DISTANCE,
                        self.aabb_mode,
                    );
                    state.hovered_sticker = hovered_sticker;
                    state.debug_instances = debug_instances;
                }

                state.last_mouse_pos = Some(position);
                return event::Status::Captured;
            }
            mouse::Event::ButtonPressed(button) => {
                if cursor.position_in(bounds).is_some()
                    && *button == self.rotate_button.to_mouse_button()
                {
                    state.mouse_pressed = true;
                    return event::Status::Captured;
                }
                if cursor.position_in(bounds).is_some()
                    && *button == self.rotate_button.click_button()
                    && state.animating_move.is_none()
                    && let Some(sticker_index) = state.hovered_sticker
                {
                    self.handle_facet_click(state, sticker_index);
                    return event::Status::Captured;
                }
            }
            mouse::Event::ButtonReleased(_) => {
                if state.mouse_pressed {
                    state.mouse_pressed = false;
                    return event::Status::Captured;
                }
            }
            mouse::Event::WheelScrolled { delta } => {
                if cursor.position_in(bounds).is_some() {
                    let scroll_delta = match delta {
                        mouse::ScrollDelta::Lines { y, .. } => *y,
                        mouse::ScrollDelta::Pixels { y, .. } => y * 0.01,
                    };
                    state.camera_controller.process_scroll(scroll_delta);
                    return event::Status::Captured;
                }
            }
            mouse::Event::CursorEntered => {
                // Handle cursor enter if needed
            }
            mouse::Event::CursorLeft => {
                // Clear hover state when cursor leaves the viewport
                state.hovered_sticker = None;
            }
        }

        event::Status::Ignored
    }

    /// Applies the move triggered by clicking the given facet, if any -
    /// non-actionable facets (cell-centers, the invisible center) are a
    /// no-op. Direction follows the clicked facet's own signed local
    /// coordinates, which is inherently viewer-relative; Shift reverses it.
    fn handle_facet_click(&self, state: &mut HypercubeShaderState, sticker_index: usize) {
        let facet = &FACET_TABLE[sticker_index];
        if !facet.is_actionable {
            return;
        }

        let local_nonzero_count = facet.local_coords.iter().filter(|c| **c != 0).count();
        let magnitude = base_angle(local_nonzero_count);
        let angle = if state.shift_pressed {
            -magnitude
        } else {
            magnitude
        };

        let pre_move_pieces = state.hypercube.pieces.clone();
        state
            .hypercube
            .apply_move(facet.axis, facet.side_sign, facet.local_coords, angle);

        state.animating_move = Some(AnimatingMove {
            side_axis: facet.axis,
            side_sign: facet.side_sign,
            local_coords: facet.local_coords,
            angle,
            pre_move_pieces,
            elapsed: Duration::ZERO,
            duration: Duration::from_millis(self.animation_duration_ms as u64),
        });
        state.last_redraw_instant = None;
        state.hovered_sticker = None;
    }

    /// Advances the in-progress move animation (if any) by the time elapsed
    /// since the last redraw, and requests another redraw if it isn't done
    /// yet - self-sustaining until the animation completes, at which point
    /// no further redraw is requested and the loop naturally stops.
    fn advance_animation(state: &mut HypercubeShaderState, now: Instant) -> event::Status {
        let Some(animating) = state.animating_move.as_mut() else {
            return event::Status::Ignored;
        };

        let delta = state
            .last_redraw_instant
            .map(|last| now.duration_since(last))
            .unwrap_or_default();
        animating.elapsed += delta;
        state.last_redraw_instant = Some(now);

        if animating.elapsed >= animating.duration {
            state.animating_move = None;
            state.last_redraw_instant = None;
        }

        event::Status::Captured
    }

    /// Handle keyboard events for additional controls
    fn handle_keyboard_event(
        &self,
        state: &mut HypercubeShaderState,
        keyboard_event: &iced::keyboard::Event,
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
            camera,
            camera_controller,
            projection,
            rotation_4d,
            mouse_pressed: false,
            last_mouse_pos: None,
            shift_pressed: false,
            cached_indices,
            cached_normals,
            hovered_sticker: None,
            debug_instances: Vec::new(),
            hypercube: Hypercube::solved(),
            animating_move: None,
            last_redraw_instant: None,
        }
    }
}
