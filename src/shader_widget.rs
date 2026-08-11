//! Custom shader widget for 4D hypercube rendering.
//!
//! This module implements the shader widget that encapsulates all 3D rendering
//! logic, camera controls, and 4D transformations. It follows Option C architecture
//! where the shader widget manages its own state independently.

use std::time::{Duration, Instant};

use iced::wgpu;
use iced::widget::{Action, shader};
use iced::{Event, Point, Rectangle, event, mouse};
use nalgebra::{Matrix4, Vector3, Vector4};

use crate::camera::{Camera, CameraController, Projection};
use crate::geometry::{
    BASE_CUBE_VERTICES, FACE_CENTERS, FIXED_DIMS, NORMAL_TO_BASE_INDICES, VERTEX_NORMAL_INDICES,
};
use crate::math::{
    GRID_EXTENT, VIEWER_DISTANCE, create_4d_plane_rotation, process_4d_rotation,
    project_cube_point, shortest_arc_plane,
};
use crate::moves::{base_angle, clockwise_sign, rotate_local_position};
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

/// Outcome of advancing the move animation by one tick.
enum AnimationTick {
    /// No animation was in progress this tick.
    Ignored,
    /// Animation advanced but is still running.
    Running,
    /// Animation just crossed its duration threshold this tick.
    Completed,
}

/// An in-progress "center this face" animation, triggered by double-clicking
/// a sticker: sweeps `rotation_4d` from its value when the double-click
/// landed toward `start_rotation` rotated by `total_angle` in `plane`, which
/// by construction carries the double-clicked face's normal onto the
/// screen-centered pole (see `shortest_arc_plane` and its call site below).
struct AnimatingFocus {
    start_rotation: Matrix4<f32>,
    plane: (Vector4<f32>, Vector4<f32>),
    total_angle: f32,
    elapsed: Duration,
    duration: Duration,
}

/// Smoothstep ease: slow-fast-slow, applied to the normalized \[0,1\] progress.
fn ease(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Max cursor movement between a rotate-button press and release for it to
/// still count as a click rather than a drag.
const CLICK_DRAG_THRESHOLD_PX: f32 = 4.0;
/// Max gap between two qualifying clicks on the same face for them to count
/// as a double-click.
const DOUBLE_CLICK_WINDOW: Duration = Duration::from_millis(400);

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

            let (position_4d, basis, face_normal_4d) = if pre_move_piece.position
                [animating.side_axis]
                == animating.side_sign
            {
                // `facet_position_4d`'s static convention is `pos *
                // GRID_EXTENT + extension`, where `extension` is zero except
                // at the facet's own axis (the extra push from grid-scale
                // out to the tesseract boundary). `extension` is fixed to
                // the piece's own body, so it must rotate along with the
                // piece rather than stay pinned to the pre-move axis -
                // folding it into the local position before rotating
                // (instead of exempting an axis after rotating) achieves
                // that, since rotation is linear.
                let mut local_combined = [
                    pre_move_piece.position[axes[0]] as f32 * GRID_EXTENT,
                    pre_move_piece.position[axes[1]] as f32 * GRID_EXTENT,
                    pre_move_piece.position[axes[2]] as f32 * GRID_EXTENT,
                ];
                let facet_axis_is_free = axes.iter().position(|&axis| axis == facet.axis);
                if let Some(i) = facet_axis_is_free {
                    local_combined[i] +=
                        pre_move_piece.position[facet.axis] as f32 * (1.0 - GRID_EXTENT);
                }
                let rotated =
                    rotate_local_position(animating.local_coords, partial_angle, local_combined);

                let mut position_4d = [0.0f32; 4];
                position_4d[animating.side_axis] = pre_move_piece.position[animating.side_axis]
                    as f32
                    * if facet.axis == animating.side_axis {
                        1.0
                    } else {
                        GRID_EXTENT
                    };
                for i in 0..3 {
                    position_4d[axes[i]] = rotated[i];
                }

                // The facet's static mesh basis is the unit vectors along
                // `facet.free_axes`. Each one is either exactly `side_axis`
                // (untouched by the slab's rotation) or a one-hot vector
                // inside the rotating `axes` subspace - rotating that one-hot
                // vector the same way position is rotated above gives the
                // basis vector's new direction directly, by linearity.
                let mut basis = [[0.0f32; 4]; 3];
                for (i, &a) in facet.free_axes.iter().enumerate() {
                    basis[i] = if a == animating.side_axis {
                        let mut v = [0.0f32; 4];
                        v[a] = 1.0;
                        v
                    } else {
                        let j = axes.iter().position(|&x| x == a).expect(
                            "a facet free axis other than side_axis must be one of the move's free axes",
                        );
                        let mut one_hot = [0.0f32; 3];
                        one_hot[j] = 1.0;
                        let rotated =
                            rotate_local_position(animating.local_coords, partial_angle, one_hot);
                        let mut v = [0.0f32; 4];
                        for k in 0..3 {
                            v[axes[k]] = rotated[k];
                        }
                        v
                    };
                }

                // The facet's outward normal is the one-hot vector along its
                // own axis (signed by `side_sign`), used for 4D face
                // culling. When `facet.axis == side_axis`, this is the
                // slab's own outer face - it doesn't rotate, so it stays
                // static. Otherwise it's genuinely sweeping toward a
                // different tesseract cell along with the rest of the
                // rotating subspace, so it rotates the same way the
                // tangent basis vectors above do.
                let face_normal_4d = if let Some(j) = facet_axis_is_free {
                    let mut one_hot = [0.0f32; 3];
                    one_hot[j] = facet.side_sign as f32;
                    let rotated =
                        rotate_local_position(animating.local_coords, partial_angle, one_hot);
                    let mut v = [0.0f32; 4];
                    for k in 0..3 {
                        v[axes[k]] = rotated[k];
                    }
                    v
                } else {
                    FACE_CENTERS[facet.face_id].into()
                };

                (position_4d, basis, face_normal_4d)
            } else {
                (facet.position_4d, facet.basis, FACE_CENTERS[facet.face_id].into())
            };

            StickerInstance {
                position_4d,
                color: nalgebra::Vector4::from(color).into(),
                basis,
                face_normal_4d,
            }
        })
        .collect()
}

/// Parameters controlled from the ui.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UiControls {
    pub(crate) sticker_scale: f32,
    pub(crate) face_gap: f32,
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
            self.ui_controls.face_gap,
        );
        pipeline.update_camera(queue, &self.camera, &self.projection);
        pipeline.update_light(queue, &self.camera);
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
    hovered_sticker: Option<usize>,
    debug_instances: Vec<DebugInstanceWithDistance>,
    hypercube: Hypercube,
    animating_move: Option<AnimatingMove>,
    animating_focus: Option<AnimatingFocus>,
    /// Position and hovered sticker (if any) recorded when the rotate button
    /// was last pressed, used at release time to tell a click from a drag.
    rotate_press: Option<(Point, Option<usize>)>,
    /// Time and face of an unmatched first click on the rotate button,
    /// waiting to see if a second click lands within `DOUBLE_CLICK_WINDOW`.
    pending_face_click: Option<(Instant, usize)>,
    last_redraw_instant: Option<Instant>,
    reset_generation: u64,
}

/// The shader program that handles 4D hypercube rendering
pub(crate) struct HypercubeShaderProgram {
    sticker_scale: f32,
    face_gap: f32,
    render_mode: RenderMode,
    aabb_mode: AABBMode,
    rotate_button: RotateButton,
    animation_duration_ms: u32,
    reset_generation: u64,
}

impl HypercubeShaderProgram {
    /// Create a new shader program with the given parameters
    pub(crate) fn new(
        sticker_scale: f32,
        face_gap: f32,
        render_mode: RenderMode,
        aabb_mode: AABBMode,
        rotate_button: RotateButton,
        animation_duration_ms: u32,
        reset_generation: u64,
    ) -> Self {
        Self {
            sticker_scale,
            face_gap,
            render_mode,
            aabb_mode,
            rotate_button,
            animation_duration_ms,
            reset_generation,
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
        if self.reset_generation != state.reset_generation {
            state.hypercube = Hypercube::solved();
            state.animating_move = None;
            state.animating_focus = None;
            state.rotate_press = None;
            state.pending_face_click = None;
            state.last_redraw_instant = None;
            state.hovered_sticker = None;
            state.debug_instances.clear();
            state.reset_generation = self.reset_generation;
            return Some(Action::request_redraw());
        }

        // Update camera each frame
        state.camera_controller.update_camera(&mut state.camera);

        // Update viewport size if bounds changed
        if bounds.width > 0.0 && bounds.height > 0.0 {
            state.projection.aspect = bounds.width / bounds.height;
        }

        // Check if 4D rotation changed and recalculate indices
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
                let delta = state
                    .last_redraw_instant
                    .map(|last| now.duration_since(last))
                    .unwrap_or_default();

                let move_tick = Self::advance_animation(state, delta);
                let focus_tick = Self::advance_focus_animation(state, delta);

                if state.animating_move.is_none() && state.animating_focus.is_none() {
                    state.last_redraw_instant = None;
                } else {
                    state.last_redraw_instant = Some(*now);
                }

                if matches!(
                    focus_tick,
                    AnimationTick::Running | AnimationTick::Completed
                ) {
                    rotation_changed = true;
                }

                if matches!(
                    (&move_tick, &focus_tick),
                    (AnimationTick::Completed, _) | (_, AnimationTick::Completed)
                ) && !state.mouse_pressed
                    && let Some(position) = cursor.position_in(bounds)
                {
                    self.update_hover(state, position, bounds);
                }

                if matches!(move_tick, AnimationTick::Ignored)
                    && matches!(focus_tick, AnimationTick::Ignored)
                {
                    event::Status::Ignored
                } else {
                    event::Status::Captured
                }
            }
            _ => event::Status::Ignored,
        };

        // Recalculate indices if rotation changed
        if rotation_changed {
            state.cached_indices = Self::calculate_indices(&state.rotation_4d);
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
                face_gap: self.face_gap,
                render_mode: self.render_mode,
            },
            cached_indices: state.cached_indices.clone(),
            hovered_sticker: state.hovered_sticker,
            debug_instances: state.debug_instances.clone(),
            sticker_instances: sticker_instances_for_render(state),
        }
    }
}

impl HypercubeShaderProgram {
    /// Calculate the winding-corrected index buffer for all cube faces after
    /// 4D transformation and 3D projection. Shading normals are computed
    /// directly in the vertex shader from each instance's own basis instead
    /// (see `compute_world_normal` in shader.wgsl/normal_shader.wgsl).
    fn calculate_indices(rotation_4d: &nalgebra::Matrix4<f32>) -> Vec<u16> {
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

                indices.extend(triangle_indices);
            }
        }

        indices
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
                            // 4D rotation, camera-relative
                            let (right, up) = state.camera.right_and_up();
                            state.rotation_4d = process_4d_rotation(
                                &state.rotation_4d,
                                delta_x,
                                delta_y,
                                right,
                                up,
                            );
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
                    self.update_hover(state, position, bounds);
                }

                state.last_mouse_pos = Some(position);
                return event::Status::Captured;
            }
            mouse::Event::ButtonPressed(button) => {
                if let Some(position) = cursor.position_in(bounds)
                    && *button == self.rotate_button.to_mouse_button()
                {
                    // A fresh press always takes precedence over an
                    // in-progress auto-centering animation, so a deliberate
                    // drag never has to fight it frame-by-frame. This only
                    // ever cancels an animation from an *earlier*
                    // interaction: the animation this same press might
                    // trigger doesn't start until its matching release.
                    state.animating_focus = None;
                    state.rotate_press = Some((position, state.hovered_sticker));
                    state.mouse_pressed = true;
                    return event::Status::Captured;
                }
                if cursor.position_in(bounds).is_some()
                    && *button == self.rotate_button.click_button()
                    && state.animating_move.is_none()
                    && state.animating_focus.is_none()
                    && let Some(sticker_index) = state.hovered_sticker
                {
                    self.handle_facet_click(state, sticker_index);
                    return event::Status::Captured;
                }
            }
            mouse::Event::ButtonReleased(_) => {
                let was_dragging = state.mouse_pressed;
                if was_dragging {
                    state.mouse_pressed = false;
                }

                if let Some((press_pos, sticker_at_press)) = state.rotate_press.take() {
                    self.handle_rotate_click(
                        state,
                        press_pos,
                        cursor.position_in(bounds),
                        sticker_at_press,
                    );
                    return event::Status::Captured;
                }

                if was_dragging {
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

    /// Casts a ray from the given position and updates `hovered_sticker` and
    /// `debug_instances` on `state` with the result.
    fn update_hover(&self, state: &mut HypercubeShaderState, position: Point, bounds: Rectangle) {
        let mouse_ray = calculate_mouse_ray(position, bounds, &state.camera, &state.projection);

        let (hovered_sticker, debug_instances) = find_intersected_sticker(
            &mouse_ray,
            state,
            self.sticker_scale,
            self.face_gap,
            VIEWER_DISTANCE,
            self.aabb_mode,
        );
        state.hovered_sticker = hovered_sticker;
        state.debug_instances = debug_instances;
    }

    /// Resolves a rotate-button release into a click-vs-drag decision and,
    /// for a qualifying click, double-click detection. `press_pos`/
    /// `sticker_at_press` were captured when the button went down;
    /// `release_pos` is the cursor position now (`None` if it left the
    /// widget bounds). A release that isn't a same-spot click on a sticker
    /// (a real drag, or landing off any sticker) always clears any pending
    /// click, matching standard double-click behavior.
    fn handle_rotate_click(
        &self,
        state: &mut HypercubeShaderState,
        press_pos: Point,
        release_pos: Option<Point>,
        sticker_at_press: Option<usize>,
    ) {
        let is_click = release_pos.is_some_and(|release_pos| {
            let dx = release_pos.x - press_pos.x;
            let dy = release_pos.y - press_pos.y;
            (dx * dx + dy * dy).sqrt() < CLICK_DRAG_THRESHOLD_PX
        });

        let Some(sticker_index) = sticker_at_press.filter(|_| is_click) else {
            state.pending_face_click = None;
            return;
        };

        let face_id = FACET_TABLE[sticker_index].face_id;
        let now = Instant::now();
        let is_double_click = state
            .pending_face_click
            .is_some_and(|(last_time, last_face)| {
                last_face == face_id && now.duration_since(last_time) <= DOUBLE_CLICK_WINDOW
            });

        if is_double_click && state.animating_move.is_none() && state.animating_focus.is_none() {
            self.start_focus_animation(state, face_id);
            state.pending_face_click = None;
        } else {
            state.pending_face_click = Some((now, face_id));
        }
    }

    /// Starts an animation that reorients the puzzle in 4D so `face_id`'s
    /// normal ends up centered and facing the viewer. The target is
    /// `FACE_CENTERS[0]` (W=-1), not `FACE_CENTERS[7]` (W=+1): the latter is
    /// exactly the pole `is_face_visible` culls (see `math.rs`'s doc
    /// comment), so aiming there would make the double-clicked face vanish
    /// instead of centering it.
    fn start_focus_animation(&self, state: &mut HypercubeShaderState, face_id: usize) {
        let target = FACE_CENTERS[0];
        let current_normal = (state.rotation_4d * FACE_CENTERS[face_id]).normalize();
        let (u, v, total_angle) = shortest_arc_plane(current_normal, target);

        state.animating_focus = Some(AnimatingFocus {
            start_rotation: state.rotation_4d,
            plane: (u, v),
            total_angle,
            elapsed: Duration::ZERO,
            duration: Duration::from_millis(self.animation_duration_ms as u64),
        });
        state.last_redraw_instant = None;
    }

    /// Applies the move triggered by clicking the given facet, if any -
    /// non-actionable facets (cell-centers, the invisible center) are a
    /// no-op. A plain click always turns clockwise as viewed from beyond the
    /// clicked facet, looking back in along its own rotation axis
    /// (`moves::clockwise_sign`) - independent of the puzzle's current
    /// orientation or camera position. Shift reverses it to
    /// counterclockwise.
    fn handle_facet_click(&self, state: &mut HypercubeShaderState, sticker_index: usize) {
        let facet = &FACET_TABLE[sticker_index];
        if !facet.is_actionable {
            return;
        }

        let local_nonzero_count = facet.local_coords.iter().filter(|c| **c != 0).count();
        let magnitude = base_angle(local_nonzero_count);

        let sign = clockwise_sign(facet);
        let angle = if state.shift_pressed {
            -sign * magnitude
        } else {
            sign * magnitude
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
    fn advance_animation(state: &mut HypercubeShaderState, delta: Duration) -> AnimationTick {
        let Some(animating) = state.animating_move.as_mut() else {
            return AnimationTick::Ignored;
        };

        animating.elapsed += delta;

        if animating.elapsed >= animating.duration {
            state.animating_move = None;
            return AnimationTick::Completed;
        }

        AnimationTick::Running
    }

    /// Advances an in-progress "center this face" animation (see
    /// `AnimatingFocus`) by `delta`, mirroring `advance_animation`'s shape.
    /// Unlike a move animation, this one has an externally visible effect
    /// every tick - it directly drives `rotation_4d` - rather than only
    /// affecting a separate per-frame sweep computation.
    fn advance_focus_animation(state: &mut HypercubeShaderState, delta: Duration) -> AnimationTick {
        let Some(animating) = state.animating_focus.as_mut() else {
            return AnimationTick::Ignored;
        };

        animating.elapsed += delta;

        let t = if animating.duration.is_zero() {
            1.0
        } else {
            (animating.elapsed.as_secs_f32() / animating.duration.as_secs_f32()).clamp(0.0, 1.0)
        };
        let (u, v) = animating.plane;
        let angle = animating.total_angle * ease(t);
        state.rotation_4d = create_4d_plane_rotation(u, v, angle) * animating.start_rotation;

        if animating.elapsed >= animating.duration {
            state.animating_focus = None;
            return AnimationTick::Completed;
        }

        AnimationTick::Running
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
            fovy: std::f32::consts::FRAC_PI_4,
            znear: 0.1,
            zfar: 100.0,
        };

        let rotation_4d = nalgebra::Matrix4::identity();
        let cached_indices = HypercubeShaderProgram::calculate_indices(&rotation_4d);

        Self {
            camera,
            camera_controller,
            projection,
            rotation_4d,
            mouse_pressed: false,
            last_mouse_pos: None,
            shift_pressed: false,
            cached_indices,
            hovered_sticker: None,
            debug_instances: Vec::new(),
            hypercube: Hypercube::solved(),
            animating_move: None,
            animating_focus: None,
            rotate_press: None,
            pending_face_click: None,
            last_redraw_instant: None,
            reset_generation: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::FACE_CENTERS;
    use iced::widget::shader::Program;

    fn round_key(v: [f32; 4]) -> [i32; 4] {
        v.map(|x| (x * 1000.0).round() as i32)
    }

    fn color_key(c: [f32; 4]) -> [u8; 4] {
        c.map(|x| (x * 255.0).round() as u8)
    }

    /// At the end of a move, a rotated basis vector is `±` some world unit
    /// vector, matching `discrete_rotation`'s signed-permutation snap - but
    /// unlike position, the *sign* isn't independently meaningful here: the
    /// sticker mesh spans `[-s, s]` symmetrically along each of its 3 basis
    /// vectors, so any signed permutation of a facet's basis sweeps out the
    /// exact same rendered point set (e.g. a piece that spins in place on
    /// the turning face's own layer can land on a basis that's a nontrivial
    /// signed permutation of the static identity basis and still be
    /// pixel-identical, since the mesh has no per-face markings to reveal
    /// that it rotated). What *is* meaningful, and load-bearing for the
    /// sticker to render on the correct face at all, is which 3 of the 4
    /// world axes end up spanned - reduce to that set, dropping sign and
    /// order, for comparisons against the static post-move basis.
    fn basis_axis_set(basis: [[f32; 4]; 3]) -> Vec<usize> {
        let mut axes: Vec<usize> = basis
            .iter()
            .map(|v| {
                let (axis, value) = v
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
                    .expect("basis vector has 4 components");
                assert!(
                    value.abs() > 0.5,
                    "basis vector isn't close to a signed unit axis: {v:?}"
                );
                axis
            })
            .collect();
        axes.sort_unstable();
        axes
    }

    /// At `partial_angle = 0` (start of a move), the rotated basis must
    /// exactly reproduce the static pre-move basis - `rotate_local_position`
    /// at angle 0 is the identity, so this should hold bit-for-bit.
    #[test]
    fn animated_basis_matches_static_basis_at_start_of_move() {
        for side_axis in 0..4usize {
            for side_sign in [-1i8, 1] {
                for local_coords in [[1i8, 0, 0], [1, 1, 0], [1, 1, 1]] {
                    let nonzero = local_coords.iter().filter(|c| **c != 0).count();
                    let angle = base_angle(nonzero);

                    let pre_move = Hypercube::solved();
                    let state = HypercubeShaderState {
                        hypercube: pre_move.clone(),
                        animating_move: Some(AnimatingMove {
                            side_axis,
                            side_sign,
                            local_coords,
                            angle,
                            pre_move_pieces: pre_move.pieces.clone(),
                            elapsed: Duration::ZERO,
                            duration: Duration::from_millis(250),
                        }),
                        ..Default::default()
                    };

                    let instances = sticker_instances_for_render(&state);
                    for (facet, instance) in FACET_TABLE.iter().zip(instances.iter()) {
                        assert_eq!(
                            instance.basis, facet.basis,
                            "mismatch for side_axis={side_axis} side_sign={side_sign} \
                             local_coords={local_coords:?} piece_slot={} axis={}",
                            facet.piece_slot, facet.axis
                        );
                    }
                }
            }
        }
    }

    /// At `partial_angle = 0` (start of a move), every instance's
    /// `face_normal_4d` - the vector the shader culls against - must exactly
    /// reproduce the static pre-move `FACE_CENTERS[face_id]`, the same way
    /// `basis` does above.
    #[test]
    fn animated_face_normal_matches_static_face_center_at_start_of_move() {
        for side_axis in 0..4usize {
            for side_sign in [-1i8, 1] {
                for local_coords in [[1i8, 0, 0], [1, 1, 0], [1, 1, 1]] {
                    let nonzero = local_coords.iter().filter(|c| **c != 0).count();
                    let angle = base_angle(nonzero);

                    let pre_move = Hypercube::solved();
                    let state = HypercubeShaderState {
                        hypercube: pre_move.clone(),
                        animating_move: Some(AnimatingMove {
                            side_axis,
                            side_sign,
                            local_coords,
                            angle,
                            pre_move_pieces: pre_move.pieces.clone(),
                            elapsed: Duration::ZERO,
                            duration: Duration::from_millis(250),
                        }),
                        ..Default::default()
                    };

                    let instances = sticker_instances_for_render(&state);
                    for (facet, instance) in FACET_TABLE.iter().zip(instances.iter()) {
                        let expected: [f32; 4] = FACE_CENTERS[facet.face_id].into();
                        assert_eq!(
                            instance.face_normal_4d, expected,
                            "mismatch for side_axis={side_axis} side_sign={side_sign} \
                             local_coords={local_coords:?} piece_slot={} axis={}",
                            facet.piece_slot, facet.axis
                        );
                    }
                }
            }
        }
    }

    /// Position, color, spanned basis axes, and face normal for one rendered
    /// row, used to compare animated vs. static render output as a set.
    type RenderRow = ([i32; 4], [u8; 4], Vec<usize>, [i32; 4]);

    /// At the end of an animation, the full set of rendered (position,
    /// color) pairs must exactly match what the static post-move render
    /// would show - checked as a set (not a row-by-row comparison), since
    /// each animated row keeps its pre-move identity while sweeping to
    /// wherever its content ends up, which is a different GPU row than the
    /// static render uses for the same visual result.
    #[test]
    fn animated_end_state_matches_post_move_static_render_for_all_move_types() {
        for side_axis in 0..4usize {
            for side_sign in [-1i8, 1] {
                for local_coords in [
                    [1i8, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1],
                ] {
                    for direction in [1i8, -1] {
                        let nonzero = local_coords.iter().filter(|c| **c != 0).count();
                        let angle = base_angle(nonzero) * direction as f32;

                        let pre_move = Hypercube::solved();
                        let mut post_move = pre_move.clone();
                        post_move.apply_move(side_axis, side_sign, local_coords, angle);

                        let state = HypercubeShaderState {
                            hypercube: pre_move.clone(),
                            animating_move: Some(AnimatingMove {
                                side_axis,
                                side_sign,
                                local_coords,
                                angle,
                                pre_move_pieces: pre_move.pieces.clone(),
                                elapsed: Duration::from_millis(250),
                                duration: Duration::from_millis(250),
                            }),
                            ..Default::default()
                        };

                        let mut animated_end: Vec<RenderRow> = sticker_instances_for_render(&state)
                            .iter()
                            .map(|inst| {
                                (
                                    round_key(inst.position_4d),
                                    color_key(inst.color),
                                    basis_axis_set(inst.basis),
                                    round_key(inst.face_normal_4d),
                                )
                            })
                            .collect();
                        let mut static_post: Vec<RenderRow> =
                            generate_sticker_instances(&post_move)
                                .iter()
                                .map(|inst| {
                                    (
                                        round_key(inst.position_4d),
                                        color_key(inst.color),
                                        basis_axis_set(inst.basis),
                                        round_key(inst.face_normal_4d),
                                    )
                                })
                                .collect();
                        animated_end.sort_unstable();
                        static_post.sort_unstable();

                        assert_eq!(
                            animated_end, static_post,
                            "mismatch for side_axis={side_axis} side_sign={side_sign} \
                             local_coords={local_coords:?} direction={direction}"
                        );
                    }
                }
            }
        }
    }

    /// The set-based checks above can't catch a wrong basis on one row being
    /// masked by another row that legitimately has the same spanned axis
    /// set (face-swapping facets sharing a move come in groups). This pins
    /// down the one thing that's actually load-bearing per facet: at the end
    /// of a move, a facet whose own axis isn't `side_axis` swaps onto
    /// whichever new axis `apply_move`'s permutation sends it to - checked
    /// directly against `discrete_rotation` (already covered by its own
    /// tests in `moves.rs`), independent of the position/color machinery.
    #[test]
    fn animated_basis_flat_direction_matches_new_facet_axis_at_end_of_move() {
        use crate::moves::discrete_rotation;

        for side_axis in 0..4usize {
            for side_sign in [-1i8, 1] {
                for local_coords in [
                    [1i8, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1],
                ] {
                    for direction in [1i8, -1] {
                        let nonzero = local_coords.iter().filter(|c| **c != 0).count();
                        let angle = base_angle(nonzero) * direction as f32;
                        let axes = free_axes(side_axis);
                        let (perm, _sign) = discrete_rotation(local_coords, angle);
                        let mut inv_perm = [0usize; 3];
                        for slot in 0..3 {
                            inv_perm[perm[slot]] = slot;
                        }

                        let pre_move = Hypercube::solved();
                        let state = HypercubeShaderState {
                            hypercube: pre_move.clone(),
                            animating_move: Some(AnimatingMove {
                                side_axis,
                                side_sign,
                                local_coords,
                                angle,
                                pre_move_pieces: pre_move.pieces.clone(),
                                elapsed: Duration::from_millis(250),
                                duration: Duration::from_millis(250),
                            }),
                            ..Default::default()
                        };

                        let instances = sticker_instances_for_render(&state);
                        for (facet, instance) in FACET_TABLE.iter().zip(instances.iter()) {
                            // Facets on the turning face's own layer
                            // (`facet.axis == side_axis`) genuinely spin in
                            // place but their basis stays entirely within
                            // `axes`, spanning the same set regardless of
                            // rotation - nothing to pin down there, covered
                            // by the symmetric-mesh reasoning above instead.
                            if facet.axis == side_axis {
                                continue;
                            }
                            if pre_move.pieces[facet.piece_slot].position[side_axis] != side_sign {
                                continue;
                            }
                            let p = axes
                                .iter()
                                .position(|&x| x == facet.axis)
                                .expect("facet.axis != side_axis must be one of axes");
                            let new_axis = axes[inv_perm[p]];
                            let spanned = basis_axis_set(instance.basis);
                            assert!(
                                !spanned.contains(&new_axis),
                                "facet piece_slot={} axis={} should have swapped flat \
                                 direction onto axis {new_axis}, but basis still spans it: \
                                 {spanned:?} (side_axis={side_axis} side_sign={side_sign} \
                                 local_coords={local_coords:?} direction={direction})",
                                facet.piece_slot,
                                facet.axis
                            );
                        }
                    }
                }
            }
        }
    }

    /// A bumped `reset_generation` must resolve the puzzle back to solved,
    /// cancel any in-progress move animation, and request a redraw - the
    /// mechanism a "Reset" button relies on to reach state owned by the
    /// shader widget's `Program::State`.
    #[test]
    fn reset_generation_mismatch_resets_hypercube_and_cancels_animation() {
        let mut state = HypercubeShaderState::default();
        assert_eq!(state.reset_generation, 0);

        let facet = FACET_TABLE
            .iter()
            .find(|f| f.is_actionable)
            .expect("at least one actionable facet exists");
        let nonzero = facet.local_coords.iter().filter(|c| **c != 0).count();
        let angle = base_angle(nonzero);
        let pre_move_pieces = state.hypercube.pieces.clone();
        state
            .hypercube
            .apply_move(facet.axis, facet.side_sign, facet.local_coords, angle);
        assert!(!state.hypercube.is_solved());
        state.animating_move = Some(AnimatingMove {
            side_axis: facet.axis,
            side_sign: facet.side_sign,
            local_coords: facet.local_coords,
            angle,
            pre_move_pieces,
            elapsed: Duration::ZERO,
            duration: Duration::from_millis(250),
        });

        let program = HypercubeShaderProgram::new(
            0.5,
            2.0,
            RenderMode::Standard,
            AABBMode::None,
            RotateButton::default(),
            250,
            1,
        );

        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));
        let action = program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );

        assert!(action.is_some(), "reset must request a redraw");
        assert!(state.hypercube.is_solved());
        assert!(state.animating_move.is_none());
        assert_eq!(state.reset_generation, 1);
    }
}

#[cfg(test)]
mod clockwise_sign_tests {
    use super::*;
    use crate::math::project_4d_to_3d;
    use crate::moves::clockwise_sign;

    /// For every actionable facet, `moves::clockwise_sign` must agree with
    /// an oracle built independently of the cofactor formula it uses
    /// internally: two probes `p1`, `p2 = axis x p1` (perpendicular to the
    /// rotation axis) have true rendered velocities `v1`, `v2` under a small
    /// rotation, and `v1 x v2` is the true spin axis, measured with no axis
    /// transform formula at all - just how two points actually move.
    ///
    /// Clockwise (as viewed from beyond the facet, looking back in) means
    /// that true spin axis points away from a viewer standing further out
    /// along the same ray the facet sits on - the direction given by
    /// transforming `local_coords` as an ordinary vector, not a pseudovector.
    #[test]
    fn clockwise_sign_matches_velocity_oracle() {
        let rotation_4d = Matrix4::identity();

        for facet in FACET_TABLE.iter().filter(|f| f.is_actionable) {
            let position_4d = Vector4::from(facet.position_4d);
            let base = project_4d_to_3d(position_4d, &rotation_4d, VIEWER_DISTANCE);

            const EPSILON: f32 = 1e-3;
            let tangent = |axis: usize| -> Vector3<f32> {
                let mut offset_4d = position_4d;
                offset_4d[axis] += EPSILON;
                (project_4d_to_3d(offset_4d, &rotation_4d, VIEWER_DISTANCE) - base) / EPSILON
            };
            let local_coords = facet.local_coords.map(|c| c as f32);
            let d = facet.free_axes.map(tangent);
            let ordinary_direction =
                d[0] * local_coords[0] + d[1] * local_coords[1] + d[2] * local_coords[2];

            let axis_unit =
                Vector3::new(local_coords[0], local_coords[1], local_coords[2]).normalize();
            let candidate = if axis_unit.x.abs() < 0.9 {
                Vector3::x()
            } else {
                Vector3::y()
            };
            let p1 = (candidate - axis_unit * candidate.dot(&axis_unit)).normalize();
            let p2 = axis_unit.cross(&p1).normalize();

            const ANGLE_EPSILON: f32 = 1e-5;
            let velocity = |p: Vector3<f32>| -> Vector3<f32> {
                let pre =
                    project_cube_point(p, position_4d, facet.axis, &rotation_4d, VIEWER_DISTANCE);
                let rotated = rotate_local_position(facet.local_coords, ANGLE_EPSILON, p.into());
                let post = project_cube_point(
                    Vector3::from(rotated),
                    position_4d,
                    facet.axis,
                    &rotation_4d,
                    VIEWER_DISTANCE,
                );
                (post - pre) / ANGLE_EPSILON
            };
            let ground_truth_axis = velocity(p1).cross(&velocity(p2));

            let expected_sign = if ground_truth_axis.dot(&ordinary_direction) > 0.0 {
                -1.0
            } else {
                1.0
            };
            let actual_sign = clockwise_sign(facet);
            assert_eq!(
                actual_sign, expected_sign,
                "facet piece_slot={} axis={} side_sign={} local_coords={:?}: \
                 clockwise_sign returned {actual_sign}, oracle expected {expected_sign}",
                facet.piece_slot, facet.axis, facet.side_sign, facet.local_coords
            );
        }
    }
}
