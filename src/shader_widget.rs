//! Custom shader widget for 4D hypercube rendering.
//!
//! This module implements the shader widget that encapsulates all 3D rendering
//! logic, camera controls, and 4D transformations. It follows Option C architecture
//! where the shader widget manages its own state independently.

use std::sync::Arc;
use std::time::{Duration, Instant};

use iced::wgpu;
use iced::widget::{Action, shader};
use iced::{Event, Point, Rectangle, event, mouse};
use nalgebra::{Matrix4, UnitQuaternion, Vector3, Vector4};

use crate::app::{AABBMode, Message, RenderMode};
use crate::camera::{Camera, CameraController, Projection};
use crate::geometry::{
    BASE_CUBE_VERTICES, FACE_CENTERS, FIXED_DIMS, NORMAL_TO_BASE_INDICES, VERTEX_NORMAL_INDICES,
};
use crate::math::{
    GRID_EXTENT, VIEWER_DISTANCE, compose_so4, create_4d_plane_rotation, decompose_so4,
    process_4d_rotation, project_cube_point, quat_slerp_exact, shortest_arc_plane, visible_faces,
};
use crate::moves::{base_angle, clockwise_sign, rotate_local_position};
use crate::piece::{
    FACET_TABLE, Hypercube, Piece, StickerInstance, free_axes, generate_sticker_instances,
};
use crate::ray_casting::{calculate_mouse_ray, find_intersected_sticker};
use crate::renderer::{DebugInstanceWithDistance, Renderer};
use crate::settings::RotateButton;

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

/// An in-progress "return to default orientation" animation, triggered by
/// Reset: slerps the isoclinic quaternion pair (see `math::decompose_so4`)
/// describing `rotation_4d` when Reset was pressed toward the identity
/// quaternion pair, recomposing `rotation_4d` each tick. Unlike
/// `AnimatingFocus`, which rotates in a single plane, this can undo an
/// arbitrary accumulated 4D orientation.
struct AnimatingReset {
    start_p: UnitQuaternion<f32>,
    start_q: UnitQuaternion<f32>,
    elapsed: Duration,
    duration: Duration,
}

/// An in-progress reveal/hide flourish: sweeps sticker scale, face gap, and
/// camera yaw from their values when the toggle button was pressed toward
/// the reveal's (or hide's) target values, all driven by the same `elapsed`/
/// `duration`/`ease` progress.
struct AnimatingReveal {
    start_scale: f32,
    target_scale: f32,
    start_gap: f32,
    target_gap: f32,
    start_yaw: f32,
    target_yaw: f32,
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

/// Duration of the reveal/hide flourish (camera spin + scale/gap animation),
/// independent of `animation_duration_ms` which is tuned for quick move/focus
/// animations rather than a two-revolution camera spin.
pub(crate) const REVEAL_ANIMATION_DURATION: Duration = Duration::from_millis(2500);
/// Camera yaw delta applied by a single reveal or hide flourish. A multiple
/// of 360 degrees, so the camera always visually ends up where it started.
const REVEAL_YAW_SPIN_DEGREES: f32 = 720.0;
/// Sticker scale/face gap in the app's raw (slider) domain before a reveal.
/// Shared with `app.rs::HypercubeApp::new()` as the single source of truth.
pub(crate) const PRIMARY_STICKER_SCALE: f32 = 0.02;
pub(crate) const PRIMARY_FACE_GAP: f32 = 0.0;
/// Sticker scale/face gap in the app's raw (slider) domain a reveal animates
/// toward.
const SECONDARY_STICKER_SCALE: f32 = 0.4;
const SECONDARY_FACE_GAP: f32 = 1.5;

/// Builds the GPU instance list for the current frame. Piece state is
/// already final (`apply_move` commits atomically) - while a move is
/// animating, the 27 affected facets are instead swept from their pre-move
/// position/color toward that already-committed final position, using the
/// exact same rotation formula `apply_move` used, so the last animated
/// frame always lines up perfectly with the static post-move render it
/// hands off to.
pub fn sticker_instances_for_render(state: &HypercubeShaderState) -> Vec<StickerInstance> {
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
    pub(crate) cached_indices: Arc<[u16]>,
    pub(crate) indices_generation: u64,
    pub(crate) hovered_sticker: Option<usize>,
    pub(crate) debug_instances: Vec<DebugInstanceWithDistance>,
    pub(crate) sticker_instances: Arc<[StickerInstance]>,
    pub(crate) sticker_generation: u64,
    pub(crate) visible_faces: [bool; 8],
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
        pipeline.update_indices(queue, &self.cached_indices, self.indices_generation);
        pipeline.update_highlighting(queue, self.hovered_sticker);
        pipeline.update_debug_instances(queue, &self.debug_instances);
        pipeline.update_sticker_instances(queue, &self.sticker_instances, self.sticker_generation);
        pipeline.set_render_mode(self.ui_controls.render_mode);
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        _clip_bounds: &Rectangle<u32>,
    ) {
        pipeline.render(encoder, target, &self.visible_faces);

        // Render transparent debug AABBs
        pipeline.render_debug_aabb(encoder, target, self.debug_instances.len() as u32);
    }
}

/// Internal state managed by the shader widget
pub struct HypercubeShaderState {
    pub(crate) camera: Camera,
    camera_controller: CameraController,
    projection: Projection,
    pub(crate) rotation_4d: nalgebra::Matrix4<f32>,
    mouse_pressed: bool,
    last_mouse_pos: Option<Point>,
    shift_pressed: bool,
    cached_indices: Arc<[u16]>,
    /// Bumped every time `cached_indices` is replaced; carried on
    /// `HypercubePrimitive` so `Renderer` can skip re-uploading the index
    /// buffer to the GPU when it hasn't actually changed since last frame.
    indices_generation: u64,
    cached_sticker_instances: Arc<[StickerInstance]>,
    /// Bumped every time `cached_sticker_instances` is replaced; same
    /// upload-skipping purpose as `indices_generation`.
    sticker_generation: u64,
    hovered_sticker: Option<usize>,
    debug_instances: Vec<DebugInstanceWithDistance>,
    hypercube: Hypercube,
    animating_move: Option<AnimatingMove>,
    animating_focus: Option<AnimatingFocus>,
    animating_reset: Option<AnimatingReset>,
    animating_reveal: Option<AnimatingReveal>,
    /// Live sticker scale/face gap while a reveal/hide flourish is playing,
    /// consulted by `draw()`/`update_hover` in preference to
    /// `HypercubeShaderProgram`'s own fields. Self-corrects back to `None`
    /// once `HypercubeApp` has caught up to the animation's final value (see
    /// `Program::update`), so it never masks a later manual slider drag.
    reveal_scale_override: Option<f32>,
    reveal_gap_override: Option<f32>,
    /// Position and hovered sticker (if any) recorded when the rotate button
    /// was last pressed, used at release time to tell a click from a drag.
    rotate_press: Option<(Point, Option<usize>)>,
    /// Time and face of an unmatched first click on the rotate button,
    /// waiting to see if a second click lands within `DOUBLE_CLICK_WINDOW`.
    pending_face_click: Option<(Instant, usize)>,
    last_redraw_instant: Option<Instant>,
    reset_generation: u64,
    random_moves_generation: u64,
    /// Seeded once at construction, reused across every random-move press so
    /// results are reproducible from a fixed seed in tests but still vary
    /// run-to-run in the live app (`Rng::new()` seeds from OS entropy).
    rng: fastrand::Rng,
    reveal_generation: u64,
}

impl HypercubeShaderState {
    /// Replaces `cached_indices` and bumps `indices_generation`, so
    /// `Renderer::update_indices` can tell this frame's data apart from
    /// what's already on the GPU.
    fn set_cached_indices(&mut self, indices: Vec<u16>) {
        self.cached_indices = indices.into();
        self.indices_generation += 1;
    }

    /// Replaces `cached_sticker_instances` and bumps `sticker_generation`,
    /// mirroring `set_cached_indices`.
    fn set_cached_sticker_instances(&mut self, instances: Vec<StickerInstance>) {
        self.cached_sticker_instances = instances.into();
        self.sticker_generation += 1;
    }
}

/// The shader program that handles 4D hypercube rendering
pub struct HypercubeShaderProgram {
    sticker_scale: f32,
    face_gap: f32,
    render_mode: RenderMode,
    aabb_mode: AABBMode,
    rotate_button: RotateButton,
    animation_duration_ms: u32,
    reset_generation: u64,
    random_moves_generation: u64,
    random_move_count: u32,
    reveal_generation: u64,
    revealed_target: bool,
}

impl HypercubeShaderProgram {
    /// Create a new shader program with the given parameters
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        sticker_scale: f32,
        face_gap: f32,
        render_mode: RenderMode,
        aabb_mode: AABBMode,
        rotate_button: RotateButton,
        animation_duration_ms: u32,
        reset_generation: u64,
        random_moves_generation: u64,
        random_move_count: u32,
        reveal_generation: u64,
        revealed_target: bool,
    ) -> Self {
        Self {
            sticker_scale,
            face_gap,
            render_mode,
            aabb_mode,
            rotate_button,
            animation_duration_ms,
            reset_generation,
            random_moves_generation,
            random_move_count,
            reveal_generation,
            revealed_target,
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
            state.hovered_sticker = None;
            state.debug_instances.clear();
            state.reset_generation = self.reset_generation;

            let (start_p, start_q) = decompose_so4(&state.rotation_4d);
            state.animating_reset = Some(AnimatingReset {
                start_p,
                start_q,
                elapsed: Duration::ZERO,
                duration: Duration::from_millis(self.animation_duration_ms as u64),
            });
            state.last_redraw_instant = None;

            let instances = sticker_instances_for_render(state);
            state.set_cached_sticker_instances(instances);
            return Some(Action::request_redraw());
        }

        if self.random_moves_generation != state.random_moves_generation {
            state
                .hypercube
                .apply_random_moves(self.random_move_count, &mut state.rng);
            state.animating_move = None;
            state.animating_focus = None;
            state.rotate_press = None;
            state.pending_face_click = None;
            state.last_redraw_instant = None;
            state.hovered_sticker = None;
            state.debug_instances.clear();
            state.random_moves_generation = self.random_moves_generation;
            let instances = sticker_instances_for_render(state);
            state.set_cached_sticker_instances(instances);
            return Some(Action::request_redraw());
        }

        // Once `HypercubeApp` has caught up to a completed reveal/hide
        // flourish's final value (via `Message::RevealAnimationComplete`),
        // the fresh `self` built from it matches the override's stored
        // target - clear it so future manual slider drags aren't masked.
        if let Some(target) = state.reveal_scale_override
            && self.sticker_scale == target
        {
            state.reveal_scale_override = None;
        }
        if let Some(target) = state.reveal_gap_override
            && self.face_gap == target
        {
            state.reveal_gap_override = None;
        }

        if self.reveal_generation != state.reveal_generation {
            state.animating_move = None;
            state.animating_focus = None;

            let (target_scale_raw, target_gap) = if self.revealed_target {
                (SECONDARY_STICKER_SCALE, SECONDARY_FACE_GAP)
            } else {
                (PRIMARY_STICKER_SCALE, PRIMARY_FACE_GAP)
            };
            let start_yaw = state.camera_controller.yaw;

            state.animating_reveal = Some(AnimatingReveal {
                start_scale: self.sticker_scale,
                target_scale: 1.0 - target_scale_raw,
                start_gap: self.face_gap,
                target_gap,
                start_yaw,
                target_yaw: start_yaw + REVEAL_YAW_SPIN_DEGREES,
                elapsed: Duration::ZERO,
                duration: REVEAL_ANIMATION_DURATION,
            });
            state.reveal_scale_override = Some(self.sticker_scale);
            state.reveal_gap_override = Some(self.face_gap);
            state.reveal_generation = self.reveal_generation;
            state.last_redraw_instant = None;
            let instances = sticker_instances_for_render(state);
            state.set_cached_sticker_instances(instances);
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
        // Whether `sticker_instances_for_render` needs to run again this
        // tick: true whenever a move animation started, ended, or is still
        // in progress (its swept position changes every tick), false when
        // `animating_move` was and still is absent.
        let mut regenerate_stickers = false;
        let mut reveal_completed_message: Option<Message> = None;
        let mut reset_completed_message: Option<Message> = None;

        let status = match event {
            Event::Mouse(mouse_event) => {
                let old_rotation = state.rotation_4d;
                let was_animating = state.animating_move.is_some();
                let result = self.handle_mouse_event(state, mouse_event, bounds, cursor);
                if state.rotation_4d != old_rotation {
                    rotation_changed = true;
                }
                if was_animating || state.animating_move.is_some() {
                    regenerate_stickers = true;
                }
                result
            }
            Event::Keyboard(keyboard_event) => self.handle_keyboard_event(state, keyboard_event),
            Event::Window(iced::window::Event::RedrawRequested(now)) => {
                let delta = state
                    .last_redraw_instant
                    .map(|last| now.duration_since(last))
                    .unwrap_or_default();

                let was_animating = state.animating_move.is_some();
                let move_tick = Self::advance_animation(state, delta);
                let focus_tick = Self::advance_focus_animation(state, delta);
                let reset_tick = Self::advance_reset_animation(state, delta);
                let reveal_tick = Self::advance_reveal_animation(state, delta);

                if was_animating || state.animating_move.is_some() {
                    regenerate_stickers = true;
                }

                if state.animating_move.is_none()
                    && state.animating_focus.is_none()
                    && state.animating_reset.is_none()
                    && state.animating_reveal.is_none()
                {
                    state.last_redraw_instant = None;
                } else {
                    state.last_redraw_instant = Some(*now);
                }

                if matches!(
                    focus_tick,
                    AnimationTick::Running | AnimationTick::Completed
                ) || matches!(
                    reset_tick,
                    AnimationTick::Running | AnimationTick::Completed
                ) {
                    rotation_changed = true;
                }

                if matches!(
                    (&move_tick, &focus_tick, &reset_tick, &reveal_tick),
                    (AnimationTick::Completed, _, _, _)
                        | (_, AnimationTick::Completed, _, _)
                        | (_, _, AnimationTick::Completed, _)
                        | (_, _, _, AnimationTick::Completed)
                ) && !state.mouse_pressed
                    && let Some(position) = cursor.position_in(bounds)
                {
                    self.update_hover(state, position, bounds);
                }

                if matches!(reset_tick, AnimationTick::Completed) {
                    reset_completed_message = Some(Message::ResetAnimationComplete);
                }

                if matches!(reveal_tick, AnimationTick::Completed) {
                    let (final_scale, final_gap) = if self.revealed_target {
                        (SECONDARY_STICKER_SCALE, SECONDARY_FACE_GAP)
                    } else {
                        (PRIMARY_STICKER_SCALE, PRIMARY_FACE_GAP)
                    };
                    reveal_completed_message = Some(Message::RevealAnimationComplete {
                        final_scale,
                        final_gap,
                    });
                }

                if matches!(move_tick, AnimationTick::Ignored)
                    && matches!(focus_tick, AnimationTick::Ignored)
                    && matches!(reset_tick, AnimationTick::Ignored)
                    && matches!(reveal_tick, AnimationTick::Ignored)
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
            state.set_cached_indices(Self::calculate_indices(&state.rotation_4d));
        }
        if regenerate_stickers {
            let instances = sticker_instances_for_render(state);
            state.set_cached_sticker_instances(instances);
        }

        if let Some(message) = reset_completed_message.or(reveal_completed_message) {
            return Some(Action::publish(message));
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
                sticker_scale: state.reveal_scale_override.unwrap_or(self.sticker_scale),
                face_gap: state.reveal_gap_override.unwrap_or(self.face_gap),
                render_mode: self.render_mode,
            },
            cached_indices: state.cached_indices.clone(),
            indices_generation: state.indices_generation,
            hovered_sticker: state.hovered_sticker,
            debug_instances: state.debug_instances.clone(),
            sticker_instances: state.cached_sticker_instances.clone(),
            sticker_generation: state.sticker_generation,
            // A move animation can rotate a facet's `face_normal_4d` away
            // from its static `face_id`'s `FACE_CENTERS` direction (see
            // `sticker_instances_for_render`'s `facet_axis_is_free`
            // branch), so the per-`face_id` visibility this is based on
            // can't be trusted while one is in progress - fall back to
            // drawing every face and let the vertex shader's own
            // `is_face_visible` cull per-instance instead.
            visible_faces: if state.animating_move.is_some() {
                [true; 8]
            } else {
                visible_faces(&state.rotation_4d, VIEWER_DISTANCE)
            },
        }
    }
}

impl HypercubeShaderProgram {
    /// Calculate the winding-corrected index buffer for all cube faces after
    /// 4D transformation and 3D projection. Shading normals are computed
    /// directly in the vertex shader from each instance's own basis instead
    /// (see `compute_world_normal` in shader.wgsl/normal_shader.wgsl).
    pub fn calculate_indices(rotation_4d: &nalgebra::Matrix4<f32>) -> Vec<u16> {
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

            let cube_center = transformed_vertices.iter().sum::<Vector3<f32>>() / 8.0;

            // Calculate one normal per cube face (6 faces), each spanning two
            // triangles (6 index slots); the two triangles of a face always
            // share a winding decision since they lie on the same plane.
            for (local_face_idx, mut face_indices) in VERTEX_NORMAL_INDICES
                .as_chunks::<6>()
                .0
                .iter()
                .copied()
                .enumerate()
            {
                let corner = |slot: usize| {
                    transformed_vertices[NORMAL_TO_BASE_INDICES[face_indices[slot] as usize]]
                };

                let v0 = corner(0);
                let v1 = corner(1);
                let v2 = corner(2);

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
                        "Degenerate triangle detected for 4D face {face_idx} cube face {local_face_idx}: vertices {v0:?}, {v1:?}, {v2:?}"
                    );
                    normal = Vector3::new(0.0, 0.0, 1.0);
                }

                // Check winding order: the normal should point away from the
                // cube's own center toward this face's own center, not
                // toward/away from the world origin (the cube's center is at
                // an arbitrary offset from the origin, so that comparison is
                // unrelated to winding).
                //
                // A face's 4 unique corners sit at slots 0, 1, 2, 4 of its
                // 6-slot index chunk (slots 3 and 5 repeat slots 2 and 0 to
                // close the second triangle) — see VERTEX_NORMAL_INDICES /
                // NORMAL_TO_BASE_INDICES.
                let face_center = (corner(0) + corner(1) + corner(2) + corner(4)) / 4.0;
                if normal.dot(&(face_center - cube_center)) < 0.0 {
                    log::debug!(
                        "Bad winding order detected for 4D face {face_idx} cube face {local_face_idx}: normal {normal:?} points inward, flipping"
                    );
                    face_indices.swap(1, 2);
                    face_indices.swap(4, 5);
                }

                indices.extend(face_indices);
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
                            // 4D rotation, camera-relative. Skipped while
                            // `animating_reset` is already driving
                            // `rotation_4d`, rather than fighting it
                            // frame-by-frame.
                            if state.animating_reset.is_none() {
                                let (right, up) = state.camera.right_and_up();
                                state.rotation_4d = process_4d_rotation(
                                    &state.rotation_4d,
                                    delta_x,
                                    delta_y,
                                    right,
                                    up,
                                );
                            }
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
                    && state.animating_reveal.is_none()
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
                    && state.animating_reset.is_none()
                    && state.animating_reveal.is_none()
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
        let sticker_scale = state.reveal_scale_override.unwrap_or(self.sticker_scale);
        let face_gap = state.reveal_gap_override.unwrap_or(self.face_gap);

        let (hovered_sticker, debug_instances) = find_intersected_sticker(
            &mouse_ray,
            state,
            sticker_scale,
            face_gap,
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

        if is_double_click
            && state.animating_move.is_none()
            && state.animating_focus.is_none()
            && state.animating_reset.is_none()
        {
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

    /// Advances an in-progress reset animation (see `AnimatingReset`) by
    /// `delta`, slerping both quaternions toward identity and recomposing
    /// `rotation_4d`, mirroring `advance_focus_animation`'s shape.
    fn advance_reset_animation(state: &mut HypercubeShaderState, delta: Duration) -> AnimationTick {
        let Some(animating) = state.animating_reset.as_mut() else {
            return AnimationTick::Ignored;
        };

        animating.elapsed += delta;

        let t = if animating.duration.is_zero() {
            1.0
        } else {
            (animating.elapsed.as_secs_f32() / animating.duration.as_secs_f32()).clamp(0.0, 1.0)
        };
        let eased = ease(t);
        let identity = UnitQuaternion::identity();
        let p = quat_slerp_exact(animating.start_p, identity, eased);
        let q = quat_slerp_exact(animating.start_q, identity, eased);
        state.rotation_4d = compose_so4(p, q);

        if animating.elapsed >= animating.duration {
            state.rotation_4d = Matrix4::identity();
            state.animating_reset = None;
            return AnimationTick::Completed;
        }

        AnimationTick::Running
    }

    /// Advances an in-progress reveal/hide flourish (see `AnimatingReveal`)
    /// by `delta`. Drives `state.camera_controller.yaw` directly - picked up
    /// automatically by the unconditional `update_camera` call each tick -
    /// and updates the scale/gap overrides `draw()`/`update_hover` consult.
    /// On completion the overrides are left set to the exact target rather
    /// than cleared, so `Program::update`'s reconciliation check can drop
    /// them once `HypercubeApp` has caught up via the published completion
    /// message, avoiding a one-frame flash back to the pre-animation value.
    fn advance_reveal_animation(
        state: &mut HypercubeShaderState,
        delta: Duration,
    ) -> AnimationTick {
        let Some(animating) = state.animating_reveal.as_mut() else {
            return AnimationTick::Ignored;
        };

        animating.elapsed += delta;

        let t = if animating.duration.is_zero() {
            1.0
        } else {
            (animating.elapsed.as_secs_f32() / animating.duration.as_secs_f32()).clamp(0.0, 1.0)
        };
        let eased = ease(t);

        state.reveal_scale_override =
            Some(animating.start_scale + (animating.target_scale - animating.start_scale) * eased);
        state.reveal_gap_override =
            Some(animating.start_gap + (animating.target_gap - animating.start_gap) * eased);
        state.camera_controller.yaw =
            animating.start_yaw + (animating.target_yaw - animating.start_yaw) * eased;

        if animating.elapsed >= animating.duration {
            state.reveal_scale_override = Some(animating.target_scale);
            state.reveal_gap_override = Some(animating.target_gap);
            state.camera_controller.yaw = animating.target_yaw;
            state.animating_reveal = None;
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
        let cached_indices = HypercubeShaderProgram::calculate_indices(&rotation_4d).into();
        let hypercube = Hypercube::solved();
        let cached_sticker_instances = generate_sticker_instances(&hypercube).into();

        Self {
            camera,
            camera_controller,
            projection,
            rotation_4d,
            mouse_pressed: false,
            last_mouse_pos: None,
            shift_pressed: false,
            cached_indices,
            indices_generation: 0,
            cached_sticker_instances,
            sticker_generation: 0,
            hovered_sticker: None,
            debug_instances: Vec::new(),
            hypercube,
            animating_move: None,
            animating_focus: None,
            animating_reset: None,
            animating_reveal: None,
            reveal_scale_override: None,
            reveal_gap_override: None,
            rotate_press: None,
            pending_face_click: None,
            last_redraw_instant: None,
            reset_generation: 0,
            random_moves_generation: 0,
            rng: fastrand::Rng::new(),
            reveal_generation: 0,
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

    /// At `partial_angle` fully swept (end of a move, right before the snap
    /// to `generate_sticker_instances`), a facet whose own axis is one of
    /// the rotating slab's `free_axes` (`facet_axis_is_free`) must have its
    /// `face_normal_4d` fully onto its new face's `FACE_CENTERS`, not still
    /// mid-sweep or left on the old one - the counterpart to the start-of-move
    /// check above, checked directly per facet (not just as part of the
    /// set-based end-state comparison, which can't distinguish a wrong
    /// normal on one row from a legitimate match on another).
    #[test]
    fn animated_face_normal_matches_post_move_face_center_at_end_of_move() {
        use crate::moves::discrete_rotation;
        use crate::piece::face_id_for;

        for side_axis in 0..4usize {
            for side_sign in [-1i8, 1] {
                for local_coords in [[1i8, 0, 0], [1, 1, 0], [1, 1, 1]] {
                    let nonzero = local_coords.iter().filter(|c| **c != 0).count();
                    let angle = base_angle(nonzero);
                    let axes = free_axes(side_axis);
                    let (perm, sign) = discrete_rotation(local_coords, angle);
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
                        let expected_face_id = if facet.axis == side_axis
                            || pre_move.pieces[facet.piece_slot].position[side_axis] != side_sign
                        {
                            facet.face_id
                        } else {
                            let p = axes
                                .iter()
                                .position(|&x| x == facet.axis)
                                .expect("facet.axis != side_axis must be one of axes");
                            let slot = inv_perm[p];
                            face_id_for(axes[slot], sign[slot] * facet.side_sign)
                        };
                        let expected: [f32; 4] = FACE_CENTERS[expected_face_id].into();
                        assert_eq!(
                            round_key(instance.face_normal_4d),
                            round_key(expected),
                            "mismatch for side_axis={side_axis} side_sign={side_sign} \
                             local_coords={local_coords:?} piece_slot={} axis={}",
                            facet.piece_slot,
                            facet.axis
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

        let x = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let w = Vector4::new(0.0, 0.0, 0.0, 1.0);
        let starting_rotation = create_4d_plane_rotation(x, w, 1.2);
        state.rotation_4d = starting_rotation;

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
            0,
            0,
            0,
            false,
        );

        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));
        let sticker_generation_before = state.sticker_generation;
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
        assert_eq!(
            state.sticker_generation,
            sticker_generation_before + 1,
            "reset must regenerate cached sticker instances, not leave the \
             pre-reset (mid-move) cache in place"
        );
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(state.cached_sticker_instances.as_ref()),
            bytemuck::cast_slice::<_, u8>(&generate_sticker_instances(&Hypercube::solved())),
        );

        // The 4D orientation must not snap instantly - it's handed off to
        // `AnimatingReset` to animate back to identity over subsequent
        // ticks.
        assert!(
            (state.rotation_4d - starting_rotation).norm() < 1e-4,
            "rotation_4d must be untouched at the instant reset is pressed"
        );
        let animating = state
            .animating_reset
            .as_ref()
            .expect("reset must start a 4D orientation animation");
        let (expected_p, expected_q) = decompose_so4(&starting_rotation);
        assert!((animating.start_p.coords - expected_p.coords).norm() < 1e-4);
        assert!((animating.start_q.coords - expected_q.coords).norm() < 1e-4);
        assert_eq!(animating.duration, Duration::from_millis(250));
    }

    #[test]
    fn random_moves_generation_mismatch_applies_moves_and_cancels_animation() {
        let mut state = HypercubeShaderState {
            rng: fastrand::Rng::with_seed(1),
            ..Default::default()
        };
        assert_eq!(state.random_moves_generation, 0);

        let mut expected = Hypercube::solved();
        expected.apply_random_moves(3, &mut fastrand::Rng::with_seed(1));

        let program = HypercubeShaderProgram::new(
            0.5,
            2.0,
            RenderMode::Standard,
            AABBMode::None,
            RotateButton::default(),
            250,
            0,
            1,
            3,
            0,
            false,
        );

        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));
        let sticker_generation_before = state.sticker_generation;
        let action = program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );

        assert!(action.is_some(), "random moves must request a redraw");
        assert_eq!(state.hypercube, expected);
        assert!(state.animating_move.is_none());
        assert!(state.animating_focus.is_none());
        assert_eq!(state.random_moves_generation, 1);
        assert_eq!(
            state.sticker_generation,
            sticker_generation_before + 1,
            "random moves must regenerate cached sticker instances"
        );
    }

    #[test]
    fn random_moves_generation_mismatch_with_zero_count_is_a_no_op_move_wise() {
        let mut state = HypercubeShaderState::default();
        assert!(state.hypercube.is_solved());

        let program = HypercubeShaderProgram::new(
            0.5,
            2.0,
            RenderMode::Standard,
            AABBMode::None,
            RotateButton::default(),
            250,
            0,
            1,
            0,
            0,
            false,
        );

        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));
        let action = program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );

        assert!(action.is_some());
        assert!(state.hypercube.is_solved());
        assert_eq!(state.random_moves_generation, 1);
    }

    /// A `RedrawRequested` tick with nothing animating and no input must not
    /// regenerate or re-upload cached indices/sticker instances - the "camera
    /// at rest" case #3's generation-counter dirty-flag mechanism exists to
    /// skip.
    #[test]
    fn idle_redraw_does_not_bump_generations() {
        let mut state = HypercubeShaderState::default();
        let program = HypercubeShaderProgram::new(
            0.9,
            0.0,
            RenderMode::Standard,
            AABBMode::None,
            RotateButton::default(),
            250,
            state.reset_generation,
            state.random_moves_generation,
            0,
            state.reveal_generation,
            false,
        );
        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));

        program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );

        assert_eq!(state.indices_generation, 0);
        assert_eq!(state.sticker_generation, 0);
    }

    /// Clicking an actionable facet starts a move animation and must
    /// regenerate (and bump the generation of) cached sticker instances -
    /// otherwise the render would keep showing the pre-move snapshot.
    #[test]
    fn clicking_actionable_facet_bumps_sticker_generation() {
        let mut state = HypercubeShaderState::default();
        let sticker_index = FACET_TABLE
            .iter()
            .position(|f| f.is_actionable)
            .expect("at least one actionable facet exists");
        state.hovered_sticker = Some(sticker_index);
        let sticker_generation_before = state.sticker_generation;

        let rotate_button = RotateButton::default();
        let program = HypercubeShaderProgram::new(
            0.9,
            0.0,
            RenderMode::Standard,
            AABBMode::None,
            rotate_button,
            250,
            state.reset_generation,
            state.random_moves_generation,
            0,
            state.reveal_generation,
            false,
        );
        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));
        let cursor = mouse::Cursor::Available(Point::new(10.0, 10.0));

        program.update(
            &mut state,
            &Event::Mouse(mouse::Event::ButtonPressed(rotate_button.click_button())),
            bounds,
            cursor,
        );

        assert!(state.animating_move.is_some(), "click must start a move");
        assert_eq!(state.sticker_generation, sticker_generation_before + 1);
    }

    /// A "center this face" animation tick rotates `rotation_4d` every frame
    /// but never touches `Hypercube` state or `animating_move` - it must bump
    /// `indices_generation` (the winding-corrected index buffer depends on
    /// rotation) but leave `sticker_generation` untouched.
    #[test]
    fn focus_animation_tick_bumps_indices_generation_but_not_sticker_generation() {
        let mut state = HypercubeShaderState {
            animating_focus: Some(AnimatingFocus {
                start_rotation: Matrix4::identity(),
                plane: (
                    Vector4::new(1.0, 0.0, 0.0, 0.0),
                    Vector4::new(0.0, 1.0, 0.0, 0.0),
                ),
                total_angle: 90.0,
                elapsed: Duration::ZERO,
                duration: Duration::from_millis(250),
            }),
            ..Default::default()
        };
        let indices_generation_before = state.indices_generation;
        let sticker_generation_before = state.sticker_generation;

        let program = HypercubeShaderProgram::new(
            0.9,
            0.0,
            RenderMode::Standard,
            AABBMode::None,
            RotateButton::default(),
            250,
            state.reset_generation,
            state.random_moves_generation,
            0,
            state.reveal_generation,
            false,
        );
        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));

        program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );

        assert_eq!(state.indices_generation, indices_generation_before + 1);
        assert_eq!(state.sticker_generation, sticker_generation_before);
    }

    /// At `t=0` a reveal animation's overrides/yaw must exactly reproduce
    /// its start values; once `elapsed` reaches `duration`, they must snap
    /// exactly to the target values and the animation must report Completed
    /// and clear itself.
    #[test]
    fn advance_reveal_animation_interpolates_and_completes() {
        let mut state = HypercubeShaderState {
            animating_reveal: Some(AnimatingReveal {
                start_scale: 0.9,
                target_scale: 0.98,
                start_gap: 0.0,
                target_gap: 1.0,
                start_yaw: 10.0,
                target_yaw: 10.0 + REVEAL_YAW_SPIN_DEGREES,
                elapsed: Duration::ZERO,
                duration: Duration::from_millis(1000),
            }),
            ..Default::default()
        };

        let tick = HypercubeShaderProgram::advance_reveal_animation(&mut state, Duration::ZERO);
        assert!(matches!(tick, AnimationTick::Running));
        assert_eq!(state.reveal_scale_override, Some(0.9));
        assert_eq!(state.reveal_gap_override, Some(0.0));
        assert_eq!(state.camera_controller.yaw, 10.0);

        let tick = HypercubeShaderProgram::advance_reveal_animation(
            &mut state,
            Duration::from_millis(2000),
        );
        assert!(matches!(tick, AnimationTick::Completed));
        assert_eq!(state.reveal_scale_override, Some(0.98));
        assert_eq!(state.reveal_gap_override, Some(1.0));
        assert_eq!(state.camera_controller.yaw, 10.0 + REVEAL_YAW_SPIN_DEGREES);
        assert!(state.animating_reveal.is_none());
    }

    #[test]
    fn advance_reset_animation_interpolates_and_completes() {
        let x = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let w = Vector4::new(0.0, 0.0, 0.0, 1.0);
        let start_rotation = create_4d_plane_rotation(x, w, 1.2);
        let (start_p, start_q) = decompose_so4(&start_rotation);

        let mut state = HypercubeShaderState {
            rotation_4d: start_rotation,
            animating_reset: Some(AnimatingReset {
                start_p,
                start_q,
                elapsed: Duration::ZERO,
                duration: Duration::from_millis(1000),
            }),
            ..Default::default()
        };

        let tick = HypercubeShaderProgram::advance_reset_animation(&mut state, Duration::ZERO);
        assert!(matches!(tick, AnimationTick::Running));
        assert!((state.rotation_4d - start_rotation).norm() < 1e-4);

        let tick = HypercubeShaderProgram::advance_reset_animation(
            &mut state,
            Duration::from_millis(2000),
        );
        assert!(matches!(tick, AnimationTick::Completed));
        assert!((state.rotation_4d - Matrix4::identity()).norm() < 1e-4);
        assert!(state.animating_reset.is_none());
    }

    /// A bumped `reveal_generation` with `revealed_target: true` must start
    /// an `AnimatingReveal` toward the secondary defaults from the program's
    /// current values, cancel any in-progress move/focus animation, and sync
    /// `state.reveal_generation`.
    #[test]
    fn reveal_generation_mismatch_starts_reveal_animation_and_cancels_others() {
        let mut state = HypercubeShaderState::default();
        assert_eq!(state.reveal_generation, 0);

        let facet = FACET_TABLE
            .iter()
            .find(|f| f.is_actionable)
            .expect("at least one actionable facet exists");
        let nonzero = facet.local_coords.iter().filter(|c| **c != 0).count();
        let angle = base_angle(nonzero);
        let pre_move_pieces = state.hypercube.pieces.clone();
        state.animating_move = Some(AnimatingMove {
            side_axis: facet.axis,
            side_sign: facet.side_sign,
            local_coords: facet.local_coords,
            angle,
            pre_move_pieces,
            elapsed: Duration::ZERO,
            duration: Duration::from_millis(250),
        });
        state.animating_focus = Some(AnimatingFocus {
            start_rotation: Matrix4::identity(),
            plane: (
                Vector4::new(1.0, 0.0, 0.0, 0.0),
                Vector4::new(0.0, 1.0, 0.0, 0.0),
            ),
            total_angle: 1.0,
            elapsed: Duration::ZERO,
            duration: Duration::from_millis(250),
        });
        state.camera_controller.yaw = 42.0;

        let program = HypercubeShaderProgram::new(
            0.9,
            0.0,
            RenderMode::Standard,
            AABBMode::None,
            RotateButton::default(),
            250,
            0,
            0,
            0,
            1,
            true,
        );

        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));
        let action = program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );

        assert!(action.is_some(), "starting a reveal must request a redraw");
        assert!(state.animating_move.is_none());
        assert!(state.animating_focus.is_none());
        let animating = state
            .animating_reveal
            .as_ref()
            .expect("reveal animation should have started");
        assert_eq!(animating.start_scale, 0.9);
        assert_eq!(animating.start_gap, 0.0);
        assert_eq!(animating.start_yaw, 42.0);
        assert_eq!(animating.target_yaw, 42.0 + REVEAL_YAW_SPIN_DEGREES);
        assert_eq!(animating.target_scale, 1.0 - SECONDARY_STICKER_SCALE);
        assert_eq!(animating.target_gap, SECONDARY_FACE_GAP);
        assert_eq!(state.reveal_generation, 1);
    }

    /// With `revealed_target: false` (hiding), the animation must target the
    /// primary defaults instead of the secondary ones.
    #[test]
    fn reveal_generation_mismatch_targets_primary_defaults_when_hiding() {
        let mut state = HypercubeShaderState::default();
        let program = HypercubeShaderProgram::new(
            1.0 - SECONDARY_STICKER_SCALE,
            SECONDARY_FACE_GAP,
            RenderMode::Standard,
            AABBMode::None,
            RotateButton::default(),
            250,
            0,
            0,
            0,
            1,
            false,
        );
        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));
        program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );

        let animating = state
            .animating_reveal
            .as_ref()
            .expect("reveal animation should have started");
        assert_eq!(animating.target_scale, 1.0 - PRIMARY_STICKER_SCALE);
        assert_eq!(animating.target_gap, PRIMARY_FACE_GAP);
    }

    /// The reveal overrides must stay put until the program's own
    /// sticker_scale/face_gap (i.e. `HypercubeApp`, once it has processed
    /// `RevealAnimationComplete`) match the stored target, then clear so a
    /// later manual slider drag isn't masked.
    #[test]
    fn reveal_override_clears_once_program_value_catches_up() {
        let mut state = HypercubeShaderState {
            reveal_scale_override: Some(0.9),
            reveal_gap_override: Some(1.0),
            ..Default::default()
        };
        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));

        let stale_program = HypercubeShaderProgram::new(
            0.5,
            0.0,
            RenderMode::Standard,
            AABBMode::None,
            RotateButton::default(),
            250,
            0,
            0,
            0,
            0,
            false,
        );
        stale_program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );
        assert_eq!(state.reveal_scale_override, Some(0.9));
        assert_eq!(state.reveal_gap_override, Some(1.0));

        let caught_up_program = HypercubeShaderProgram::new(
            0.9,
            1.0,
            RenderMode::Standard,
            AABBMode::None,
            RotateButton::default(),
            250,
            0,
            0,
            0,
            0,
            false,
        );
        caught_up_program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );
        assert_eq!(state.reveal_scale_override, None);
        assert_eq!(state.reveal_gap_override, None);
    }

    /// Once a reveal animation completes, `Program::update` must publish
    /// `Message::RevealAnimationComplete` carrying the raw-domain constants
    /// directly (not back-derived from the render-domain target), since a
    /// float round-trip through `1.0 - x` isn't guaranteed to reproduce the
    /// exact constant the override-reconciliation `==` check needs.
    #[test]
    fn reveal_completion_publishes_message_with_raw_domain_constants() {
        let mut state = HypercubeShaderState {
            animating_reveal: Some(AnimatingReveal {
                start_scale: 1.0 - PRIMARY_STICKER_SCALE,
                target_scale: 1.0 - SECONDARY_STICKER_SCALE,
                start_gap: PRIMARY_FACE_GAP,
                target_gap: SECONDARY_FACE_GAP,
                start_yaw: 0.0,
                target_yaw: REVEAL_YAW_SPIN_DEGREES,
                elapsed: REVEAL_ANIMATION_DURATION + Duration::from_millis(100),
                duration: REVEAL_ANIMATION_DURATION,
            }),
            ..Default::default()
        };
        let program = HypercubeShaderProgram::new(
            1.0 - PRIMARY_STICKER_SCALE,
            PRIMARY_FACE_GAP,
            RenderMode::Standard,
            AABBMode::None,
            RotateButton::default(),
            250,
            0,
            0,
            0,
            0,
            true,
        );
        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));
        let action = program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );

        let (message, ..) = action
            .expect("a completed reveal must produce an action")
            .into_inner();
        match message.expect("a completed reveal must publish a message") {
            Message::RevealAnimationComplete {
                final_scale,
                final_gap,
            } => {
                assert_eq!(final_scale, SECONDARY_STICKER_SCALE);
                assert_eq!(final_gap, SECONDARY_FACE_GAP);
            }
            other => panic!("expected RevealAnimationComplete, got {other:?}"),
        }
    }

    /// Camera-drag orbit start and facet turn-clicks must both be ignored
    /// while a reveal/hide flourish is playing, per the "locked cutscene"
    /// requirement.
    #[test]
    fn mouse_input_is_ignored_while_reveal_animation_plays() {
        let mut state = HypercubeShaderState {
            animating_reveal: Some(AnimatingReveal {
                start_scale: 0.9,
                target_scale: 0.98,
                start_gap: 0.0,
                target_gap: 1.0,
                start_yaw: 0.0,
                target_yaw: REVEAL_YAW_SPIN_DEGREES,
                elapsed: Duration::ZERO,
                duration: REVEAL_ANIMATION_DURATION,
            }),
            ..Default::default()
        };
        let rotate_button = RotateButton::default();
        let program = HypercubeShaderProgram::new(
            0.9,
            0.0,
            RenderMode::Standard,
            AABBMode::None,
            rotate_button,
            250,
            0,
            0,
            0,
            0,
            true,
        );
        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));
        let cursor = mouse::Cursor::Available(Point::new(10.0, 10.0));

        program.update(
            &mut state,
            &Event::Mouse(mouse::Event::ButtonPressed(rotate_button.to_mouse_button())),
            bounds,
            cursor,
        );
        assert!(
            !state.mouse_pressed,
            "camera drag must not start during the reveal flourish"
        );
        assert!(state.rotate_press.is_none());

        state.hovered_sticker = Some(0);
        let pieces_before = state.hypercube.pieces.clone();
        program.update(
            &mut state,
            &Event::Mouse(mouse::Event::ButtonPressed(rotate_button.click_button())),
            bounds,
            cursor,
        );
        assert_eq!(
            state.hypercube.pieces, pieces_before,
            "facet turn must not apply during the reveal flourish"
        );
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
