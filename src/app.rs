//! Gui elements and messaging for the application

use std::time::{Duration, Instant};

use iced::widget::{Button, Checkbox, Column, PickList, Row, Shader, Slider, Space};
use iced::{Element, Length, Subscription, Task, window};
use serde::{Deserialize, Serialize};

use crate::animation::{ease, lerp};
use crate::menu_overlay;
use crate::piece::Hypercube;
use crate::puzzle_state;
use crate::settings::{self, ANIMATION_DURATION_MS_RANGE, AppSettings, RotateButton};
use crate::shader_widget::{
    HypercubeShaderProgram, PRIMARY_FACE_GAP, PRIMARY_FACE_GAP_4D, PRIMARY_STICKER_SCALE,
    REVEAL_ANIMATION_DURATION, SECONDARY_FACE_GAP, SECONDARY_FACE_GAP_4D, SECONDARY_STICKER_SCALE,
    SolveCommand,
};
use crate::solver::{SolveError, Stage};
use crate::theme::Theme;

/// Rendering modes for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum RenderMode {
    Standard,
    Normals,
    Depth,
}

/// AABB visualization modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum AABBMode {
    None,
    Face,
    Sticker,
}

impl std::fmt::Display for RenderMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderMode::Standard => write!(f, "Standard"),
            RenderMode::Normals => write!(f, "Normal Map"),
            RenderMode::Depth => write!(f, "Depth Map"),
        }
    }
}

impl RenderMode {
    pub(crate) const ALL: [RenderMode; 3] =
        [RenderMode::Standard, RenderMode::Normals, RenderMode::Depth];
}

impl std::fmt::Display for AABBMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AABBMode::None => write!(f, "AABB Rendering Disabled"),
            AABBMode::Face => write!(f, "Face AABB"),
            AABBMode::Sticker => write!(f, "Sticker AABB"),
        }
    }
}

impl AABBMode {
    pub(crate) const ALL: [AABBMode; 3] = [AABBMode::None, AABBMode::Face, AABBMode::Sticker];
}

/// Formats the floating tooltip text for the sticker scale slider.
fn format_sticker_scale(value: f32) -> String {
    let value = 1.0 - value;
    format!("{value:.2}")
}

/// Formats the floating tooltip text for the face gap slider.
fn format_face_gap(value: f32) -> String {
    format!("{value:.2}")
}

/// Formats the floating tooltip text for the 4D face gap slider.
fn format_face_gap_4d(value: f32) -> String {
    format!("{value:.2}")
}

/// Formats the floating tooltip text for the 4D viewer distance slider.
fn format_viewer_distance(value: f32) -> String {
    format!("{value:.2}")
}

/// Formats the floating tooltip text for the animation duration slider.
fn format_animation_duration(duration_ms: u32) -> String {
    format!("{duration_ms}ms")
}

/// Delay before the floating value tooltip appears: instant while the
/// slider is actively being adjusted, otherwise a short hover delay.
fn tooltip_delay(is_adjusting: bool) -> Duration {
    if is_adjusting {
        Duration::ZERO
    } else {
        Duration::from_millis(400)
    }
}

/// Label for the reveal/hide toggle button. `revealed` flips the instant the
/// button is pressed (so the shader program picks up the new target that
/// same frame), so the label alone can't read `revealed` directly while
/// `reveal_animating` is still true or it would flip early - it keeps
/// reporting the pre-press state until the flourish settles.
pub(crate) fn reveal_button_label(revealed: bool, reveal_animating: bool) -> &'static str {
    match (revealed, reveal_animating) {
        (true, false) => "Hide",
        (false, false) => "Reveal",
        (true, true) => "Reveal",
        (false, true) => "Hide",
    }
}

/// Label for the Puzzle menu's solve item, which doubles as its stop control
/// while a solve plays back.
pub(crate) fn solve_button_label(solving: bool) -> &'static str {
    if solving { "Stop Solving" } else { "Solve" }
}

/// `n` with thousands separators, e.g. "1,904".
fn format_count(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// How long a solve's closing notice ("Solved in N moves", ...) stays up.
const SOLVE_NOTICE_DURATION: Duration = Duration::from_millis(2500);

/// How a solve ended, as reported by `shader_widget.rs`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SolveOutcome {
    Completed { total: usize },
    AlreadySolved,
    Failed(SolveError),
}

/// The closing notice shown for a solve outcome.
fn solve_notice_text(outcome: &SolveOutcome) -> String {
    match outcome {
        SolveOutcome::Completed { total } => format!(
            "Solved in {} {}",
            format_count(*total),
            if *total == 1 { "move" } else { "moves" }
        ),
        SolveOutcome::AlreadySolved => "Already solved".to_string(),
        SolveOutcome::Failed(error) => format!("Can't solve: {error}"),
    }
}

/// The viewport overlay's text: live progress while solving, the closing
/// notice for a while afterwards, otherwise nothing.
fn solve_overlay_text(
    solving: bool,
    progress: Option<(Stage, usize, usize)>,
    notice: Option<&str>,
) -> Option<String> {
    if solving {
        return Some(match progress {
            Some((stage, done, total)) => format!(
                "Solving: {stage} \u{b7} move {} / {}",
                format_count(done),
                format_count(total)
            ),
            None => "Solving\u{2026}".to_string(),
        });
    }
    notice.map(str::to_string)
}

/// Exact natural height, in pixels, of a single default-size label+slider
/// row (`text` + `spacing(5)` + `Slider`): a `1.3 * 16.0 = 20.8`px text
/// line height, plus `5.0`px spacing, plus a default-height (`16.0`px)
/// `Slider`.
const SLIDER_ROW_HEIGHT: f32 = 20.8 + 5.0 + 16.0;

/// Height of a blank margin/gap within the sticker-scale/face-gap panel,
/// equal to the `20` spacing used between every other pair of controls in
/// the column.
const SLIDERS_PANEL_GAP: f32 = 20.0;

/// Height of the sticker-scale/face-gap panel when fully open: a blank
/// leading margin, the four rows with a gap between each, and a blank
/// trailing margin.
const SLIDERS_PANEL_OPEN_HEIGHT: f32 = SLIDERS_PANEL_GAP * 5.0 + SLIDER_ROW_HEIGHT * 4.0;

/// Height of the sticker-scale/face-gap panel when fully closed: its blank
/// leading margin alone.
const SLIDERS_PANEL_CLOSED_HEIGHT: f32 = SLIDERS_PANEL_GAP;

/// Default value of the 4D viewer distance slider.
const DEFAULT_VIEWER_DISTANCE: f32 = 4.0;

/// Portion of a reveal/hide flourish's total duration spent opening or
/// closing the sticker-scale/face-gap panel: the panel snaps open in the
/// first `PANEL_ANIMATION_FRACTION` of a reveal, or snaps shut in the last
/// `PANEL_ANIMATION_FRACTION` of a hide, so it reads as a drawer sliding
/// open/shut rather than tracking the slower value slide at the same pace.
const PANEL_ANIMATION_FRACTION: f32 = 0.1;

/// Remaps a flourish's overall `[0,1]` progress `t` to the panel's own
/// `[0,1]` open/close progress: for `revealing`, the panel's progress
/// reaches 1.0 once `t` reaches `PANEL_ANIMATION_FRACTION` and holds there;
/// for hiding, it holds at 0.0 until the last `PANEL_ANIMATION_FRACTION` of
/// `t`, then rises to 1.0 as `t` reaches 1.0.
fn panel_progress(t: f32, revealing: bool) -> f32 {
    if revealing {
        (t / PANEL_ANIMATION_FRACTION).clamp(0.0, 1.0)
    } else {
        ((t - (1.0 - PANEL_ANIMATION_FRACTION)) / PANEL_ANIMATION_FRACTION).clamp(0.0, 1.0)
    }
}

/// Main application state - handles UI controls only
#[derive(Debug)]
pub(crate) struct HypercubeApp {
    sticker_scale: f32,
    face_gap: f32,
    face_gap_4d: f32,
    viewer_distance: f32,
    render_mode: RenderMode,
    aabb_mode: AABBMode,
    /// Shows a small inset with one face's cells rendered opaque and
    /// depth-tested (cycling through whichever faces hold a Fire sticker),
    /// to eyeball whether `fire_draw_order` got the occlusion right against
    /// the GPU's own depth buffer.
    fire_ground_truth_debug: bool,
    /// Smoothed frames-per-second, updated on every `Message::FpsTick` while
    /// `debug_mode` is on (see `subscription`); displayed as a viewport
    /// overlay.
    fps: f32,
    /// Timestamp of the previous `Message::FpsTick`, for computing the
    /// per-frame delta backing `fps`. `None` right after debug mode is
    /// (re-)enabled, so the first tick doesn't compute a delta against a
    /// stale, stretched-out gap.
    last_fps_frame: Option<Instant>,
    settings: AppSettings,
    reset_generation: u64,
    random_moves_generation: u64,
    /// Move count carried alongside `random_moves_generation` for the shader
    /// program to pick up, since a bare generation bump carries no payload
    /// (mirrors how `revealed` is threaded alongside `reveal_generation`).
    pending_random_move_count: u32,
    sticker_scale_adjusting: bool,
    face_gap_adjusting: bool,
    face_gap_4d_adjusting: bool,
    viewer_distance_adjusting: bool,
    animation_duration_adjusting: bool,
    /// Target reveal state. Flips immediately on `ToggleReveal` (so the
    /// shader program picks up the new direction that same frame), not only
    /// once the flourish settles.
    revealed: bool,
    reveal_generation: u64,
    /// True from a `ToggleReveal` press until `RevealAnimationComplete`
    /// arrives; gates the button (disabled) and slider interaction (locked
    /// while animating).
    reveal_animating: bool,
    /// When the current reveal/hide flourish was triggered, for timing the
    /// application-level slider interpolation. `None` when idle.
    reveal_animation_started: Option<Instant>,
    /// How open the sticker-scale/face-gap panel is: 0.0 fully collapsed,
    /// 1.0 fully open.
    reveal_panel_fraction: f32,
    /// Remaining scripted reveal/hide flourishes after the one the boot task
    /// already kicked off. See [`next_reveal_loop_action`].
    #[cfg(feature = "gpu-capture-hooks")]
    reveal_loop_remaining: u32,
    about_open: bool,
    save_generation: u64,
    load_generation: u64,
    pending_load: Option<Hypercube>,
    save_snapshot_generation: u64,
    solve_command_generation: u64,
    /// Carried alongside `solve_command_generation` (see
    /// `shader_widget::SolveCommand`).
    solve_command: SolveCommand,
    /// True from a `Solve` press until playback ends or is cancelled; flips
    /// the Puzzle menu's solve item to "Stop Solving".
    solving: bool,
    /// The latest (stage, move number, total moves) reported by playback.
    solve_progress: Option<(Stage, usize, usize)>,
    /// A finished solve's closing notice and when it was posted; cleared by
    /// `SolveNoticeTick` once `SOLVE_NOTICE_DURATION` has passed.
    solve_notice: Option<(String, Instant)>,
}

/// Number of scripted flourishes still to run, after the one the boot task
/// already kicked off, so `--features gpu-capture-hooks` totals 5 runs.
#[cfg(feature = "gpu-capture-hooks")]
const REVEAL_LOOP_REPEATS: u32 = 4;

/// What to do when a scripted reveal/hide flourish completes under
/// `--features gpu-capture-hooks`: keep cycling, or exit once the fixed
/// number of runs (see [`REVEAL_LOOP_REPEATS`]) has played.
#[cfg(feature = "gpu-capture-hooks")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RevealLoopAction {
    Repeat,
    Exit,
}

#[cfg(feature = "gpu-capture-hooks")]
fn next_reveal_loop_action(remaining: u32) -> RevealLoopAction {
    if remaining > 0 {
        RevealLoopAction::Repeat
    } else {
        RevealLoopAction::Exit
    }
}

/// Messages that the application can receive
#[derive(Debug, Clone)]
pub(crate) enum Message {
    StickerScale(f32),
    StickerScaleReleased,
    FaceGap(f32),
    FaceGapReleased,
    FaceGap4d(f32),
    FaceGap4dReleased,
    ViewerDistance(f32),
    ViewerDistanceReleased,
    RenderMode(RenderMode),
    AABBMode(AABBMode),
    /// Debug-only: toggles a small inset showing one face's cells opaque and
    /// depth-tested, cycling through whichever faces hold a Fire sticker
    /// (see `shader_widget::ground_truth_debug_face`).
    FireGroundTruthDebug(bool),
    /// Toggles rendering the rotation-axis gizmo ring during focus
    /// animations, Shift+drag, and Reset (see `shader_widget::build_gizmo_vertices`).
    ShowGizmoRing(bool),
    DebugMode(bool),
    /// Per-frame tick driving the debug-mode FPS overlay (see
    /// `HypercubeApp::subscription`).
    FpsTick(Instant),
    RotateButton(RotateButton),
    Theme(Theme),
    AnimationDuration(u32),
    AnimationDurationReleased,
    Reset,
    RandomMoves(u32),
    /// Solve the puzzle and play the solution back.
    Solve,
    /// Stop solve playback (the Puzzle menu while solving, or Esc).
    StopSolving,
    /// Published by `shader_widget.rs` as each solve move starts.
    SolveProgress {
        generation: u64,
        stage: Stage,
        done: usize,
        total: usize,
    },
    /// Published by `shader_widget.rs` once a solve finishes or can't start.
    SolveEnded {
        generation: u64,
        outcome: SolveOutcome,
    },
    /// Per-frame tick while a solve notice is showing, to expire it.
    SolveNoticeTick(Instant),
    ToggleReveal,
    RevealAnimationTick(Instant),
    RevealAnimationComplete {
        final_scale: f32,
        final_gap: f32,
        final_gap_4d: f32,
    },
    /// Swallows a mouse-wheel scroll over the sticker-scale/face-gap panel,
    /// so the `Scrollable` it's clipped by never scrolls itself.
    SlidersPanelWheelScroll,
    SavePuzzle,
    PuzzleReadyToSave(Hypercube),
    LoadPuzzle,
    /// Debug-only: captures the current frame and full view state to a
    /// timestamped file pair on disk (see `snapshot.rs`).
    SaveSnapshot,
    Quit,
    OpenAbout,
    CloseAbout,
    /// Opens a license/attribution URL from the About dialog in the
    /// system's default browser.
    OpenUrl(&'static str),
    /// Target of the Puzzle menu's inert spacer rows (`menu_layout::puzzle_items`).
    NoOp,
}

impl HypercubeApp {
    fn new_inner() -> Self {
        Self {
            sticker_scale: PRIMARY_STICKER_SCALE,
            face_gap: PRIMARY_FACE_GAP,
            face_gap_4d: PRIMARY_FACE_GAP_4D,
            viewer_distance: DEFAULT_VIEWER_DISTANCE,
            render_mode: RenderMode::Standard,
            aabb_mode: AABBMode::None,
            fire_ground_truth_debug: false,
            fps: 0.0,
            last_fps_frame: None,
            settings: settings::load(),
            reset_generation: 0,
            random_moves_generation: 0,
            pending_random_move_count: 0,
            sticker_scale_adjusting: false,
            face_gap_adjusting: false,
            face_gap_4d_adjusting: false,
            viewer_distance_adjusting: false,
            animation_duration_adjusting: false,
            revealed: false,
            reveal_generation: 0,
            reveal_animating: false,
            reveal_animation_started: None,
            reveal_panel_fraction: 0.0,
            #[cfg(feature = "gpu-capture-hooks")]
            reveal_loop_remaining: REVEAL_LOOP_REPEATS,
            about_open: false,
            save_generation: 0,
            load_generation: 0,
            pending_load: None,
            save_snapshot_generation: 0,
            solve_command_generation: 0,
            solve_command: SolveCommand::Stop,
            solving: false,
            solve_progress: None,
            solve_notice: None,
        }
    }

    fn send_solve_command(&mut self, command: SolveCommand) {
        self.solve_command = command;
        self.solve_command_generation = self.solve_command_generation.wrapping_add(1);
    }

    /// Forgets any solve in progress. Reset, Random Moves and a successful
    /// Load already cancel playback inside `shader_widget.rs`, so this is all
    /// they need; any late message from the cancelled run is then ignored.
    fn stop_solving_locally(&mut self) {
        self.solving = false;
        self.solve_progress = None;
    }

    /// Create a new application instance
    #[cfg(not(feature = "gpu-capture-hooks"))]
    pub(crate) fn new() -> Self {
        Self::new_inner()
    }

    /// Create a new application instance and kick off the scripted
    /// reveal/hide loop immediately, so a profiler attached to the process
    /// has a fixed, reproducible GPU workload without manual clicking.
    #[cfg(feature = "gpu-capture-hooks")]
    pub(crate) fn new() -> (Self, Task<Message>) {
        (Self::new_inner(), Task::done(Message::ToggleReveal))
    }

    /// Get the title of the application
    pub(crate) fn title(&self) -> String {
        "4D Hypercube".to_string()
    }

    /// Update the application state
    pub(crate) fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::StickerScale(value) => {
                if !self.reveal_animating {
                    self.sticker_scale = value;
                    self.sticker_scale_adjusting = true;
                }
            }
            Message::StickerScaleReleased => {
                self.sticker_scale_adjusting = false;
            }
            Message::FaceGap(value) => {
                if !self.reveal_animating {
                    self.face_gap = value;
                    self.face_gap_adjusting = true;
                }
            }
            Message::FaceGapReleased => {
                self.face_gap_adjusting = false;
            }
            Message::FaceGap4d(value) => {
                if !self.reveal_animating {
                    self.face_gap_4d = value;
                    self.face_gap_4d_adjusting = true;
                }
            }
            Message::FaceGap4dReleased => {
                self.face_gap_4d_adjusting = false;
            }
            Message::ViewerDistance(value) => {
                self.viewer_distance = value;
                self.viewer_distance_adjusting = true;
            }
            Message::ViewerDistanceReleased => {
                self.viewer_distance_adjusting = false;
            }
            Message::RenderMode(mode) => {
                self.render_mode = mode;
            }
            Message::AABBMode(mode) => {
                self.aabb_mode = mode;
            }
            Message::FireGroundTruthDebug(enabled) => {
                self.fire_ground_truth_debug = enabled;
            }
            Message::ShowGizmoRing(enabled) => {
                self.settings.show_gizmo_ring = enabled;
                settings::save(&self.settings);
            }
            Message::DebugMode(enabled) => {
                self.settings.debug_mode = enabled;
                settings::save(&self.settings);
                if enabled {
                    self.fps = 0.0;
                    self.last_fps_frame = None;
                }
            }
            Message::FpsTick(now) => {
                if let Some(last) = self.last_fps_frame {
                    let delta = now.duration_since(last).as_secs_f32();
                    if delta > 0.0 {
                        let instant_fps = 1.0 / delta;
                        self.fps = if self.fps == 0.0 {
                            instant_fps
                        } else {
                            self.fps * 0.9 + instant_fps * 0.1
                        };
                    }
                }
                self.last_fps_frame = Some(now);
            }
            Message::RotateButton(button) => {
                self.settings.rotate_button = button;
                settings::save(&self.settings);
            }
            Message::Theme(theme) => {
                self.settings.theme = theme;
                settings::save(&self.settings);
            }
            Message::AnimationDuration(duration_ms) => {
                self.settings.animation_duration_ms = duration_ms;
                settings::save(&self.settings);
                self.animation_duration_adjusting = true;
            }
            Message::AnimationDurationReleased => {
                self.animation_duration_adjusting = false;
            }
            Message::Reset => {
                self.reset_generation = self.reset_generation.wrapping_add(1);
                self.stop_solving_locally();
            }
            Message::RandomMoves(count) => {
                self.pending_random_move_count = count;
                self.random_moves_generation = self.random_moves_generation.wrapping_add(1);
                self.stop_solving_locally();
            }
            Message::Solve => {
                if !self.solving {
                    self.send_solve_command(SolveCommand::Start);
                    self.solving = true;
                    self.solve_progress = None;
                    self.solve_notice = None;
                }
            }
            Message::StopSolving => {
                if self.solving {
                    self.send_solve_command(SolveCommand::Stop);
                    self.stop_solving_locally();
                }
            }
            Message::SolveProgress {
                generation,
                stage,
                done,
                total,
            } => {
                if self.solving && generation == self.solve_command_generation {
                    self.solve_progress = Some((stage, done, total));
                }
            }
            Message::SolveEnded {
                generation,
                outcome,
            } => {
                if self.solving && generation == self.solve_command_generation {
                    self.stop_solving_locally();
                    self.solve_notice = Some((solve_notice_text(&outcome), Instant::now()));
                }
            }
            Message::SolveNoticeTick(now) => {
                if self
                    .solve_notice
                    .as_ref()
                    .is_some_and(|(_, posted)| now.duration_since(*posted) >= SOLVE_NOTICE_DURATION)
                {
                    self.solve_notice = None;
                }
            }
            Message::ToggleReveal => {
                self.revealed = !self.revealed;
                self.reveal_generation = self.reveal_generation.wrapping_add(1);
                self.reveal_animating = true;
                self.reveal_animation_started = Some(Instant::now());
            }
            Message::RevealAnimationTick(now) => {
                let elapsed = self
                    .reveal_animation_started
                    .map(|started| now.duration_since(started))
                    .unwrap_or_default();
                let t = (elapsed.as_secs_f32() / REVEAL_ANIMATION_DURATION.as_secs_f32())
                    .clamp(0.0, 1.0);
                let eased = ease(t);
                let panel_eased = ease(panel_progress(t, self.revealed));
                // (sticker scale, face gap, 4D face gap, panel fraction) at
                // the flourish's start and target, in the direction
                // currently underway.
                let primary = (
                    PRIMARY_STICKER_SCALE,
                    PRIMARY_FACE_GAP,
                    PRIMARY_FACE_GAP_4D,
                    0.0,
                );
                let secondary = (
                    SECONDARY_STICKER_SCALE,
                    SECONDARY_FACE_GAP,
                    SECONDARY_FACE_GAP_4D,
                    1.0,
                );
                let (start, target) = if self.revealed {
                    (primary, secondary)
                } else {
                    (secondary, primary)
                };
                self.sticker_scale = lerp(start.0, target.0, eased);
                self.face_gap = lerp(start.1, target.1, eased);
                self.face_gap_4d = lerp(start.2, target.2, eased);
                self.reveal_panel_fraction = lerp(start.3, target.3, panel_eased);
            }
            Message::RevealAnimationComplete {
                final_scale,
                final_gap,
                final_gap_4d,
            } => {
                self.sticker_scale = final_scale;
                self.face_gap = final_gap;
                self.face_gap_4d = final_gap_4d;
                self.reveal_animating = false;
                self.reveal_animation_started = None;
                self.reveal_panel_fraction = if self.revealed { 1.0 } else { 0.0 };

                #[cfg(feature = "gpu-capture-hooks")]
                {
                    return match next_reveal_loop_action(self.reveal_loop_remaining) {
                        RevealLoopAction::Repeat => {
                            self.reveal_loop_remaining -= 1;
                            Task::done(Message::ToggleReveal)
                        }
                        RevealLoopAction::Exit => iced::exit(),
                    };
                }
            }
            Message::SlidersPanelWheelScroll => {}
            Message::SavePuzzle => {
                self.save_generation = self.save_generation.wrapping_add(1);
            }
            Message::PuzzleReadyToSave(hypercube) => {
                puzzle_state::save(&hypercube);
            }
            Message::SaveSnapshot => {
                self.save_snapshot_generation = self.save_snapshot_generation.wrapping_add(1);
            }
            Message::LoadPuzzle => {
                self.pending_load = puzzle_state::load();
                self.load_generation = self.load_generation.wrapping_add(1);
                if self.pending_load.is_some() {
                    self.stop_solving_locally();
                }
            }
            Message::Quit => return iced::exit(),
            Message::OpenAbout => {
                self.about_open = true;
            }
            Message::CloseAbout => {
                self.about_open = false;
            }
            Message::OpenUrl(url) => {
                if let Err(err) = open::that(url) {
                    log::warn!("Failed to open {url}: {err}");
                }
            }
            Message::NoOp => {}
        }

        Task::none()
    }

    /// Global keyboard shortcuts, mirroring the File/Help menu items (plus
    /// Esc for Stop Solving - gated in `update`, since these closures can't
    /// capture `self`), a per-frame tick while a reveal/hide flourish is
    /// animating, another while `debug_mode` is on (driving the FPS overlay),
    /// and another while a solve notice is showing (to expire it).
    pub(crate) fn subscription(&self) -> Subscription<Message> {
        use iced::keyboard::{Key, key};

        let keyboard = iced::keyboard::listen().filter_map(|event| {
            let iced::keyboard::Event::KeyPressed { key, modifiers, .. } = event else {
                return None;
            };

            match (key.as_ref(), modifiers.control()) {
                (Key::Character("q"), true) => Some(Message::Quit),
                (Key::Character("s"), true) => Some(Message::SavePuzzle),
                (Key::Character("o"), true) => Some(Message::LoadPuzzle),
                (Key::Named(key::Named::F1), _) => Some(Message::OpenAbout),
                (Key::Named(key::Named::Escape), _) => Some(Message::StopSolving),
                _ => None,
            }
        });

        let mut subscriptions = vec![keyboard];
        if self.reveal_animating {
            subscriptions.push(window::frames().map(Message::RevealAnimationTick));
        }
        if self.settings.debug_mode {
            subscriptions.push(window::frames().map(Message::FpsTick));
        }
        if self.solve_notice.is_some() {
            subscriptions.push(window::frames().map(Message::SolveNoticeTick));
        }
        Subscription::batch(subscriptions)
    }

    /// Create the view for the application
    pub(crate) fn view(&self) -> Element<'_, Message> {
        // Left pane with controls
        let mut controls = Column::new()
            .spacing(20)
            .push(
                Checkbox::new(self.settings.debug_mode)
                    .label("Debug Mode")
                    .on_toggle(Message::DebugMode),
            )
            .push(
                Checkbox::new(self.settings.show_gizmo_ring)
                    .label("Show Gizmo Ring")
                    .on_toggle(Message::ShowGizmoRing),
            )
            .push(
                Column::new()
                    .spacing(5)
                    .push(iced::widget::text("Rotate Button"))
                    .push(
                        PickList::new(
                            &RotateButton::ALL[..],
                            Some(self.settings.rotate_button),
                            Message::RotateButton,
                        )
                        .width(250),
                    ),
            )
            .push(
                Column::new()
                    .spacing(5)
                    .push(iced::widget::text("Theme"))
                    .push(
                        PickList::new(&Theme::ALL[..], Some(self.settings.theme), Message::Theme)
                            .width(250),
                    ),
            );

        if self.settings.debug_mode {
            controls = controls
                .push(
                    Column::new()
                        .spacing(5)
                        .push(iced::widget::text("Render Mode"))
                        .push(
                            PickList::new(
                                &RenderMode::ALL[..],
                                Some(self.render_mode),
                                Message::RenderMode,
                            )
                            .width(250),
                        ),
                )
                .push(
                    Column::new()
                        .spacing(5)
                        .push(iced::widget::text("AABB Mode"))
                        .push(
                            PickList::new(
                                &AABBMode::ALL[..],
                                Some(self.aabb_mode),
                                Message::AABBMode,
                            )
                            .width(250),
                        ),
                )
                .push(
                    Checkbox::new(self.fire_ground_truth_debug)
                        .label("Fire Ground Truth Debug")
                        .on_toggle(Message::FireGroundTruthDebug),
                )
                .push(Button::new("Save Snapshot").on_press(Message::SaveSnapshot));
        }

        let sliders = Column::new()
            .spacing(0)
            .push(iced::widget::Space::new().height(SLIDERS_PANEL_GAP))
            .push(
                Column::new()
                    .spacing(5)
                    .push(iced::widget::text("Sticker Scale"))
                    .push(
                        iced::widget::tooltip(
                            Slider::new(0.0..=0.9, self.sticker_scale, Message::StickerScale)
                                .step(0.01f32)
                                .width(250)
                                .on_release(Message::StickerScaleReleased),
                            iced::widget::text(format_sticker_scale(self.sticker_scale)),
                            iced::widget::tooltip::Position::FollowCursor,
                        )
                        .delay(tooltip_delay(self.sticker_scale_adjusting))
                        .style(iced::widget::container::rounded_box),
                    ),
            )
            .push(iced::widget::Space::new().height(SLIDERS_PANEL_GAP))
            .push(
                Column::new()
                    .spacing(5)
                    .push(iced::widget::text("Face Gap"))
                    .push(
                        iced::widget::tooltip(
                            Slider::new(0.0..=1.5, self.face_gap, Message::FaceGap)
                                .step(0.01f32)
                                .width(250)
                                .on_release(Message::FaceGapReleased),
                            iced::widget::text(format_face_gap(self.face_gap)),
                            iced::widget::tooltip::Position::FollowCursor,
                        )
                        .delay(tooltip_delay(self.face_gap_adjusting))
                        .style(iced::widget::container::rounded_box),
                    ),
            )
            .push(iced::widget::Space::new().height(SLIDERS_PANEL_GAP))
            .push(
                Column::new()
                    .spacing(5)
                    .push(iced::widget::text("4D Face Gap"))
                    .push(
                        iced::widget::tooltip(
                            Slider::new(1.0..=2.0, self.face_gap_4d, Message::FaceGap4d)
                                .step(0.01f32)
                                .width(250)
                                .on_release(Message::FaceGap4dReleased),
                            iced::widget::text(format_face_gap_4d(self.face_gap_4d)),
                            iced::widget::tooltip::Position::FollowCursor,
                        )
                        .delay(tooltip_delay(self.face_gap_4d_adjusting))
                        .style(iced::widget::container::rounded_box),
                    ),
            )
            .push(iced::widget::Space::new().height(SLIDERS_PANEL_GAP))
            .push(
                Column::new()
                    .spacing(5)
                    .push(iced::widget::text("4D Viewer Distance"))
                    .push(
                        iced::widget::tooltip(
                            Slider::new(2.0..=10.0, self.viewer_distance, Message::ViewerDistance)
                                .step(0.01f32)
                                .width(250)
                                .on_release(Message::ViewerDistanceReleased),
                            iced::widget::text(format_viewer_distance(self.viewer_distance)),
                            iced::widget::tooltip::Position::FollowCursor,
                        )
                        .delay(tooltip_delay(self.viewer_distance_adjusting))
                        .style(iced::widget::container::rounded_box),
                    ),
            )
            .push(iced::widget::Space::new().height(SLIDERS_PANEL_GAP));

        let reveal_group = Column::new()
            .spacing(0)
            .push(
                Button::new(reveal_button_label(self.revealed, self.reveal_animating))
                    .on_press_maybe((!self.reveal_animating).then_some(Message::ToggleReveal)),
            )
            .push(
                iced::widget::scrollable(
                    iced::widget::mouse_area(sliders)
                        .on_scroll(|_delta| Message::SlidersPanelWheelScroll),
                )
                .height(Length::Fixed(lerp(
                    SLIDERS_PANEL_CLOSED_HEIGHT,
                    SLIDERS_PANEL_OPEN_HEIGHT,
                    self.reveal_panel_fraction,
                )))
                .direction(iced::widget::scrollable::Direction::Vertical(
                    iced::widget::scrollable::Scrollbar::hidden(),
                )),
            )
            .push(
                Column::new()
                    .spacing(5)
                    .push(iced::widget::text("Animation Duration (ms)"))
                    .push(
                        iced::widget::tooltip(
                            Slider::new(
                                ANIMATION_DURATION_MS_RANGE,
                                self.settings.animation_duration_ms,
                                Message::AnimationDuration,
                            )
                            .step(10u32)
                            .width(250)
                            .on_release(Message::AnimationDurationReleased),
                            iced::widget::text(format_animation_duration(
                                self.settings.animation_duration_ms,
                            )),
                            iced::widget::tooltip::Position::FollowCursor,
                        )
                        .delay(tooltip_delay(self.animation_duration_adjusting))
                        .style(iced::widget::container::rounded_box),
                    ),
            );

        controls = controls.push(reveal_group);

        // Right pane with 3D viewport
        let viewport = Shader::new(HypercubeShaderProgram::new(
            // Invert value since the slider can't work in reverse.
            1.0 - self.sticker_scale,
            self.face_gap,
            self.face_gap_4d,
            self.viewer_distance,
            self.render_mode,
            self.settings.theme,
            self.aabb_mode,
            self.fire_ground_truth_debug,
            self.settings.show_gizmo_ring,
            self.settings.rotate_button,
            self.settings.animation_duration_ms,
            self.reset_generation,
            self.random_moves_generation,
            self.pending_random_move_count,
            self.reveal_generation,
            self.revealed,
            self.save_generation,
            self.load_generation,
            self.pending_load.clone(),
            self.save_snapshot_generation,
            self.solve_command_generation,
            self.solve_command,
        ))
        .width(Length::Fill)
        .height(Length::Fill);

        // Main layout: menu bar above menu bar + left controls + right viewport
        let main_row = Row::new()
            .spacing(10)
            .padding(10)
            .push(
                iced::widget::container(controls)
                    .width(Length::Shrink)
                    .height(Length::Fill),
            )
            .push(viewport);

        let menu_bar = menu_overlay::bar(
            self.settings.debug_mode,
            self.render_mode,
            self.aabb_mode,
            self.revealed,
            self.reveal_animating,
            self.solving,
        );

        let content: Element<'_, Message> = Column::new().push(menu_bar).push(main_row).into();

        // Always a 4-layer stack regardless of `about_open`/`debug_mode`/
        // solve state, not a conditional stack - keeps `content`'s
        // widget-tree position stable.
        let solve_layer: Element<'_, Message> = match solve_overlay_text(
            self.solving,
            self.solve_progress,
            self.solve_notice.as_ref().map(|(text, _)| text.as_str()),
        ) {
            Some(text) => iced::widget::container(
                iced::widget::container(iced::widget::text(text))
                    .padding(6)
                    .style(iced::widget::container::rounded_box),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(iced::alignment::Horizontal::Right)
            .align_y(iced::alignment::Vertical::Bottom)
            .padding(10)
            .into(),
            None => Space::new().into(),
        };

        let about_layer: Element<'_, Message> = if self.about_open {
            about_modal()
        } else {
            Space::new().into()
        };

        let fps_layer: Element<'_, Message> = if self.settings.debug_mode {
            iced::widget::container(
                iced::widget::container(iced::widget::text(format!("{:.0} FPS", self.fps)))
                    .padding(6)
                    .style(iced::widget::container::rounded_box),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(iced::alignment::Horizontal::Right)
            .align_y(iced::alignment::Vertical::Top)
            .padding(10)
            .into()
        } else {
            Space::new().into()
        };

        iced::widget::stack([content, solve_layer, about_layer, fps_layer]).into()
    }
}

/// A clickable, link-styled button that opens `url` in the system's
/// default browser via `Message::OpenUrl`.
/// Text color for a `link_button`, changing on hover so a link reads as
/// clickable both from its underline and from a color shift.
///
/// `extended_palette().primary.base.color` (the raw accent color) isn't
/// vetted for contrast against any particular surface, so it can wash out
/// against the About popup's `container::rounded_box` background
/// (`background.weak.color`). `palette::readable` nudges it toward
/// legibility against that exact background instead, and `palette::deviate`
/// gives the hover state a visibly different, still-readable shade.
fn link_button_style(
    theme: &iced::Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    use iced::theme::palette;

    let extended = theme.extended_palette();
    let readable_link =
        palette::readable(extended.background.weak.color, extended.primary.base.color);
    let color = match status {
        iced::widget::button::Status::Hovered => palette::deviate(readable_link, 0.1),
        _ => readable_link,
    };

    iced::widget::button::Style {
        text_color: color,
        ..iced::widget::button::Style::default()
    }
}

fn link_button<'a>(label: &'static str, url: &'static str) -> Element<'a, Message> {
    let underlined_label = iced::widget::text::Rich::<(), Message>::with_spans([
        iced::widget::text::Span::new(label).underline(true),
    ])
    .size(14);

    Button::new(underlined_label)
        .padding(0)
        .style(link_button_style)
        .on_press(Message::OpenUrl(url))
        .into()
}

/// One license/attribution entry: a line of descriptive text followed by
/// its source and/or license link(s).
fn license_entry<'a>(
    description: &'static str,
    links: impl IntoIterator<Item = Element<'a, Message>>,
) -> Element<'a, Message> {
    Column::new()
        .spacing(2)
        .push(iced::widget::text(description))
        .push(Row::with_children(links).spacing(12))
        .into()
}

/// The Help menu's "About" popup: version, the mouse/keyboard control
/// scheme, and license/attribution details for the project and the
/// external code and assets it incorporates.
fn about_modal<'a>() -> Element<'a, Message> {
    let content = Column::new()
        .spacing(10)
        .padding(20)
        .push(iced::widget::text("4D Hypercube").size(24))
        .push(iced::widget::text(format!(
            "Version {}",
            env!("CARGO_PKG_VERSION")
        )))
        .push(iced::widget::text(
            "Drag with the rotate button to orbit in 3D.",
        ))
        .push(iced::widget::text(
            "Hold Shift while dragging to rotate in 4D.",
        ))
        .push(iced::widget::text(
            "Click a facet with the other mouse button to turn that side.",
        ))
        .push(iced::widget::text("Double-click a face to center it."))
        .push(iced::widget::text(
            "Puzzle > Solve plays back a solution; Esc stops it.",
        ))
        .push(iced::widget::text(
            "Ctrl+S save puzzle, Ctrl+O load puzzle, Ctrl+Q quit.",
        ))
        .push(iced::widget::rule::horizontal(1))
        .push(license_entry(
            "This project is dual-licensed under MIT or Apache 2.0.",
            [
                link_button("MIT", "https://opensource.org/license/mit"),
                link_button("Apache 2.0", "https://www.apache.org/licenses/LICENSE-2.0"),
            ],
        ))
        .push(license_entry(
            "Solver: a port of NdSolve by Don Hatch, from Magic Cube 4D by \
             Melinda Green & Don Hatch.",
            [
                link_button("Project page", "http://superliminal.com/cube/cube.htm"),
                link_button(
                    "License",
                    "https://github.com/cutelyaware/magiccube4d/blob/master/LICENSE.md",
                ),
            ],
        ))
        .push(license_entry(
            "Water shader by TDM, licensed under CC BY-NC-SA 3.0.",
            [
                link_button("Source", "https://www.shadertoy.com/view/Ms2SD1"),
                link_button(
                    "License",
                    "https://creativecommons.org/licenses/by-nc-sa/3.0/legalcode.en",
                ),
            ],
        ))
        .push(license_entry(
            "Ice shader by Sébastien Bérubé (Bers), licensed under CC BY-NC 4.0.",
            [
                link_button("Source", "https://www.shadertoy.com/view/MscXzn"),
                link_button(
                    "License",
                    "https://creativecommons.org/licenses/by-nc/4.0/legalcode",
                ),
            ],
        ))
        .push(license_entry(
            "Lightning shader by MonsterMan (no license stated).",
            [link_button(
                "Source",
                "https://www.shadertoy.com/view/dsXfDn",
            )],
        ))
        .push(license_entry(
            "Skybox by Screaming Brain Studios, licensed under CC0.",
            [link_button(
                "Source",
                "https://opengameart.org/content/cloudy-skyboxes-0",
            )],
        ))
        .push(Button::new("Close").on_press(Message::CloseAbout));

    let popup = iced::widget::container(iced::widget::scrollable(content))
        .width(480)
        .max_height(560)
        .style(iced::widget::container::rounded_box);

    iced::widget::opaque(
        iced::widget::mouse_area(iced::widget::center(iced::widget::opaque(popup)))
            .on_press(Message::CloseAbout),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_sticker_scale_rounds_to_two_decimals() {
        assert_eq!(format_sticker_scale(0.4234), "0.58");
    }

    #[test]
    fn format_sticker_scale_formats_range_endpoints() {
        assert_eq!(format_sticker_scale(0.0), "1.00");
        assert_eq!(format_sticker_scale(0.9), "0.10");
    }

    #[test]
    fn format_face_gap_rounds_to_two_decimals() {
        assert_eq!(format_face_gap(1.5), "1.50");
    }

    #[test]
    fn format_animation_duration_appends_ms_suffix_at_range_bounds() {
        assert_eq!(format_animation_duration(100), "100ms");
        assert_eq!(format_animation_duration(600), "600ms");
    }

    #[test]
    fn tooltip_delay_is_zero_while_adjusting() {
        assert_eq!(tooltip_delay(true), Duration::ZERO);
    }

    #[test]
    fn tooltip_delay_is_nonzero_while_idle() {
        assert_eq!(tooltip_delay(false), Duration::from_millis(400));
    }

    #[test]
    fn reveal_button_label_reads_settled_state_while_idle() {
        assert_eq!(reveal_button_label(false, false), "Reveal");
        assert_eq!(reveal_button_label(true, false), "Hide");
    }

    #[test]
    fn reveal_button_label_keeps_pre_press_text_while_animating() {
        // revealed already flipped to the target, but the flourish hasn't
        // settled yet - label should still read the state being left.
        assert_eq!(reveal_button_label(true, true), "Reveal");
        assert_eq!(reveal_button_label(false, true), "Hide");
    }

    #[test]
    fn reveal_animation_tick_interpolates_between_primary_and_secondary_when_revealing() {
        let mut app = HypercubeApp::new_inner();
        let _ = app.update(Message::ToggleReveal);
        let started = app
            .reveal_animation_started
            .expect("ToggleReveal must record a start instant");

        let _ = app.update(Message::RevealAnimationTick(started));
        assert_eq!(app.sticker_scale, PRIMARY_STICKER_SCALE);
        assert_eq!(app.face_gap, PRIMARY_FACE_GAP);
        assert_eq!(app.face_gap_4d, PRIMARY_FACE_GAP_4D);
        assert_eq!(app.reveal_panel_fraction, 0.0);

        let _ = app.update(Message::RevealAnimationTick(
            started + REVEAL_ANIMATION_DURATION,
        ));
        assert_eq!(app.sticker_scale, SECONDARY_STICKER_SCALE);
        assert_eq!(app.face_gap, SECONDARY_FACE_GAP);
        assert_eq!(app.face_gap_4d, SECONDARY_FACE_GAP_4D);
        assert_eq!(app.reveal_panel_fraction, 1.0);

        app.reveal_animation_started = Some(started);
        let _ = app.update(Message::RevealAnimationTick(
            started + REVEAL_ANIMATION_DURATION / 2,
        ));
        assert!(app.sticker_scale > PRIMARY_STICKER_SCALE);
        assert!(app.sticker_scale < SECONDARY_STICKER_SCALE);
        assert!(app.face_gap > PRIMARY_FACE_GAP);
        assert!(app.face_gap < SECONDARY_FACE_GAP);
        assert!(app.face_gap_4d > PRIMARY_FACE_GAP_4D);
        assert!(app.face_gap_4d < SECONDARY_FACE_GAP_4D);
        // `panel_progress` has its own dedicated coverage above; this only
        // confirms the tick handler actually wires it into the field.
        assert_eq!(app.reveal_panel_fraction, ease(panel_progress(0.5, true)));
    }

    #[test]
    fn reveal_animation_tick_interpolates_from_secondary_to_primary_when_hiding() {
        let mut app = HypercubeApp::new_inner();
        app.sticker_scale = SECONDARY_STICKER_SCALE;
        app.face_gap = SECONDARY_FACE_GAP;
        app.face_gap_4d = SECONDARY_FACE_GAP_4D;
        app.revealed = true;
        let _ = app.update(Message::ToggleReveal);
        assert!(!app.revealed);
        let started = app
            .reveal_animation_started
            .expect("ToggleReveal must record a start instant");

        let _ = app.update(Message::RevealAnimationTick(
            started + REVEAL_ANIMATION_DURATION,
        ));
        assert!((app.sticker_scale - PRIMARY_STICKER_SCALE).abs() < 1e-6);
        assert!((app.face_gap - PRIMARY_FACE_GAP).abs() < 1e-6);
        assert!((app.face_gap_4d - PRIMARY_FACE_GAP_4D).abs() < 1e-6);
        assert!((app.reveal_panel_fraction - 0.0).abs() < 1e-6);
    }

    #[test]
    fn panel_progress_opens_over_the_leading_window_when_revealing() {
        assert_eq!(panel_progress(0.0, true), 0.0);
        assert_eq!(panel_progress(PANEL_ANIMATION_FRACTION / 2.0, true), 0.5);
        assert_eq!(panel_progress(PANEL_ANIMATION_FRACTION, true), 1.0);
        assert_eq!(panel_progress(1.0, true), 1.0);
    }

    #[test]
    fn panel_progress_closes_over_the_trailing_window_when_hiding() {
        let window_start = 1.0 - PANEL_ANIMATION_FRACTION;
        assert_eq!(panel_progress(0.0, false), 0.0);
        assert_eq!(panel_progress(window_start, false), 0.0);
        assert!(
            (panel_progress(window_start + PANEL_ANIMATION_FRACTION / 2.0, false) - 0.5).abs()
                < 1e-6
        );
        assert_eq!(panel_progress(1.0, false), 1.0);
    }

    #[test]
    fn manual_slider_drags_are_ignored_while_reveal_animation_plays() {
        let mut app = HypercubeApp::new_inner();
        app.reveal_animating = true;
        let scale_before = app.sticker_scale;
        let gap_before = app.face_gap;

        let _ = app.update(Message::StickerScale(0.5));
        let _ = app.update(Message::FaceGap(0.9));

        assert_eq!(app.sticker_scale, scale_before);
        assert_eq!(app.face_gap, gap_before);
    }

    #[test]
    fn solve_starts_once_and_stop_cancels_it() {
        let mut app = HypercubeApp::new_inner();
        let _ = app.update(Message::StopSolving);
        assert_eq!(
            app.solve_command_generation, 0,
            "stopping while idle is a no-op"
        );

        let _ = app.update(Message::Solve);
        assert!(app.solving);
        assert_eq!(app.solve_command, SolveCommand::Start);
        assert_eq!(app.solve_command_generation, 1);

        let _ = app.update(Message::Solve);
        assert_eq!(app.solve_command_generation, 1, "already solving");

        let _ = app.update(Message::StopSolving);
        assert!(!app.solving);
        assert_eq!(app.solve_command, SolveCommand::Stop);
        assert_eq!(app.solve_command_generation, 2);
    }

    #[test]
    fn stale_and_post_cancel_solve_messages_are_ignored() {
        let mut app = HypercubeApp::new_inner();
        let _ = app.update(Message::Solve);
        let progress = |generation| Message::SolveProgress {
            generation,
            stage: Stage::Position(2),
            done: 3,
            total: 9,
        };

        let _ = app.update(progress(0));
        assert_eq!(app.solve_progress, None, "wrong generation");
        let _ = app.update(progress(1));
        assert_eq!(app.solve_progress, Some((Stage::Position(2), 3, 9)));

        let _ = app.update(Message::Reset);
        assert!(!app.solving);
        assert_eq!(app.solve_progress, None);
        let _ = app.update(progress(1));
        let _ = app.update(Message::SolveEnded {
            generation: 1,
            outcome: SolveOutcome::Completed { total: 9 },
        });
        assert_eq!(app.solve_progress, None);
        assert_eq!(app.solve_notice, None);

        let _ = app.update(Message::Solve);
        let _ = app.update(Message::RandomMoves(1));
        assert!(!app.solving, "random moves cancel a solve too");
    }

    #[test]
    fn solve_ended_posts_the_matching_notice() {
        use crate::solver::Unsolvable;
        for (outcome, expected) in [
            (
                SolveOutcome::Completed { total: 1904 },
                "Solved in 1,904 moves",
            ),
            (SolveOutcome::AlreadySolved, "Already solved"),
            (
                SolveOutcome::Failed(SolveError::Unsolvable(Unsolvable::TwirlParity)),
                "Can't solve: a corner piece is twisted",
            ),
        ] {
            let mut app = HypercubeApp::new_inner();
            let _ = app.update(Message::Solve);
            let _ = app.update(Message::SolveEnded {
                generation: app.solve_command_generation,
                outcome,
            });
            assert!(!app.solving);
            assert_eq!(
                app.solve_notice.as_ref().map(|(text, _)| text.as_str()),
                Some(expected)
            );
        }
    }

    #[test]
    fn solve_notice_expires_after_its_duration() {
        let mut app = HypercubeApp::new_inner();
        let posted = Instant::now();
        app.solve_notice = Some(("Already solved".to_string(), posted));
        let _ = app.update(Message::SolveNoticeTick(posted + SOLVE_NOTICE_DURATION / 2));
        assert!(app.solve_notice.is_some());
        let _ = app.update(Message::SolveNoticeTick(posted + SOLVE_NOTICE_DURATION));
        assert!(app.solve_notice.is_none());
    }

    #[test]
    fn format_count_groups_thousands() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1000), "1,000");
        assert_eq!(format_count(1904), "1,904");
        assert_eq!(format_count(1_234_567), "1,234,567");
    }

    #[test]
    fn solve_labels_and_overlay_text() {
        assert_eq!(solve_button_label(false), "Solve");
        assert_eq!(solve_button_label(true), "Stop Solving");

        assert_eq!(
            solve_overlay_text(true, Some((Stage::Orient(3), 812, 1904)), None).as_deref(),
            Some("Solving: orienting 3-sticker pieces \u{b7} move 812 / 1,904")
        );
        assert_eq!(
            solve_overlay_text(true, None, None).as_deref(),
            Some("Solving\u{2026}")
        );
        assert_eq!(
            solve_overlay_text(false, None, Some("Already solved")).as_deref(),
            Some("Already solved")
        );
        assert_eq!(solve_overlay_text(false, None, None), None);
        assert_eq!(
            solve_notice_text(&SolveOutcome::Completed { total: 1 }),
            "Solved in 1 move"
        );
    }

    #[cfg(feature = "gpu-capture-hooks")]
    #[test]
    fn next_reveal_loop_action_repeats_until_remaining_is_exhausted() {
        assert_eq!(next_reveal_loop_action(4), RevealLoopAction::Repeat);
        assert_eq!(next_reveal_loop_action(1), RevealLoopAction::Repeat);
        assert_eq!(next_reveal_loop_action(0), RevealLoopAction::Exit);
    }
}
