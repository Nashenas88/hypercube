//! Gui elements and messaging for the application

use std::time::Duration;

use iced::widget::{Button, Checkbox, Column, PickList, Row, Shader, Slider};
use iced::{Element, Length, Task};

use crate::settings::{self, ANIMATION_DURATION_MS_RANGE, AppSettings, RotateButton};
use crate::shader_widget::{HypercubeShaderProgram, PRIMARY_FACE_GAP, PRIMARY_STICKER_SCALE};

/// Rendering modes for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RenderMode {
    Standard,
    Normals,
    Depth,
}

/// AABB visualization modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    const ALL: [RenderMode; 3] = [RenderMode::Standard, RenderMode::Normals, RenderMode::Depth];
}

/// Move count for the "Scramble" button. 25 mixes a 27-piece side several
/// times over (180-degree edge and 120-degree corner turns disturb most of
/// a side per move), enough that the puzzle reads as thoroughly shuffled
/// without an excessive click-to-solved feel for manual play.
const SCRAMBLE_MOVE_COUNT: u32 = 25;

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
    const ALL: [AABBMode; 3] = [AABBMode::None, AABBMode::Face, AABBMode::Sticker];
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
fn reveal_button_label(revealed: bool, reveal_animating: bool) -> &'static str {
    match (revealed, reveal_animating) {
        (true, false) => "Hide",
        (false, false) => "Reveal",
        (true, true) => "Reveal",
        (false, true) => "Hide",
    }
}

/// Whether the sticker-scale/face-gap sliders should be shown: only once a
/// reveal has settled, hidden again the instant a hide flourish starts.
fn sliders_visible(revealed: bool, reveal_animating: bool) -> bool {
    revealed && !reveal_animating
}

/// Main application state - handles UI controls only
#[derive(Debug)]
pub(crate) struct HypercubeApp {
    sticker_scale: f32,
    face_gap: f32,
    render_mode: RenderMode,
    aabb_mode: AABBMode,
    debug_mode: bool,
    settings: AppSettings,
    reset_generation: u64,
    random_moves_generation: u64,
    /// Move count carried alongside `random_moves_generation` for the shader
    /// program to pick up, since a bare generation bump carries no payload
    /// (mirrors how `revealed` is threaded alongside `reveal_generation`).
    pending_random_move_count: u32,
    sticker_scale_adjusting: bool,
    face_gap_adjusting: bool,
    animation_duration_adjusting: bool,
    /// Target reveal state. Flips immediately on `ToggleReveal` (so the
    /// shader program picks up the new direction that same frame), not only
    /// once the flourish settles.
    revealed: bool,
    reveal_generation: u64,
    /// True from a `ToggleReveal` press until `RevealAnimationComplete`
    /// arrives; gates the button (disabled) and the sliders (hidden).
    reveal_animating: bool,
    /// Remaining scripted reveal/hide flourishes after the one the boot task
    /// already kicked off. See [`next_reveal_loop_action`].
    #[cfg(feature = "gpu-capture-hooks")]
    reveal_loop_remaining: u32,
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
    RenderMode(RenderMode),
    AABBMode(AABBMode),
    DebugMode(bool),
    RotateButton(RotateButton),
    AnimationDuration(u32),
    AnimationDurationReleased,
    Reset,
    RandomMoves(u32),
    ToggleReveal,
    RevealAnimationComplete { final_scale: f32, final_gap: f32 },
}

impl HypercubeApp {
    fn new_inner() -> Self {
        Self {
            sticker_scale: PRIMARY_STICKER_SCALE,
            face_gap: PRIMARY_FACE_GAP,
            render_mode: RenderMode::Standard,
            aabb_mode: AABBMode::None,
            debug_mode: false,
            settings: settings::load(),
            reset_generation: 0,
            random_moves_generation: 0,
            pending_random_move_count: 0,
            sticker_scale_adjusting: false,
            face_gap_adjusting: false,
            animation_duration_adjusting: false,
            revealed: false,
            reveal_generation: 0,
            reveal_animating: false,
            #[cfg(feature = "gpu-capture-hooks")]
            reveal_loop_remaining: REVEAL_LOOP_REPEATS,
        }
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
                self.sticker_scale = value;
                self.sticker_scale_adjusting = true;
            }
            Message::StickerScaleReleased => {
                self.sticker_scale_adjusting = false;
            }
            Message::FaceGap(value) => {
                self.face_gap = value;
                self.face_gap_adjusting = true;
            }
            Message::FaceGapReleased => {
                self.face_gap_adjusting = false;
            }
            Message::RenderMode(mode) => {
                self.render_mode = mode;
            }
            Message::AABBMode(mode) => {
                self.aabb_mode = mode;
            }
            Message::DebugMode(enabled) => {
                self.debug_mode = enabled;
            }
            Message::RotateButton(button) => {
                self.settings.rotate_button = button;
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
            }
            Message::RandomMoves(count) => {
                self.pending_random_move_count = count;
                self.random_moves_generation = self.random_moves_generation.wrapping_add(1);
            }
            Message::ToggleReveal => {
                self.revealed = !self.revealed;
                self.reveal_generation = self.reveal_generation.wrapping_add(1);
                self.reveal_animating = true;
            }
            Message::RevealAnimationComplete {
                final_scale,
                final_gap,
            } => {
                self.sticker_scale = final_scale;
                self.face_gap = final_gap;
                self.reveal_animating = false;

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
        }

        Task::none()
    }

    /// Create the view for the application
    pub(crate) fn view(&self) -> Element<'_, Message> {
        // Left pane with controls
        let mut controls = Column::new()
            .spacing(20)
            .push(
                Checkbox::new(self.debug_mode)
                    .label("Debug Mode")
                    .on_toggle(Message::DebugMode),
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
            .push(Button::new("Reset").on_press(Message::Reset))
            .push(
                Column::new()
                    .spacing(5)
                    .push(Button::new("1 Random Move").on_press(Message::RandomMoves(1)))
                    .push(Button::new("2 Random Moves").on_press(Message::RandomMoves(2)))
                    .push(Button::new("3 Random Moves").on_press(Message::RandomMoves(3)))
                    .push(
                        Button::new("Scramble").on_press(Message::RandomMoves(SCRAMBLE_MOVE_COUNT)),
                    ),
            );

        if self.debug_mode {
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
                );
        }

        controls = controls.push(
            Button::new(reveal_button_label(self.revealed, self.reveal_animating))
                .on_press_maybe((!self.reveal_animating).then_some(Message::ToggleReveal)),
        );

        if sliders_visible(self.revealed, self.reveal_animating) {
            controls = controls
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
                );
        }

        controls = controls.push(
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

        // Right pane with 3D viewport
        let viewport = Shader::new(HypercubeShaderProgram::new(
            // Invert value since the slider can't work in reverse.
            1.0 - self.sticker_scale,
            self.face_gap,
            self.render_mode,
            self.aabb_mode,
            self.settings.rotate_button,
            self.settings.animation_duration_ms,
            self.reset_generation,
            self.random_moves_generation,
            self.pending_random_move_count,
            self.reveal_generation,
            self.revealed,
        ))
        .width(Length::Fill)
        .height(Length::Fill);

        // Main layout: left controls + right viewport
        Row::new()
            .spacing(10)
            .padding(10)
            .push(
                iced::widget::container(controls)
                    .width(Length::Shrink)
                    .height(Length::Fill),
            )
            .push(viewport)
            .into()
    }
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
    fn sliders_visible_only_once_revealed_and_settled() {
        assert!(!sliders_visible(false, false));
        assert!(!sliders_visible(true, true));
        assert!(!sliders_visible(false, true));
        assert!(sliders_visible(true, false));
    }

    #[cfg(feature = "gpu-capture-hooks")]
    #[test]
    fn next_reveal_loop_action_repeats_until_remaining_is_exhausted() {
        assert_eq!(next_reveal_loop_action(4), RevealLoopAction::Repeat);
        assert_eq!(next_reveal_loop_action(1), RevealLoopAction::Repeat);
        assert_eq!(next_reveal_loop_action(0), RevealLoopAction::Exit);
    }
}
