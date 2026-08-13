//! Gui elements and messaging for the application


use iced::widget::{Button, Checkbox, Column, PickList, Row, Shader, Slider};
use iced::{Element, Length, Task};

use crate::settings::{self, ANIMATION_DURATION_MS_RANGE, AppSettings, RotateButton};
use crate::shader_widget::HypercubeShaderProgram;

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
}

/// Messages that the application can receive
#[derive(Debug, Clone)]
pub(crate) enum Message {
    StickerScale(f32),
    FaceGap(f32),
    RenderMode(RenderMode),
    AABBMode(AABBMode),
    DebugMode(bool),
    RotateButton(RotateButton),
    AnimationDuration(u32),
    Reset,
}

impl HypercubeApp {
    /// Create a new application instance
    pub(crate) fn new() -> Self {
        Self {
            sticker_scale: 0.5, // Default from existing code
            face_gap: 0.4,
            render_mode: RenderMode::Standard,
            aabb_mode: AABBMode::None,
            debug_mode: false,
            settings: settings::load(),
            reset_generation: 0,
        }
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
            }
            Message::FaceGap(value) => {
                self.face_gap = value;
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
            }
            Message::Reset => {
                self.reset_generation = self.reset_generation.wrapping_add(1);
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
            .push(Button::new("Reset").on_press(Message::Reset));

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

        controls = controls
            .push(
                Column::new()
                    .spacing(5)
                    .push(iced::widget::text("Sticker Scale"))
                    .push(
                        Slider::new(0.0..=0.9, self.sticker_scale, Message::StickerScale)
                            .step(0.01f32)
                            .width(250),
                    ),
            )
            .push(
                Column::new()
                    .spacing(5)
                    .push(iced::widget::text("Face Gap"))
                    .push(
                        Slider::new(0.0..=1.5, self.face_gap, Message::FaceGap)
                            .step(0.01f32)
                            .width(250),
                    ),
            )
            .push(
                Column::new()
                    .spacing(5)
                    .push(iced::widget::text("Animation Duration (ms)"))
                    .push(
                        Slider::new(
                            ANIMATION_DURATION_MS_RANGE,
                            self.settings.animation_duration_ms,
                            Message::AnimationDuration,
                        )
                        .step(10u32)
                        .width(250),
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