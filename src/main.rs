//! 4D Hypercube visualization application with iced UI.
//!
//! An interactive 4D Rubik's cube that can be rotated in 4D space and viewed
//! through 3D projection. Uses iced for UI and wgpu for GPU rendering.

use iced::widget::{Column, PickList, Row, Shader, Slider};
use iced::{Element, Length, Settings, Task};

mod camera;
mod cube;
mod math;
mod ray_casting;
mod renderer;
mod shader_widget;

use shader_widget::HypercubeShaderProgram;

/// Rendering modes for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RenderMode {
    Standard,
    Normals,
    Depth,
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

/// Main application state - handles UI controls only
#[derive(Debug)]
pub(crate) struct HypercubeApp {
    sticker_scale: f32,
    face_scale: f32,
    render_mode: RenderMode,
}

/// Messages that the application can receive
#[derive(Debug, Clone)]
pub(crate) enum Message {
    StickerScale(f32),
    FaceScale(f32),
    RenderMode(RenderMode),
}

impl HypercubeApp {
    /// Create a new application instance
    pub(crate) fn new() -> Self {
        Self {
            sticker_scale: 0.5, // Default from existing code
            face_scale: 2.0,    // New parameter for future use
            render_mode: RenderMode::Standard,
        }
    }

    /// Get the title of the application
    pub(crate) fn title(&self) -> &'static str {
        "4D Hypercube"
    }

    /// Update the application state
    pub(crate) fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::StickerScale(value) => {
                self.sticker_scale = value;
            }
            Message::FaceScale(value) => {
                self.face_scale = value;
            }
            Message::RenderMode(mode) => {
                self.render_mode = mode;
            }
        }

        Task::none()
    }

    /// Create the view for the application
    pub(crate) fn view(&self) -> Element<Message> {
        // Left pane with controls
        let controls = Column::new()
            .spacing(20)
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
                    .push(iced::widget::text("Sticker Scale"))
                    .push(
                        Slider::new(0.0..=0.9, self.sticker_scale, Message::StickerScale)
                            .step(0.01)
                            .width(250),
                    ),
            )
            .push(
                Column::new()
                    .spacing(5)
                    .push(iced::widget::text("Face Scale"))
                    .push(
                        Slider::new(1.0..=5.0, self.face_scale, Message::FaceScale)
                            .step(0.01)
                            .width(250),
                    ),
            );

        // Right pane with 3D viewport
        let viewport = Shader::new(HypercubeShaderProgram::new(
            // Invert value since the slider can't work in reverse.
            1.0 - self.sticker_scale,
            self.face_scale,
            self.render_mode,
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

/// Entry point for the hypercube visualization application
fn main() -> iced::Result {
    env_logger::builder().format_timestamp(None).init();

    let app = HypercubeApp::new();
    iced::application(app.title(), HypercubeApp::update, HypercubeApp::view)
        .settings(Settings {
            antialiasing: true,
            ..Settings::default()
        })
        .run_with(move || (app, Task::none()))
}
