//! 4D Hypercube visualization application with iced UI.
//!
//! An interactive 4D Rubik's cube that can be rotated in 4D space and viewed
//! through 3D projection. Uses iced for UI and wgpu for GPU rendering.

use iced::widget::{Column, Row, Shader, Slider};
use iced::{Element, Length, Settings, Task};

mod camera;
mod cube;
mod math;
mod renderer;
mod shader_widget;

use shader_widget::HypercubeShaderProgram;

/// Main application state - handles UI controls only
#[derive(Debug)]
pub(crate) struct HypercubeApp {
    sticker_scale: f32,
    face_scale: f32,
}

/// Messages that the application can receive
#[derive(Debug, Clone)]
pub(crate) enum Message {
    StickerScaleChanged(f32),
    FaceScaleChanged(f32),
}

impl HypercubeApp {
    /// Create a new application instance
    pub(crate) fn new() -> Self {
        Self {
            sticker_scale: 0.8, // Default from existing code
            face_scale: 1.0,    // New parameter for future use
        }
    }

    /// Get the title of the application
    pub(crate) fn title(&self) -> &'static str {
        "4D Hypercube"
    }

    /// Update the application state
    pub(crate) fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::StickerScaleChanged(value) => {
                self.sticker_scale = value;
            }
            Message::FaceScaleChanged(value) => {
                self.face_scale = value;
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
                    .push(iced::widget::text("Sticker Scale"))
                    .push(
                        Slider::new(0.1..=1.0, self.sticker_scale, Message::StickerScaleChanged)
                            .step(0.01)
                            .width(250),
                    ),
            )
            .push(
                Column::new()
                    .spacing(5)
                    .push(iced::widget::text("Face Scale"))
                    .push(
                        Slider::new(0.1..=1.0, self.face_scale, Message::FaceScaleChanged)
                            .step(0.01)
                            .width(250),
                    ),
            );

        // Right pane with 3D viewport
        let viewport = Shader::new(HypercubeShaderProgram::new(
            self.sticker_scale,
            self.face_scale,
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
    env_logger::init();

    let app = HypercubeApp::new();
    iced::application(app.title(), HypercubeApp::update, HypercubeApp::view)
        .settings(Settings {
            antialiasing: true,
            ..Settings::default()
        })
        .run_with(move || (app, Task::none()))
}
