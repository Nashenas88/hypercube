//! 4D Hypercube visualization application with iced UI.
//!
//! An interactive 4D Rubik's cube that can be rotated in 4D space and viewed
//! through 3D projection. Uses iced for UI and wgpu for GPU rendering.

use iced::Settings;

mod app;
mod camera;
mod geometry;
mod math;
mod moves;
mod piece;
mod ray_casting;
mod renderer;
mod settings;
mod shader_widget;

/// Entry point for the hypercube visualization application
fn main() -> iced::Result {
    env_logger::builder().format_timestamp(None).init();

    iced::application(
        app::HypercubeApp::new,
        app::HypercubeApp::update,
        app::HypercubeApp::view,
    )
    .title(app::HypercubeApp::title)
    .settings(Settings {
        antialiasing: true,
        ..Settings::default()
    })
    .run()
}
