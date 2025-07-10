//! 4D Hypercube visualization application.
//! 
//! An interactive 4D Rubik's cube that can be rotated in 4D space and viewed
//! through 3D projection. Uses wgpu for GPU rendering and provides intuitive
//! mouse controls for navigation.

use std::sync::Arc;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    keyboard::ModifiersState,
};

mod app;
mod camera;
mod cube;
mod input;
mod math;
mod renderer;

use app::App;
use input::InputHandler;
use renderer::Renderer;

/// Entry point for the hypercube visualization application.
/// 
/// Sets up the window, initializes the application and renderer, and runs
/// the main event loop for handling input and rendering frames.
fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        winit::window::WindowBuilder::new()
            .with_title("Hypercube")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .build(&event_loop)
            .unwrap(),
    );

    let window_size = window.inner_size();
    let mut app = App::new(window_size.width, window_size.height);
    let mut renderer = pollster::block_on(Renderer::new(window.clone(), &app.hypercube));
    let mut modifiers = ModifiersState::default();

    event_loop
        .run(move |event, elwt| match event {
            Event::WindowEvent {
                window_id,
                event,
            } if window_id == renderer.window().id() => {
                if !app.handle_window_event(&event) {
                    match event {
                        WindowEvent::CloseRequested => {
                            elwt.exit();
                        }
                        WindowEvent::Resized(physical_size) => {
                            app.resize(physical_size);
                            renderer.resize(physical_size);
                        }
                        WindowEvent::RedrawRequested => {
                            app.update();
                            renderer.update_instances(&app.hypercube, &app.rotation_4d);
                            match renderer.render(&app.camera, &app.projection) {
                                Ok(_) => {},
                                Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size),
                                Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                                Err(e) => eprintln!("{e:?}"),
                            }
                        }
                        WindowEvent::ModifiersChanged(new_modifiers) => {
                            modifiers = new_modifiers.state();
                        }
                        _ => {}
                    }
                }
            },
            Event::DeviceEvent { event, .. } => {
                app.handle_device_event(&event, &modifiers);
            }
            Event::AboutToWait => {
                renderer.window().request_redraw();
            }
            _ => {}
        })
        .unwrap();
}