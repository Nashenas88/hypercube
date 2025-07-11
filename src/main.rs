//! 4D Hypercube visualization application.
//!
//! An interactive 4D Rubik's cube that can be rotated in 4D space and viewed
//! through 3D projection. Uses wgpu for GPU rendering and provides intuitive
//! mouse controls for navigation.

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::ModifiersState,
    window::{Window, WindowId},
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

/// Application state that implements the ApplicationHandler trait.
struct HypercubeApp {
    window: Option<Arc<Window>>,
    app: Option<App>,
    renderer: Option<Renderer<'static>>,
    modifiers: ModifiersState,
}

impl HypercubeApp {
    fn new() -> Self {
        Self {
            window: None,
            app: None,
            renderer: None,
            modifiers: ModifiersState::default(),
        }
    }
}

impl ApplicationHandler for HypercubeApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("Hypercube")
                .with_inner_size(winit::dpi::LogicalSize::new(800, 600));

            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
            let window_size = window.inner_size();

            let app = App::new(window_size.width, window_size.height);
            let renderer = pollster::block_on(Renderer::new(window.clone(), &app.hypercube));

            self.window = Some(window);
            self.app = Some(app);
            self.renderer = Some(renderer);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let (Some(window), Some(app), Some(renderer)) =
            (&self.window, &mut self.app, &mut self.renderer)
        {
            if window_id == window.id() && !app.handle_window_event(&event) {
                match event {
                    WindowEvent::CloseRequested => {
                        event_loop.exit();
                    }
                    WindowEvent::Resized(physical_size) => {
                        app.resize(physical_size);
                        renderer.resize(physical_size);
                    }
                    WindowEvent::RedrawRequested => {
                        app.update();
                        renderer.update_instances(&app.hypercube, &app.rotation_4d);
                        match renderer.render(&app.camera, &app.projection) {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size),
                            Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                            Err(e) => eprintln!("{e:?}"),
                        }
                    }
                    WindowEvent::ModifiersChanged(new_modifiers) => {
                        self.modifiers = new_modifiers.state();
                    }
                    _ => {}
                }
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let Some(app) = &mut self.app {
            app.handle_device_event(&event, &self.modifiers);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

/// Entry point for the hypercube visualization application.
///
/// Sets up the event loop and runs the application using the new trait-based API.
fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    let mut app = HypercubeApp::new();
    event_loop.run_app(&mut app).unwrap();
}
