use winit::event::{WindowEvent, DeviceEvent, MouseButton, ElementState};
use winit::keyboard::ModifiersState;

pub trait InputHandler {
    fn handle_window_event(&mut self, event: &WindowEvent) -> bool;
    fn handle_device_event(&mut self, event: &DeviceEvent, modifiers: &ModifiersState) -> bool;
}

pub struct InputState {
    pub is_right_mouse_pressed: bool,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            is_right_mouse_pressed: false,
        }
    }

    pub fn update_mouse_state(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Right {
            self.is_right_mouse_pressed = state == ElementState::Pressed;
        }
    }
}