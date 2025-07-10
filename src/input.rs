//! Input handling abstractions for the hypercube application.
//! 
//! This module provides traits and state management for processing user input
//! from mouse and keyboard events in a clean, decoupled manner.

use winit::event::{WindowEvent, DeviceEvent, MouseButton, ElementState};
use winit::keyboard::ModifiersState;

/// Trait for objects that can handle input events.
/// 
/// Provides a clean interface for routing input events to application components
/// without tight coupling to specific input handling implementations.
pub(crate) trait InputHandler {
    /// Handles window-specific input events like mouse clicks and scrolling.
    /// 
    /// # Arguments
    /// * `event` - The window event to process
    /// 
    /// # Returns
    /// `true` if the event was handled, `false` if it should be processed elsewhere
    fn handle_window_event(&mut self, event: &WindowEvent) -> bool;
    
    /// Handles device-level input events like mouse movement.
    /// 
    /// # Arguments
    /// * `event` - The device event to process
    /// * `modifiers` - Current state of modifier keys (Ctrl, Shift, etc.)
    /// 
    /// # Returns
    /// `true` if the event was handled, `false` if it should be processed elsewhere
    fn handle_device_event(&mut self, event: &DeviceEvent, modifiers: &ModifiersState) -> bool;
}

/// Tracks the current state of user input devices.
/// 
/// Maintains persistent state for input devices that need to be tracked
/// across multiple events (e.g., mouse button press/release pairs).
pub(crate) struct InputState {
    /// Whether the right mouse button is currently pressed
    pub(crate) is_right_mouse_pressed: bool,
}

impl InputState {
    /// Creates a new input state with default values.
    pub(crate) fn new() -> Self {
        Self {
            is_right_mouse_pressed: false,
        }
    }

    /// Updates the tracked mouse button state.
    /// 
    /// # Arguments
    /// * `button` - The mouse button that changed state
    /// * `state` - Whether the button was pressed or released
    pub(crate) fn update_mouse_state(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Right {
            self.is_right_mouse_pressed = state == ElementState::Pressed;
        }
    }
}