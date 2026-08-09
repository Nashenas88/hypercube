//! Persisted, user-configurable control bindings.

use std::path::PathBuf;

use iced::mouse;
use serde::{Deserialize, Serialize};

/// Which mouse button drives camera rotation (held + drag = 3D rotate, held + Shift + drag = 4D rotate).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub(crate) enum RotateButton {
    #[default]
    Left,
    Right,
}

impl RotateButton {
    pub(crate) const ALL: [RotateButton; 2] = [RotateButton::Left, RotateButton::Right];

    pub(crate) fn to_mouse_button(self) -> mouse::Button {
        match self {
            RotateButton::Left => mouse::Button::Left,
            RotateButton::Right => mouse::Button::Right,
        }
    }
}

impl std::fmt::Display for RotateButton {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RotateButton::Left => write!(f, "Left"),
            RotateButton::Right => write!(f, "Right"),
        }
    }
}

pub(crate) const ANIMATION_DURATION_MS_RANGE: std::ops::RangeInclusive<u32> = 100..=600;
const DEFAULT_ANIMATION_DURATION_MS: u32 = 250;

/// Settings persisted across application runs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct AppSettings {
    pub(crate) rotate_button: RotateButton,
    /// Duration of a move's turn animation, in milliseconds.
    pub(crate) animation_duration_ms: u32,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            rotate_button: RotateButton::default(),
            animation_duration_ms: DEFAULT_ANIMATION_DURATION_MS,
        }
    }
}

fn config_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "hypercube")
        .map(|dirs| dirs.config_dir().join("settings.toml"))
}

/// Loads settings from disk, falling back to defaults if the file is missing or invalid.
pub(crate) fn load() -> AppSettings {
    let Some(path) = config_path() else {
        log::warn!("Could not determine config directory; using default settings");
        return AppSettings::default();
    };

    match std::fs::read_to_string(&path) {
        Ok(contents) => match toml::from_str(&contents) {
            Ok(settings) => settings,
            Err(err) => {
                log::warn!("Failed to parse settings at {path:?}: {err}; using defaults");
                AppSettings::default()
            }
        },
        Err(err) => {
            log::warn!("Failed to read settings at {path:?}: {err}; using defaults");
            AppSettings::default()
        }
    }
}

/// Persists settings to disk, logging a warning on failure rather than propagating an error.
pub(crate) fn save(settings: &AppSettings) {
    let Some(path) = config_path() else {
        log::warn!("Could not determine config directory; settings not saved");
        return;
    };

    if let Some(parent) = path.parent()
        && let Err(err) = std::fs::create_dir_all(parent)
    {
        log::warn!("Failed to create config directory {parent:?}: {err}");
        return;
    }

    let contents = match toml::to_string_pretty(settings) {
        Ok(contents) => contents,
        Err(err) => {
            log::warn!("Failed to serialize settings: {err}");
            return;
        }
    };

    if let Err(err) = std::fs::write(&path, contents) {
        log::warn!("Failed to write settings to {path:?}: {err}");
    }
}
