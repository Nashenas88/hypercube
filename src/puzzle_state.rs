//! Explicit save/load of the puzzle's piece arrangement.

use std::path::PathBuf;

use crate::piece::Hypercube;

fn config_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "hypercube")
        .map(|dirs| dirs.config_dir().join("puzzle_state.json"))
}

/// Loads a previously saved puzzle arrangement, if one exists and is valid.
pub(crate) fn load() -> Option<Hypercube> {
    let path = config_path()?;

    match std::fs::read_to_string(&path) {
        Ok(contents) => match serde_json::from_str(&contents) {
            Ok(hypercube) => Some(hypercube),
            Err(err) => {
                log::warn!("Failed to parse puzzle state at {path:?}: {err}");
                None
            }
        },
        Err(err) => {
            log::warn!("Failed to read puzzle state at {path:?}: {err}");
            None
        }
    }
}

/// Persists the puzzle's current arrangement to disk, logging a warning on
/// failure rather than propagating an error.
pub(crate) fn save(hypercube: &Hypercube) {
    let Some(path) = config_path() else {
        log::warn!("Could not determine config directory; puzzle state not saved");
        return;
    };

    if let Some(parent) = path.parent()
        && let Err(err) = std::fs::create_dir_all(parent)
    {
        log::warn!("Failed to create config directory {parent:?}: {err}");
        return;
    }

    let contents = match serde_json::to_string_pretty(hypercube) {
        Ok(contents) => contents,
        Err(err) => {
            log::warn!("Failed to serialize puzzle state: {err}");
            return;
        }
    };

    if let Err(err) = std::fs::write(&path, contents) {
        log::warn!("Failed to write puzzle state to {path:?}: {err}");
    }
}
