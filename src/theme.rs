//! Selectable sticker-rendering themes.

use serde::{Deserialize, Serialize};

/// Which shader interprets a sticker's `kind` for rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub(crate) enum Theme {
    #[default]
    Classic,
    Elemental,
}

impl Theme {
    pub(crate) const ALL: [Theme; 2] = [Theme::Classic, Theme::Elemental];
}

impl std::fmt::Display for Theme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Theme::Classic => write!(f, "Classic"),
            Theme::Elemental => write!(f, "Elemental"),
        }
    }
}
