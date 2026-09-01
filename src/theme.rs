//! Selectable sticker-rendering themes.

use serde::{Deserialize, Serialize};

/// Which shader interprets a sticker's `kind` for rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub(crate) enum Theme {
    #[default]
    Classic,
    Elemental,
}

/// Particles emitted per sticker under `Theme::Elemental`, indexed by `kind`.
/// Currently one shared budget for every element; per-element budgets land
/// once particle emission is reauthored per element.
const ELEMENTAL_PARTICLES_PER_STICKER: u32 = 24;

impl Theme {
    pub(crate) const ALL: [Theme; 2] = [Theme::Classic, Theme::Elemental];

    /// Particle instances emitted per sticker of each `kind` (indexed 0..8),
    /// this theme's own emission budget. `Classic` draws no particles.
    pub(crate) fn particles_per_kind(&self) -> [u32; 8] {
        match self {
            Theme::Classic => [0; 8],
            Theme::Elemental => [ELEMENTAL_PARTICLES_PER_STICKER; 8],
        }
    }
}

impl std::fmt::Display for Theme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Theme::Classic => write!(f, "Classic"),
            Theme::Elemental => write!(f, "Elemental"),
        }
    }
}
