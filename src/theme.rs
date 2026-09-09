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
/// One shared budget for every element that still emits; Fire's, Water's,
/// Lightning's, Ice's, Dark's, Light's, Moss's and Dirt's own materials
/// carry their whole look, so they emit none.
const ELEMENTAL_PARTICLES_PER_STICKER: u32 = 24;

/// The `kind` `Theme::Elemental` renders as Moss, matching `case 1u` in
/// `elemental_shader.wgsl`. Moss's material bump-maps its own normal per
/// fragment and carries its whole look without particles, so - like Water -
/// it emits none.
pub(crate) const ELEMENTAL_MOSS_KIND: u32 = 1;

/// The `kind` `Theme::Elemental` renders as Ice, matching `case 0u` in
/// `elemental_shader.wgsl`. Ice's material raymarches its own refraction and
/// reflection and carries its whole look directly on the surface, so - like
/// Fire, Water and Lightning - it emits no particles.
pub(crate) const ELEMENTAL_ICE_KIND: u32 = 0;

/// The `kind` `Theme::Elemental` renders as Lightning, matching `case 2u` in
/// `elemental_shader.wgsl`. Lightning's material carries its whole look -
/// arcs and flash strikes - directly on the surface, so - like Fire and
/// Water - it emits no particles.
pub(crate) const ELEMENTAL_LIGHTNING_KIND: u32 = 2;

/// The `kind` `Theme::Elemental` renders as Fire, matching `case 3u` in
/// `elemental_shader.wgsl`. Fire is the one element drawn in its own blended
/// pass, so the CPU has to recognize it to build that pass's draw order.
pub(crate) const ELEMENTAL_FIRE_KIND: u32 = 3;

/// The `kind` `Theme::Elemental` renders as Water, matching `case 6u` in
/// `elemental_shader.wgsl`. Water's material bump-maps its own normal per
/// fragment and carries its whole look without particles, so - like Fire -
/// it emits none.
pub(crate) const ELEMENTAL_WATER_KIND: u32 = 6;

/// The `kind` `Theme::Elemental` renders as Dark, matching `case 7u` in
/// `elemental_shader.wgsl`. Dark's material raymarches its own portal
/// world and carries its whole look directly on the surface, so - like
/// Fire, Water, Lightning and Ice - it emits no particles.
pub(crate) const ELEMENTAL_DARK_KIND: u32 = 7;

/// The `kind` `Theme::Elemental` renders as Light, matching `case 5u` in
/// `elemental_shader.wgsl`. Light draws in its own blended, depth-sorted
/// pass alongside Fire and Ice (see `DepthLayer`), so the CPU has to
/// recognize it to build that pass's draw order.
pub(crate) const ELEMENTAL_LIGHT_KIND: u32 = 5;

/// The `kind` `Theme::Elemental` renders as Dirt, matching `case 4u` in
/// `elemental_shader.wgsl`. Dirt raymarches a bumpy rock surface in its own
/// local frame and carries its whole look directly on the surface, so -
/// like Fire, Water, Lightning, Ice, Dark and Light - it emits no
/// particles. Its rock box doesn't fill a sticker's full local extent, so
/// like Fire, Ice and Light it draws in its own blended, depth-sorted pass
/// (see `DepthLayer`), and the CPU has to recognize it to build that pass's
/// draw order.
pub(crate) const ELEMENTAL_DIRT_KIND: u32 = 4;

impl Theme {
    pub(crate) const ALL: [Theme; 2] = [Theme::Classic, Theme::Elemental];

    /// Particle instances emitted per sticker of each `kind` (indexed 0..8),
    /// this theme's own emission budget. `Classic` draws no particles.
    pub(crate) fn particles_per_kind(&self) -> [u32; 8] {
        match self {
            Theme::Classic => [0; 8],
            Theme::Elemental => {
                let mut counts = [ELEMENTAL_PARTICLES_PER_STICKER; 8];
                counts[ELEMENTAL_MOSS_KIND as usize] = 0;
                counts[ELEMENTAL_FIRE_KIND as usize] = 0;
                counts[ELEMENTAL_WATER_KIND as usize] = 0;
                counts[ELEMENTAL_LIGHTNING_KIND as usize] = 0;
                counts[ELEMENTAL_ICE_KIND as usize] = 0;
                counts[ELEMENTAL_DARK_KIND as usize] = 0;
                counts[ELEMENTAL_LIGHT_KIND as usize] = 0;
                counts[ELEMENTAL_DIRT_KIND as usize] = 0;
                counts
            }
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
