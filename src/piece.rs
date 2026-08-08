//! Piece-based puzzle state.
//!
//! Replaces sticker-tracking with piece-tracking: each `Piece` carries a
//! lattice position (one of the 81 points in {-1,0,1}^4) and, for each axis
//! where its position is nonzero, the color of the facet currently facing
//! that axis's sign. A piece's `Vec` slot is always determined by its own
//! current position (see `index_of`), so moves never need to reorder the
//! `Vec` and two `Hypercube`s can be compared with a plain `assert_eq!`.

use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::geometry::Color;
use crate::math::GRID_EXTENT;

/// A single puzzle piece: its current lattice position and, per axis, the
/// color of the facet facing that axis (if any).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Piece {
    pub(crate) position: [i8; 4],
    pub(crate) colors: [Option<Color>; 4],
}

impl Piece {
    /// Number of nonzero axes in `position`, i.e. how many stickers this
    /// piece has: 0 = invisible center, 1 = cell-center, 2 = face, 3 = edge,
    /// 4 = corner.
    pub(crate) fn facet_count(&self) -> u8 {
        self.position.iter().filter(|c| **c != 0).count() as u8
    }
}

/// The complete piece-based puzzle state: always exactly 81 pieces (80
/// movable + 1 invisible center), canonically ordered by `index_of(position)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Hypercube {
    pub(crate) pieces: Vec<Piece>,
}

/// Instance data for the vertex shader - represents one rendered facet in 4D
/// space. Lives here (rather than in `renderer.rs`) because building the
/// full instance list is a puzzle-state concern: it walks `FACET_TABLE` and
/// looks up each facet's live color from a `Hypercube`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct StickerInstance {
    /// 4D position of the sticker
    pub(crate) position_4d: [f32; 4],
    /// RGBA color of the sticker
    pub(crate) color: [f32; 4],
    /// Face ID (0-7) for this sticker
    pub(crate) face_id: u32,
    /// Padding for alignment
    pub(crate) _padding: [u32; 3],
}

/// Colors for the 8 sides of the puzzle, indexed by `face_id_for`.
const COLORS: [Color; 8] = [
    Color::White,
    Color::Yellow,
    Color::Blue,
    Color::Green,
    Color::Red,
    Color::Orange,
    Color::Purple,
    Color::Brown,
];

/// Maps a (axis, sign) side to one of the 8 face ids, reproducing the same
/// axis/sign -> face pairing as the old `FACE_CENTERS`/`FIXED_DIMS` tables in
/// `geometry.rs` (face 0..4 are the `-1` sides in axis order W,Z,Y,X; face
/// 4..8 are the `+1` sides in axis order X,Y,Z,W).
pub(crate) fn face_id_for(axis: usize, sign: i8) -> usize {
    match (axis, sign) {
        (3, -1) => 0,
        (2, -1) => 1,
        (1, -1) => 2,
        (0, -1) => 3,
        (0, 1) => 4,
        (1, 1) => 5,
        (2, 1) => 6,
        (3, 1) => 7,
        _ => unreachable!("axis must be in 0..4 and sign must be -1 or 1"),
    }
}

/// The color assigned to a given (axis, sign) side.
pub(crate) fn side_color(axis: usize, sign: i8) -> Color {
    COLORS[face_id_for(axis, sign)]
}

/// The 3 axes other than `fixed`, in ascending order.
pub(crate) fn free_axes(fixed: usize) -> [usize; 3] {
    let mut out = [0usize; 3];
    let mut i = 0;
    for axis in 0..4 {
        if axis != fixed {
            out[i] = axis;
            i += 1;
        }
    }
    out
}

fn digit(c: i8) -> usize {
    (c + 1) as usize
}

fn undigit(d: usize) -> i8 {
    d as i8 - 1
}

/// Bijection from a lattice position to its canonical `Vec` slot (base-3
/// digit encoding, axis 0 most significant).
pub(crate) fn index_of(position: [i8; 4]) -> usize {
    ((digit(position[0]) * 3 + digit(position[1])) * 3 + digit(position[2])) * 3 + digit(position[3])
}

/// Inverse of `index_of`.
pub(crate) fn position_of(mut index: usize) -> [i8; 4] {
    let mut pos = [0i8; 4];
    for axis in (0..4).rev() {
        pos[axis] = undigit(index % 3);
        index /= 3;
    }
    pos
}

impl Hypercube {
    /// Builds the solved puzzle: all 81 lattice positions, each piece's
    /// colors matching `side_color` for every axis where its position is
    /// nonzero.
    pub(crate) fn solved() -> Self {
        let mut pieces = Vec::with_capacity(81);
        for x in -1..=1i8 {
            for y in -1..=1i8 {
                for z in -1..=1i8 {
                    for w in -1..=1i8 {
                        let position = [x, y, z, w];
                        let mut colors = [None; 4];
                        for axis in 0..4 {
                            if position[axis] != 0 {
                                colors[axis] = Some(side_color(axis, position[axis]));
                            }
                        }
                        pieces.push(Piece { position, colors });
                    }
                }
            }
        }
        Self { pieces }
    }

    /// True iff every piece's colors match its position's home-side colors,
    /// i.e. no piece has ever been moved out of its solved orientation.
    pub(crate) fn is_solved(&self) -> bool {
        self.pieces.iter().all(|p| {
            (0..4).all(|axis| match (p.position[axis], p.colors[axis]) {
                (0, None) => true,
                (n, Some(c)) if n != 0 => c == side_color(axis, n),
                _ => false,
            })
        })
    }
}

/// Total number of rendered facets: sum of `facet_count()` over all 81
/// pieces, i.e. 8*1 + 24*2 + 32*3 + 16*4 = 216 (unchanged from the old
/// sticker model's 8 faces * 27 stickers).
pub(crate) const NUM_FACETS: usize = 216;

/// Static, state-independent geometry for one rendered facet: everything
/// needed to place it, hit-test it, and know which move it triggers if
/// clicked, computed once from its `(piece_slot, axis)` identity. Only the
/// facet's color is live state, looked up from a `Hypercube` at render time.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FacetGeometry {
    /// Index into `Hypercube::pieces` for the piece this facet belongs to.
    pub(crate) piece_slot: usize,
    /// The axis this facet faces (`Piece::colors[axis]` is `Some` for it).
    pub(crate) axis: usize,
    /// Which of the 8 tesseract sides this facet renders on.
    pub(crate) face_id: usize,
    /// 4D render position, matching the old `Sticker::position` layout.
    pub(crate) position_4d: [f32; 4],
    /// True for face/edge/corner-type pieces (facet_count 2..=4) - the only
    /// pieces that can be clicked to trigger a move.
    pub(crate) is_actionable: bool,
    /// The sign of `position[axis]`, i.e. which side of `axis` this facet is on.
    pub(crate) side_sign: i8,
    /// The 3 axes other than `axis`, ascending - the move's rotation lives here.
    pub(crate) free_axes: [usize; 3],
    /// This piece's position restricted to `free_axes` - the move's rotation axis.
    pub(crate) local_coords: [i8; 3],
}

fn facet_position_4d(position: [i8; 4], fixed_axis: usize) -> [f32; 4] {
    let mut pos = [0.0f32; 4];
    for axis in 0..4 {
        pos[axis] = if axis == fixed_axis {
            position[axis] as f32
        } else {
            position[axis] as f32 * GRID_EXTENT
        };
    }
    pos
}

fn build_facet_table() -> [FacetGeometry; NUM_FACETS] {
    let mut table = Vec::with_capacity(NUM_FACETS);
    for piece_slot in 0..81 {
        let position = position_of(piece_slot);
        let facet_count = position.iter().filter(|c| **c != 0).count();
        for axis in 0..4 {
            if position[axis] == 0 {
                continue;
            }
            let axes = free_axes(axis);
            table.push(FacetGeometry {
                piece_slot,
                axis,
                face_id: face_id_for(axis, position[axis]),
                position_4d: facet_position_4d(position, axis),
                is_actionable: facet_count >= 2,
                side_sign: position[axis],
                free_axes: axes,
                local_coords: [position[axes[0]], position[axes[1]], position[axes[2]]],
            });
        }
    }
    table
        .try_into()
        .unwrap_or_else(|v: Vec<FacetGeometry>| panic!("expected {NUM_FACETS} facets, got {}", v.len()))
}

/// Fixed, state-independent bijection from GPU instance index to facet
/// geometry. Built once; the only thing that varies frame-to-frame is each
/// facet's live color, looked up from a `Hypercube` in `generate_sticker_instances`.
pub(crate) static FACET_TABLE: LazyLock<[FacetGeometry; NUM_FACETS]> = LazyLock::new(build_facet_table);

/// Builds the full GPU instance list for the current puzzle state, in the
/// stable order defined by `FACET_TABLE`.
pub(crate) fn generate_sticker_instances(hypercube: &Hypercube) -> Vec<StickerInstance> {
    FACET_TABLE
        .iter()
        .map(|facet| {
            let color = hypercube.pieces[facet.piece_slot].colors[facet.axis]
                .expect("FACET_TABLE entries are only built where colors[axis] is Some");
            StickerInstance {
                position_4d: facet.position_4d,
                color: nalgebra::Vector4::from(color).into(),
                face_id: facet.face_id as u32,
                _padding: [0; 3],
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_of_position_of_round_trip() {
        for index in 0..81 {
            let position = position_of(index);
            for axis in 0..4 {
                assert!((-1..=1).contains(&position[axis]));
            }
            assert_eq!(index_of(position), index);
        }
    }

    #[test]
    fn position_of_index_of_round_trip_for_all_positions() {
        for x in -1..=1i8 {
            for y in -1..=1i8 {
                for z in -1..=1i8 {
                    for w in -1..=1i8 {
                        let position = [x, y, z, w];
                        assert_eq!(position_of(index_of(position)), position);
                    }
                }
            }
        }
    }

    #[test]
    fn solved_has_81_pieces_in_canonical_order() {
        let cube = Hypercube::solved();
        assert_eq!(cube.pieces.len(), 81);
        for (slot, piece) in cube.pieces.iter().enumerate() {
            assert_eq!(index_of(piece.position), slot);
        }
    }

    #[test]
    fn solved_is_solved() {
        assert!(Hypercube::solved().is_solved());
    }

    #[test]
    fn scrambled_colors_are_not_solved() {
        let mut cube = Hypercube::solved();
        // Manually desync one piece's color from its home side.
        let slot = index_of([1, 1, 1, 1]);
        cube.pieces[slot].colors[0] = Some(Color::Brown);
        assert!(!cube.is_solved());
    }

    #[test]
    fn center_piece_has_no_facets() {
        let cube = Hypercube::solved();
        let center = &cube.pieces[index_of([0, 0, 0, 0])];
        assert_eq!(center.facet_count(), 0);
        assert!(center.colors.iter().all(Option::is_none));
    }

    #[test]
    fn facet_counts_match_piece_types() {
        let cube = Hypercube::solved();
        let mut counts = [0usize; 5];
        for piece in &cube.pieces {
            counts[piece.facet_count() as usize] += 1;
        }
        // 1 center, 8 cell-centers, 24 faces, 32 edges, 16 corners = 81.
        assert_eq!(counts, [1, 8, 24, 32, 16]);
    }

    #[test]
    fn facet_table_has_216_entries() {
        assert_eq!(FACET_TABLE.len(), NUM_FACETS);
        assert_eq!(NUM_FACETS, 216);
    }

    #[test]
    fn facet_table_entries_are_geometrically_consistent() {
        for facet in FACET_TABLE.iter() {
            let position = position_of(facet.piece_slot);
            assert_eq!(position[facet.axis], facet.side_sign);
            assert_ne!(facet.side_sign, 0);
            assert_eq!(facet.face_id, face_id_for(facet.axis, facet.side_sign));
            let facet_count = position.iter().filter(|c| **c != 0).count();
            assert_eq!(facet.is_actionable, facet_count >= 2);
            let axes = free_axes(facet.axis);
            assert_eq!(facet.free_axes, axes);
            assert_eq!(
                facet.local_coords,
                [position[axes[0]], position[axes[1]], position[axes[2]]]
            );
        }
    }

    #[test]
    fn generate_sticker_instances_matches_solved_colors() {
        let cube = Hypercube::solved();
        let instances = generate_sticker_instances(&cube);
        assert_eq!(instances.len(), NUM_FACETS);
        for (facet, instance) in FACET_TABLE.iter().zip(instances.iter()) {
            let expected_color = cube.pieces[facet.piece_slot].colors[facet.axis].unwrap();
            let expected: [f32; 4] = nalgebra::Vector4::from(expected_color).into();
            assert_eq!(instance.color, expected);
            assert_eq!(instance.position_4d, facet.position_4d);
            assert_eq!(instance.face_id, facet.face_id as u32);
        }
    }
}
