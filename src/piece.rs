//! Piece-based puzzle state.
//!
//! Replaces sticker-tracking with piece-tracking: each `Piece` carries a
//! lattice position (one of the 81 points in {-1,0,1}^4) and, for each axis
//! where its position is nonzero, the color of the facet currently facing
//! that axis's sign. A piece's `Vec` slot is always determined by its own
//! current position (see `index_of`), so moves never need to reorder the
//! `Vec` and two `Hypercube`s can be compared with a plain `assert_eq!`.

use serde::{Deserialize, Serialize};

use crate::geometry::Color;

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
}
