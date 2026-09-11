//! Helpers shared by the solver's contract tests.

use super::coords::{Coords, SLOT_COUNT, coords_of_slot, cubie_of, is_cubie_center, nonzero_count};
use crate::piece::{Hypercube, Piece, index_of, side_for_kind};

/// The position a piece belongs in, read off its stickers' kinds.
pub(crate) fn home_of(piece: &Piece) -> [i8; 4] {
    let mut home = [0i8; 4];
    for kind in piece.kinds.iter().flatten() {
        let (axis, sign) = side_for_kind(*kind).unwrap();
        home[axis] = sign;
    }
    home
}

/// Every k-sticker cubie center, in slot order.
pub(crate) fn kcubie_centers(k: u8) -> Vec<Coords> {
    (0..SLOT_COUNT)
        .map(coords_of_slot)
        .filter(|&c| is_cubie_center(c) && nonzero_count(c) == k)
        .collect()
}

/// A cubie's stickers, in axis order.
pub(crate) fn stickers_on(center: Coords) -> Vec<Coords> {
    (0..4)
        .filter(|&axis| center[axis] != 0)
        .map(|axis| {
            let mut sticker = center;
            sticker[axis] = center[axis] * 2;
            sticker
        })
        .collect()
}

/// The axis a sticker faces.
pub(crate) fn sticker_axis(sticker: Coords) -> usize {
    (0..4).find(|&axis| sticker[axis].abs() == 4).unwrap()
}

/// The (undoubled) piece position a sticker or cubie center belongs to.
pub(crate) fn position_of_coords(coords: Coords) -> [i8; 4] {
    cubie_of(coords).map(|c| c / 2)
}

/// Every `stride`th item - a deterministic sample for the default test run,
/// with the exhaustive versions left `#[ignore]`d.
pub(crate) fn sample<T>(items: Vec<T>, stride: usize) -> Vec<T> {
    items.into_iter().step_by(stride).collect()
}

/// Asserts every piece of `cube` with at most `k` stickers is exactly as in
/// the solved state, except those at `exempt` positions.
pub(crate) fn assert_untouched_except(cube: &Hypercube, k: u8, exempt: &[[i8; 4]], what: &str) {
    let solved = Hypercube::solved();
    for piece in &cube.pieces {
        let count = piece.position.iter().filter(|&&c| c != 0).count() as u8;
        if count <= k && !exempt.contains(&piece.position) {
            assert_eq!(
                *piece,
                solved.pieces[index_of(piece.position)],
                "{what}: disturbed the piece at {:?}",
                piece.position
            );
        }
    }
}
