//! Parity and solvability checks - the port of `NdSolve.java`'s
//! `puzzleStateIsOdd`, `is_positioned_up_to`, `is_oriented_up_to` and
//! `isSolvable`.

use super::coords::{
    Coords, SLOT_COUNT, coords_of_slot, cubie_of, is_cubie_center, is_sticker, nonzero_count,
    slot_of,
};
use super::grid::WantsGrid;
use super::rotation::find_rotation_sequence;
use super::{PARITY_TWIST, SolveError, SolveResult, Unsolvable, apply_twists};
use crate::piece::Hypercube;

fn is_kcubie_center(coords: Coords, k: u8) -> bool {
    is_cubie_center(coords) && nonzero_count(coords) == k
}

fn is_kcubie_sticker(coords: Coords, k: u8) -> bool {
    is_sticker(coords) && nonzero_count(coords) == k
}

/// Whether the permutation `grid` describes, restricted to the points
/// `include` accepts, is odd: an odd number of even-length cycles.
fn permutation_is_odd(grid: &WantsGrid, include: impl Fn(Coords) -> bool) -> SolveResult<bool> {
    let mut seen = [false; SLOT_COUNT];
    let mut odd = false;
    for slot in 0..SLOT_COUNT {
        let start = coords_of_slot(slot);
        if !include(start) {
            continue;
        }
        let mut len = 0;
        let mut at = start;
        while !seen[slot_of(at)] {
            seen[slot_of(at)] = true;
            len += 1;
            at = grid.target(at)?;
        }
        if len > 0 && len % 2 == 0 {
            odd = !odd;
        }
    }
    Ok(odd)
}

/// The puzzle state is odd iff it's an odd permutation on the 2-sticker
/// pieces - which a single quarter turn toggles.
pub(crate) fn puzzle_state_is_odd(grid: &WantsGrid) -> SolveResult<bool> {
    permutation_is_odd(grid, |c| is_kcubie_center(c, 2))
}

/// Whether every piece with at most `max_k` stickers is in its home slot.
pub(crate) fn is_positioned_up_to(grid: &WantsGrid, max_k: u8) -> bool {
    (0..SLOT_COUNT)
        .map(coords_of_slot)
        .all(|c| !is_cubie_center(c) || nonzero_count(c) > max_k || grid.wants(c) == Some(c))
}

/// Whether every sticker of every piece with at most `max_k` stickers is
/// home.
pub(crate) fn is_oriented_up_to(grid: &WantsGrid, max_k: u8) -> bool {
    (0..SLOT_COUNT)
        .map(coords_of_slot)
        .all(|c| !is_sticker(c) || nonzero_count(c) > max_k || grid.wants(c) == Some(c))
}

/// Checks `cube` can be solved at all, returning why not if it can't:
/// a mirrored (inside out) corner; then, after the same parity-fixing twist
/// `solve` makes if the state is odd, per piece type: an odd permutation of
/// its pieces, then odd flip parity (non-corners) or nonzero twirl parity
/// (corners).
pub(crate) fn check_solvable(cube: &Hypercube) -> SolveResult<()> {
    let mut grid = WantsGrid::from_hypercube(cube)?;

    // A corner is right-side-out iff some rotation takes its stickers to
    // where they want to be.
    for slot in 0..SLOT_COUNT {
        let center = coords_of_slot(slot);
        if !is_kcubie_center(center, 4) {
            continue;
        }
        let these: [Coords; 4] = std::array::from_fn(|axis| {
            let mut sticker = center;
            sticker[axis] = center[axis] * 2;
            sticker
        });
        let mut those = these;
        for sticker in &mut those {
            *sticker = grid.target(*sticker)?;
        }
        if find_rotation_sequence(&these, &those, true)?.is_none() {
            return Err(SolveError::Unsolvable(Unsolvable::InsideOutCorner));
        }
    }

    if puzzle_state_is_odd(&grid)? {
        let mut twisted = cube.clone();
        apply_twists(&mut twisted, &[PARITY_TWIST]);
        grid = WantsGrid::from_hypercube(&twisted)?;
    }

    for k in 2..=4u8 {
        if permutation_is_odd(&grid, |c| is_kcubie_center(c, k))? {
            return Err(SolveError::Unsolvable(Unsolvable::OddPermutation { k }));
        }
        position_virtually(&mut grid, k)?;
        if k < 4 {
            if permutation_is_odd(&grid, |c| is_kcubie_sticker(c, k))? {
                return Err(SolveError::Unsolvable(Unsolvable::FlipParity { k }));
            }
        } else if twirl_modulus(&grid)? % 3 != 0 {
            return Err(SolveError::Unsolvable(Unsolvable::TwirlParity));
        }
    }
    Ok(())
}

/// Rewrites `grid` as if every k-sticker piece had been swapped straight
/// into its home slot (carrying each sticker to the matching sticker there),
/// so the orientation checks only see what's left within each piece. Each
/// swap moves k stickers' worth of transpositions along with one piece
/// transposition, which keeps the sticker permutation's parity meaningful:
/// the piece permutation is already known to be even.
fn position_virtually(grid: &mut WantsGrid, k: u8) -> SolveResult<()> {
    for slot in 0..SLOT_COUNT {
        let center = coords_of_slot(slot);
        if !is_kcubie_center(center, k) {
            continue;
        }
        loop {
            let target = grid.target(center)?;
            if target == center {
                break;
            }
            grid.set(center, grid.wants(target));
            grid.set(target, Some(target));
            for axis in 0..4 {
                if center[axis] == 0 {
                    continue; // no sticker in this direction
                }
                let mut sticker = center;
                sticker[axis] = center[axis] * 2;
                let sticker_target = grid.target(sticker)?;
                grid.set(sticker, grid.wants(sticker_target));
                grid.set(sticker_target, Some(sticker_target));
            }
        }
    }
    Ok(())
}

/// With every corner in place, the net amount the corners are twirled: +1
/// or -1 per 3-cycle of stickers, by which way it turns (the sign of the
/// determinant of its three stickers and center). Pairs of swaps (a
/// half turn of the corner) don't count. A solvable state's total is 0 mod 3.
fn twirl_modulus(grid: &WantsGrid) -> SolveResult<i32> {
    let mut seen = [false; SLOT_COUNT];
    let mut modulus = 0;
    for slot in 0..SLOT_COUNT {
        let start = coords_of_slot(slot);
        if !is_kcubie_sticker(start, 4) {
            continue;
        }
        let mut cycle = Vec::new();
        let mut at = start;
        while !seen[slot_of(at)] {
            seen[slot_of(at)] = true;
            cycle.push(at);
            at = grid.target(at)?;
        }
        match cycle.len() {
            0..=2 => {}
            3 => {
                let rows = [cycle[0], cycle[1], cycle[2], cubie_of(start)];
                let det = intdet4(&rows.map(|row| row.map(i32::from)));
                if det == 0 {
                    return Err(SolveError::Internal("a twirl's stickers are degenerate"));
                }
                modulus += det.signum();
            }
            _ => {
                return Err(SolveError::Internal(
                    "a corner's stickers cycle in a way no rotation can",
                ));
            }
        }
    }
    Ok(modulus)
}

/// Determinant of a 4x4 integer matrix, by cofactor expansion (as in the
/// Java original's machine-generated `intdet`).
pub(crate) fn intdet4(m: &[[i32; 4]; 4]) -> i32 {
    m[0][0]
        * (m[1][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
            - m[1][2] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
            + m[1][3] * (m[2][1] * m[3][2] - m[2][2] * m[3][1]))
        - m[0][1]
            * (m[1][0] * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
                - m[1][2] * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
                + m[1][3] * (m[2][0] * m[3][2] - m[2][2] * m[3][0]))
        + m[0][2]
            * (m[1][0] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
                - m[1][1] * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
                + m[1][3] * (m[2][0] * m[3][1] - m[2][1] * m[3][0]))
        - m[0][3]
            * (m[1][0] * (m[2][1] * m[3][2] - m[2][2] * m[3][1])
                - m[1][1] * (m[2][0] * m[3][2] - m[2][2] * m[3][0])
                + m[1][2] * (m[2][0] * m[3][1] - m[2][1] * m[3][0]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::{Piece, index_of, side_kind};
    use crate::solver::coords::make_twist90;

    #[test]
    fn intdet4_known_values() {
        let identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
        assert_eq!(intdet4(&identity), 1);
        let swapped = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
        assert_eq!(intdet4(&swapped), -1);
        let diag = [[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0], [0, 0, 0, 5]];
        assert_eq!(intdet4(&diag), 120);
        let singular = [[1, 2, 3, 4], [2, 4, 6, 8], [0, 1, 0, 1], [1, 0, 1, 0]];
        assert_eq!(intdet4(&singular), 0);
        let general = [[3, 2, 0, 1], [4, 0, 1, 2], [3, 0, 2, 1], [9, 2, 3, 1]];
        assert_eq!(intdet4(&general), 24);
    }

    #[test]
    fn a_single_quarter_turn_is_odd() {
        let solved = Hypercube::solved();
        assert!(!puzzle_state_is_odd(&WantsGrid::from_hypercube(&solved).unwrap()).unwrap());
        let mut cube = solved;
        apply_twists(&mut cube, &[make_twist90(3, 1, 0, 1)]);
        assert!(puzzle_state_is_odd(&WantsGrid::from_hypercube(&cube).unwrap()).unwrap());
    }

    #[test]
    fn scrambles_are_solvable() {
        let mut rng = fastrand::Rng::with_seed(99);
        for _ in 0..100 {
            let mut cube = Hypercube::solved();
            cube.apply_random_moves(rng.u32(1..60), &mut rng);
            assert_eq!(check_solvable(&cube), Ok(()));
        }
    }

    fn unsolvable(cube: &Hypercube) -> Unsolvable {
        match check_solvable(cube) {
            Err(SolveError::Unsolvable(why)) => why,
            other => panic!("expected Unsolvable, got {other:?}"),
        }
    }

    /// Permutes the kinds on `position`'s piece: axis `axes[i]` gets the kind
    /// from `axes[(i + 1) % n]`.
    fn rotate_kinds(cube: &mut Hypercube, position: [i8; 4], axes: &[usize]) {
        let piece = &mut cube.pieces[index_of(position)];
        let old = piece.kinds;
        for i in 0..axes.len() {
            piece.kinds[axes[i]] = old[axes[(i + 1) % axes.len()]];
        }
    }

    /// Moves the solved pieces at `p` and `q` into each other's slots,
    /// keeping each sticker's kind on the same axis.
    fn swap_pieces(cube: &mut Hypercube, p: [i8; 4], q: [i8; 4]) {
        let (pi, qi) = (index_of(p), index_of(q));
        let (pk, qk) = (cube.pieces[pi].kinds, cube.pieces[qi].kinds);
        cube.pieces[pi] = Piece {
            position: p,
            kinds: qk,
        };
        cube.pieces[qi] = Piece {
            position: q,
            kinds: pk,
        };
    }

    #[test]
    fn rejects_a_twisted_corner() {
        let mut cube = Hypercube::solved();
        rotate_kinds(&mut cube, [1, 1, 1, 1], &[0, 1, 2]);
        assert_eq!(unsolvable(&cube), Unsolvable::TwirlParity);
    }

    #[test]
    fn rejects_a_mirrored_corner() {
        let mut cube = Hypercube::solved();
        rotate_kinds(&mut cube, [1, 1, 1, 1], &[0, 1]);
        assert_eq!(unsolvable(&cube), Unsolvable::InsideOutCorner);
    }

    #[test]
    fn rejects_single_flips() {
        let mut cube = Hypercube::solved();
        rotate_kinds(&mut cube, [1, 1, 0, 0], &[0, 1]);
        assert_eq!(unsolvable(&cube), Unsolvable::FlipParity { k: 2 });

        let mut cube = Hypercube::solved();
        rotate_kinds(&mut cube, [1, 1, 1, 0], &[0, 1]);
        assert_eq!(unsolvable(&cube), Unsolvable::FlipParity { k: 3 });
    }

    #[test]
    fn rejects_swapped_pieces() {
        // Swapping two 2-sticker pieces makes the state odd; the quarter
        // turn that fixes that also cycles 3-sticker pieces in 4-cycles, so
        // it's the 3-sticker permutation left odd.
        let mut cube = Hypercube::solved();
        swap_pieces(&mut cube, [1, 1, 0, 0], [-1, -1, 0, 0]);
        assert_eq!(unsolvable(&cube), Unsolvable::OddPermutation { k: 3 });

        let mut cube = Hypercube::solved();
        swap_pieces(&mut cube, [1, 1, 1, 0], [-1, -1, 1, 0]);
        assert_eq!(unsolvable(&cube), Unsolvable::OddPermutation { k: 3 });

        // Two corners differing on two axes, so neither ends up mirrored.
        let mut cube = Hypercube::solved();
        swap_pieces(&mut cube, [1, 1, 1, 1], [-1, -1, 1, 1]);
        assert_eq!(unsolvable(&cube), Unsolvable::OddPermutation { k: 4 });
    }

    #[test]
    fn rejects_a_moved_cell_center() {
        let mut cube = Hypercube::solved();
        cube.pieces[index_of([0, 0, -1, 0])].kinds[2] = Some(side_kind(3, 1));
        assert_eq!(unsolvable(&cube), Unsolvable::CellCenterMoved);
    }

    #[test]
    fn positioned_and_oriented_checks() {
        let solved = WantsGrid::from_hypercube(&Hypercube::solved()).unwrap();
        assert!(is_positioned_up_to(&solved, 4));
        assert!(is_oriented_up_to(&solved, 4));

        let mut cube = Hypercube::solved();
        rotate_kinds(&mut cube, [1, 1, 1, 0], &[0, 1]);
        let grid = WantsGrid::from_hypercube(&cube).unwrap();
        assert!(is_positioned_up_to(&grid, 4));
        assert!(is_oriented_up_to(&grid, 2));
        assert!(!is_oriented_up_to(&grid, 3));
    }
}
