//! `WantsGrid`: for every cubie center and sticker (as `coords` grid points),
//! where whatever currently sits there wants to go. The port of
//! `NdSolve.java`'s `figureOutWhereIndicesWantToBe`, built from a
//! `Hypercube` rather than a string of sticker letters - which also makes it
//! the solver's structural validation of its input.

use super::coords::{
    Coords, SLOT_COUNT, cubie_center_coords, nonzero_count, slot_of, sticker_coords,
};
use super::{SolveError, SolveResult, Unsolvable};
use crate::piece::{Hypercube, index_of, position_of, side_for_kind};

#[derive(Clone, Debug)]
pub(crate) struct WantsGrid {
    /// Indexed by `slot_of`; `None` for the "air" slots (two or more
    /// coordinates at `+-4`), which aren't part of the puzzle.
    slots: [Option<Coords>; SLOT_COUNT],
}

impl WantsGrid {
    /// Builds the grid, rejecting any `cube` that isn't structurally a
    /// puzzle state: exactly 81 pieces in canonical order, stickers on
    /// exactly the axes a piece's position is nonzero on, every kind one of
    /// the 8 sides with no two on one piece naming the same axis, and piece
    /// homes forming a bijection. A cell center carrying another side's kind
    /// is structurally fine but can't ever be solved (the cell centers never
    /// move), so that's reported as `Unsolvable` instead.
    ///
    /// A piece's home is the position whose nonzero axes/signs are its
    /// stickers' home sides; a sticker's target is that home's cubie center
    /// pushed out to `+-4` on its own home side's axis.
    pub(crate) fn from_hypercube(cube: &Hypercube) -> SolveResult<Self> {
        if cube.pieces.len() != 81 {
            return Err(SolveError::Malformed("expected exactly 81 pieces"));
        }
        let mut slots = [None; SLOT_COUNT];
        let mut home_taken = [false; 81];
        for (index, piece) in cube.pieces.iter().enumerate() {
            let position = piece.position;
            if position != position_of(index) {
                return Err(SolveError::Malformed("pieces aren't in canonical order"));
            }
            let mut home = [0i8; 4];
            let mut sticker_homes = [None; 4];
            for axis in 0..4 {
                match (position[axis], piece.kinds[axis]) {
                    (0, None) => {}
                    (0, Some(_)) => {
                        return Err(SolveError::Malformed(
                            "a piece has a sticker on an axis it has no facet on",
                        ));
                    }
                    (_, None) => return Err(SolveError::Malformed("a piece is missing a sticker")),
                    (_, Some(kind)) => {
                        let (home_axis, home_sign) = side_for_kind(kind)
                            .ok_or(SolveError::Malformed("unknown sticker kind"))?;
                        if home[home_axis] != 0 {
                            return Err(SolveError::Malformed(
                                "two of a piece's stickers belong to the same axis",
                            ));
                        }
                        home[home_axis] = home_sign;
                        sticker_homes[axis] = Some((home_axis, home_sign));
                    }
                }
            }
            if nonzero_count(position) == 1 && home != position {
                return Err(SolveError::Unsolvable(Unsolvable::CellCenterMoved));
            }
            let home_index = index_of(home);
            if home_taken[home_index] {
                return Err(SolveError::Malformed("two pieces belong in the same place"));
            }
            home_taken[home_index] = true;

            let home_center = cubie_center_coords(home);
            slots[slot_of(cubie_center_coords(position))] = Some(home_center);
            for (axis, sticker_home) in sticker_homes.iter().enumerate() {
                if let Some((home_axis, home_sign)) = *sticker_home {
                    let mut target = home_center;
                    target[home_axis] = 4 * home_sign;
                    slots[slot_of(sticker_coords(position, axis))] = Some(target);
                }
            }
        }
        Ok(Self { slots })
    }

    /// Where whatever is at `at` wants to go (`None` for air).
    pub(crate) fn wants(&self, at: Coords) -> Option<Coords> {
        self.slots[slot_of(at)]
    }

    /// `wants`, for a point the caller knows is part of the puzzle.
    pub(crate) fn target(&self, at: Coords) -> SolveResult<Coords> {
        self.wants(at).ok_or(SolveError::Internal(
            "looked up a grid point outside the puzzle",
        ))
    }

    /// Overwrites one slot - only `solvable`'s virtual repositioning needs
    /// this; real moves go through `Hypercube::apply` and a rebuild instead.
    pub(crate) fn set(&mut self, at: Coords, target: Option<Coords>) {
        self.slots[slot_of(at)] = target;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::side_kind;
    use crate::solver::coords::coords_of_slot;

    #[test]
    fn solved_grid_maps_every_point_to_itself() {
        let grid = WantsGrid::from_hypercube(&Hypercube::solved()).unwrap();
        let mut live = 0;
        for slot in 0..SLOT_COUNT {
            let coords = coords_of_slot(slot);
            if let Some(target) = grid.wants(coords) {
                assert_eq!(target, coords);
                live += 1;
            }
        }
        // 81 cubie centers + 216 stickers.
        assert_eq!(live, 297);
    }

    #[test]
    fn scrambled_grid_is_a_bijection() {
        let mut cube = Hypercube::solved();
        cube.apply_random_moves(30, &mut fastrand::Rng::with_seed(3));
        let grid = WantsGrid::from_hypercube(&cube).unwrap();
        let mut hit = [false; SLOT_COUNT];
        for slot in 0..SLOT_COUNT {
            if let Some(target) = grid.wants(coords_of_slot(slot)) {
                assert!(!hit[slot_of(target)], "two points want {target:?}");
                hit[slot_of(target)] = true;
            }
        }
    }

    fn malformed(cube: &Hypercube) -> SolveError {
        WantsGrid::from_hypercube(cube).unwrap_err()
    }

    #[test]
    fn rejects_wrong_piece_count() {
        let mut cube = Hypercube::solved();
        cube.pieces.pop();
        assert!(matches!(malformed(&cube), SolveError::Malformed(_)));
    }

    #[test]
    fn rejects_out_of_order_pieces() {
        let mut cube = Hypercube::solved();
        cube.pieces.swap(0, 1);
        assert!(matches!(malformed(&cube), SolveError::Malformed(_)));
    }

    #[test]
    fn rejects_sticker_on_zero_axis_and_missing_sticker() {
        let mut cube = Hypercube::solved();
        let i = index_of([1, 1, 0, 0]);
        cube.pieces[i].kinds[2] = Some(side_kind(2, 1));
        assert!(matches!(malformed(&cube), SolveError::Malformed(_)));

        let mut cube = Hypercube::solved();
        cube.pieces[i].kinds[0] = None;
        assert!(matches!(malformed(&cube), SolveError::Malformed(_)));
    }

    #[test]
    fn rejects_unknown_kind_and_duplicate_axes() {
        let mut cube = Hypercube::solved();
        let i = index_of([1, 1, 0, 0]);
        cube.pieces[i].kinds[0] = Some(99);
        assert!(matches!(malformed(&cube), SolveError::Malformed(_)));

        let mut cube = Hypercube::solved();
        cube.pieces[i].kinds[1] = Some(side_kind(0, -1));
        assert!(matches!(malformed(&cube), SolveError::Malformed(_)));
    }

    #[test]
    fn rejects_two_pieces_with_the_same_home() {
        let mut cube = Hypercube::solved();
        let from = index_of([1, 1, 0, 0]);
        let to = index_of([1, -1, 0, 0]);
        cube.pieces[to].kinds = cube.pieces[from].kinds;
        assert!(matches!(malformed(&cube), SolveError::Malformed(_)));
    }

    #[test]
    fn rejects_moved_cell_center_as_unsolvable() {
        let mut cube = Hypercube::solved();
        cube.pieces[index_of([1, 0, 0, 0])].kinds[0] = Some(side_kind(1, 1));
        assert_eq!(
            malformed(&cube),
            SolveError::Unsolvable(Unsolvable::CellCenterMoved)
        );
    }
}
