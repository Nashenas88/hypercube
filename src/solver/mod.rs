//! A solver for the puzzle, ported from Don Hatch's `NdSolve.java` (Magic
//! Cube 4D, by Melinda Green & Don Hatch - http://superliminal.com/cube/cube.htm).
//!
//! NdSolve groups pieces by sticker count - 2 (24 pieces), 3 (32) and 4
//! (16); the 1-sticker cell centers never move, so each sticker's kind names
//! its home side - and, smallest group first, positions and then orients
//! each group without disturbing any group already done (groups with more
//! stickers may get scrambled along the way).
//!
//! - **Positioning** (`position`) follows where each piece wants to go into
//!   cycles, splits those into 3-cycles, and does each as setup turns into
//!   an L-shape, a fixed recipe, then undoing the setup. The recipe is four
//!   half turns for 2-sticker pieces, the classic 8-move 3D corner 3-cycle
//!   (applied to whole rows) for 3-sticker ones, and one more commutator
//!   layer on top of that for corners. An odd state first gets one
//!   arbitrary quarter turn (`PARITY_TWIST`), since 3-cycles are even.
//! - **Orienting** (`orient`) regroups what's left into pairs - "flip two
//!   pieces" for 2- and 3-sticker pieces, "twirl two corners in opposite
//!   directions" for corners - each done as a flattening setup, a fixed
//!   recipe, then undoing the setup. If all that's left is on one piece, a
//!   solved "happy helper" piece is borrowed.
//!
//! Every setup move is found by a small rotation search (`rotation`) rather
//! than hardcoded. NdSolve only outputs quarter turns of an outer side
//! (`coords::Twist`), each of which is a single face-piece click here; this
//! project's 180 degree and 120 degree moves are compositions of those, so
//! `native::merge_stage` folds consecutive same-side turns back into single
//! native moves before playback.
//!
//! Only the 3^4 paths of the original are ported: branches that only other
//! puzzle sizes reach (2^d, 5+ dimensions, multi-slice twists) return
//! `SolveError::Internal` instead. `solve` checks its own answer before
//! returning it, so a bug surfaces as an error rather than a wrong solution.

mod coords;
mod grid;
mod native;
mod orient;
mod position;
mod rotation;
mod solvable;
#[cfg(test)]
mod test_support;

use std::fmt;

use crate::moves::Move;
use crate::piece::Hypercube;
use coords::Twist;
use grid::WantsGrid;

/// Which part of the solve a move belongs to, for progress display.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    Parity,
    Position(u8),
    Orient(u8),
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stage::Parity => write!(f, "fixing parity"),
            Stage::Position(k) => write!(f, "positioning {k}-sticker pieces"),
            Stage::Orient(k) => write!(f, "orienting {k}-sticker pieces"),
        }
    }
}

/// One native move of a solution, labelled with the stage it came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SolveStep {
    pub(crate) stage: Stage,
    pub(crate) mv: Move,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Solution {
    /// The moves to play, in order. Empty if the puzzle was already solved.
    pub(crate) steps: Vec<SolveStep>,
    /// How many quarter turns the moves were merged down from.
    pub(crate) raw_twists: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SolveError {
    /// The input isn't a structurally valid puzzle state at all.
    Malformed(&'static str),
    /// A valid-looking state no sequence of moves can solve.
    Unsolvable(Unsolvable),
    /// A bug: a solver assumption failed, or its answer didn't check out.
    Internal(&'static str),
}

/// Why a structurally valid state can't be solved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unsolvable {
    CellCenterMoved,
    InsideOutCorner,
    OddPermutation { k: u8 },
    FlipParity { k: u8 },
    TwirlParity,
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolveError::Malformed(what) => write!(f, "the puzzle state is malformed ({what})"),
            SolveError::Unsolvable(why) => write!(f, "{why}"),
            SolveError::Internal(what) => write!(f, "internal solver error ({what})"),
        }
    }
}

impl fmt::Display for Unsolvable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Unsolvable::CellCenterMoved => write!(f, "a cell center is on the wrong side"),
            Unsolvable::InsideOutCorner => write!(f, "a corner piece is mirrored"),
            Unsolvable::OddPermutation { k } => {
                write!(f, "two {k}-sticker pieces are swapped")
            }
            Unsolvable::FlipParity { k } => write!(f, "a {k}-sticker piece is flipped"),
            Unsolvable::TwirlParity => write!(f, "a corner piece is twisted"),
        }
    }
}

pub(crate) type SolveResult<T> = Result<T, SolveError>;

/// The arbitrary quarter turn an odd state starts with, `{0, +1, 1, 2}`.
const PARITY_TWIST: Twist = Twist {
    face_axis: 0,
    face_sign: 1,
    from_axis: 1,
    to_axis: 2,
};

/// Works out a sequence of native moves taking `cube` to the solved state.
pub(crate) fn solve(cube: &Hypercube) -> Result<Solution, SolveError> {
    WantsGrid::from_hypercube(cube)?;
    if *cube == Hypercube::solved() {
        return Ok(Solution {
            steps: Vec::new(),
            raw_twists: 0,
        });
    }
    solvable::check_solvable(cube)?;

    let stages = solve_raw(cube)?;
    let raw_twists = stages.iter().map(|(_, twists)| twists.len()).sum();
    // Merged per stage, so every move's stage label stays exact.
    let steps: Vec<SolveStep> = stages
        .iter()
        .flat_map(|(stage, twists)| {
            native::merge_stage(twists)
                .into_iter()
                .map(|mv| SolveStep { stage: *stage, mv })
        })
        .collect();

    let mut check = cube.clone();
    for step in &steps {
        check.apply(&step.mv);
    }
    if check != Hypercube::solved() {
        return Err(SolveError::Internal("the solution didn't solve the puzzle"));
    }
    Ok(Solution { steps, raw_twists })
}

pub(crate) fn apply_twists(cube: &mut Hypercube, twists: &[Twist]) {
    for twist in twists {
        cube.apply(&native::twist_to_move(twist));
    }
}

/// The raw quarter turns for each stage, in order. After each stage its
/// twists are applied to a working copy through `Hypercube::apply` and the
/// grid rebuilt from that, rather than porting the Java original's own
/// grid-twisting, so `moves.rs` stays the only definition of what a move
/// does.
fn solve_raw(cube: &Hypercube) -> SolveResult<Vec<(Stage, Vec<Twist>)>> {
    let mut work = cube.clone();
    let mut grid = WantsGrid::from_hypercube(&work)?;
    let mut stages = Vec::new();
    let mut run = |stage: Stage, twists: Vec<Twist>, grid: &mut WantsGrid| -> SolveResult<()> {
        apply_twists(&mut work, &twists);
        *grid = WantsGrid::from_hypercube(&work)?;
        stages.push((stage, twists));
        Ok(())
    };

    if solvable::puzzle_state_is_odd(&grid)? {
        run(Stage::Parity, vec![PARITY_TWIST], &mut grid)?;
    }
    for k in 2..=4u8 {
        let twists = position::position_ksticker_cubies(k, &grid)?;
        run(Stage::Position(k), twists, &mut grid)?;
        if !solvable::is_positioned_up_to(&grid, k) || !solvable::is_oriented_up_to(&grid, k - 1) {
            return Err(SolveError::Internal(
                "positioning left a piece out of place",
            ));
        }
        let twists = orient::orient_ksticker_cubies(k, &grid)?;
        run(Stage::Orient(k), twists, &mut grid)?;
        if !solvable::is_positioned_up_to(&grid, k) || !solvable::is_oriented_up_to(&grid, k) {
            return Err(SolveError::Internal("orienting left a piece misoriented"));
        }
    }
    Ok(stages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::{index_of, side_kind};
    use std::f32::consts::{FRAC_PI_2, PI};

    fn stage_rank(stage: Stage) -> u8 {
        match stage {
            Stage::Parity => 0,
            Stage::Position(k) => 2 * k - 3,
            Stage::Orient(k) => 2 * k - 2,
        }
    }

    /// Solves, checks the solution's shape, and returns it.
    fn solve_and_check(cube: &Hypercube) -> Solution {
        let solution = solve(cube).unwrap();
        let mut replay = cube.clone();
        for step in &solution.steps {
            replay.apply(&step.mv);
        }
        assert_eq!(replay, Hypercube::solved());
        assert!(
            solution
                .steps
                .windows(2)
                .all(|w| stage_rank(w[0].stage) <= stage_rank(w[1].stage)),
            "stages out of order"
        );
        assert!(solution.steps.len() <= solution.raw_twists);
        assert!(
            solution.raw_twists < 5000,
            "{} raw twists",
            solution.raw_twists
        );
        solution
    }

    fn one_move(side_axis: usize, local_coords: [i8; 3], angle: f32) -> Hypercube {
        let mut cube = Hypercube::solved();
        cube.apply(&Move {
            side_axis,
            side_sign: 1,
            local_coords,
            angle,
        });
        cube
    }

    #[test]
    fn solved_cube_needs_no_moves() {
        let solution = solve(&Hypercube::solved()).unwrap();
        assert!(solution.steps.is_empty());
        assert_eq!(solution.raw_twists, 0);
    }

    #[test]
    fn odd_single_moves_start_with_the_parity_stage() {
        // A 90 degree face turn and a 180 degree edge turn are both odd
        // permutations of the 2-sticker pieces.
        for cube in [
            one_move(3, [1, 0, 0], FRAC_PI_2),
            one_move(1, [1, 1, 0], PI),
        ] {
            let solution = solve_and_check(&cube);
            assert_eq!(solution.steps[0].stage, Stage::Parity);
        }
    }

    #[test]
    fn corner_turn_needs_no_parity_stage() {
        let solution = solve_and_check(&one_move(2, [1, 1, 1], 2.0 * PI / 3.0));
        assert!(solution.steps.iter().all(|s| s.stage != Stage::Parity));
    }

    #[test]
    fn random_scrambles_solve() {
        let mut rng = fastrand::Rng::with_seed(2026);
        let (mut raw_total, mut merged_total, mut raw_max) = (0, 0, 0);
        let seeds = 200;
        for _ in 0..seeds {
            let mut cube = Hypercube::solved();
            cube.apply_random_moves(25, &mut rng);
            let solution = solve_and_check(&cube);
            raw_total += solution.raw_twists;
            merged_total += solution.steps.len();
            raw_max = raw_max.max(solution.raw_twists);
        }
        println!(
            "{seeds} scrambles: mean {} raw twists -> {} merged moves, max {raw_max} raw",
            raw_total / seeds,
            merged_total / seeds
        );
    }

    #[test]
    fn solving_is_deterministic() {
        let mut cube = Hypercube::solved();
        cube.apply_random_moves(40, &mut fastrand::Rng::with_seed(8));
        assert_eq!(solve(&cube), solve(&cube));
    }

    #[test]
    fn unsolvable_and_malformed_states_are_errors_not_panics() {
        let mut twisted = Hypercube::solved();
        let corner = &mut twisted.pieces[index_of([1, 1, 1, 1])];
        corner.kinds = [
            corner.kinds[1],
            corner.kinds[2],
            corner.kinds[0],
            corner.kinds[3],
        ];
        assert_eq!(
            solve(&twisted),
            Err(SolveError::Unsolvable(Unsolvable::TwirlParity))
        );

        let mut malformed = Hypercube::solved();
        malformed.pieces[index_of([1, 1, 0, 0])].kinds[1] = Some(side_kind(0, -1));
        assert!(matches!(solve(&malformed), Err(SolveError::Malformed(_))));
    }

    #[test]
    #[ignore = "soak: 5,000 long scrambles; run with --ignored --nocapture"]
    fn soak_long_scrambles() {
        let mut rng = fastrand::Rng::with_seed(4);
        let (mut raw_total, mut merged_total, mut raw_max) = (0, 0, 0);
        let seeds = 5000;
        let start = std::time::Instant::now();
        for _ in 0..seeds {
            let mut cube = Hypercube::solved();
            cube.apply_random_moves(200, &mut rng);
            let solution = solve_and_check(&cube);
            raw_total += solution.raw_twists;
            merged_total += solution.steps.len();
            raw_max = raw_max.max(solution.raw_twists);
        }
        println!(
            "{seeds} long scrambles: mean {} raw twists -> {} merged moves, max {raw_max} raw, {:?} per solve",
            raw_total / seeds,
            merged_total / seeds,
            start.elapsed() / seeds as u32
        );
    }
}
