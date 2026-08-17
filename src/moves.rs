//! Move application for the piece-based puzzle, discrete and continuous.
//!
//! A move rotates one tesseract "side" (the 27 pieces sharing a fixed value
//! on one axis) as a rigid 3x3x3 sub-cube. The rotation axis is always the
//! clicked piece's own position restricted to the 3 "free" axes
//! (`local_coords`); the number of nonzero local coords determines whether
//! it's a 90 degree face-type, 180 degree edge-type, or 120 degree
//! corner-type turn.

use std::f32::consts::{FRAC_PI_2, PI, TAU};

use nalgebra::{Matrix4, Rotation3, Unit, Vector3, Vector4};

use crate::math::{VIEWER_DISTANCE, project_4d_to_3d};
use crate::piece::{FACET_TABLE, FacetGeometry, Hypercube, Piece, free_axes, index_of};

/// Rounds a continuous 3D rotation matrix (about `local_coords`, by `angle`)
/// to an exact signed permutation: `new[row] = sign[row] * old[perm[row]]`.
/// Panics via `debug_assert` if the matrix isn't actually a signed
/// permutation at this angle/axis - a real bug indicator (e.g. an angle that
/// isn't a valid lattice symmetry for this axis), not a case to paper over.
pub(crate) fn discrete_rotation(local_coords: [i8; 3], angle: f32) -> ([usize; 3], [i8; 3]) {
    let axis = Vector3::new(
        local_coords[0] as f32,
        local_coords[1] as f32,
        local_coords[2] as f32,
    );
    let rotation = Rotation3::from_axis_angle(&Unit::new_normalize(axis), angle);
    let matrix = rotation.matrix();

    let mut perm = [usize::MAX; 3];
    let mut sign = [0i8; 3];
    for row in 0..3 {
        let mut hits = (0..3).filter(|&col| matrix[(row, col)].abs() > 0.5);
        let col = hits.next().expect("rotation row has no dominant entry");
        debug_assert!(
            hits.next().is_none(),
            "rotation row has more than one dominant entry"
        );
        debug_assert!(
            (matrix[(row, col)].abs() - 1.0).abs() < 1e-3,
            "dominant entry isn't close to +-1"
        );
        perm[row] = col;
        sign[row] = if matrix[(row, col)] > 0.0 { 1 } else { -1 };
    }
    debug_assert_eq!(
        {
            let mut sorted = perm;
            sorted.sort_unstable();
            sorted
        },
        [0, 1, 2],
        "perm isn't a bijection"
    );
    (perm, sign)
}

/// Rotates a continuous local-space position by `angle` about `local_coords`,
/// without rounding to the lattice. Used to render the in-between frames of
/// a move's animation; `discrete_rotation` is used for the final, snapped
/// state instead.
pub(crate) fn rotate_local_position(
    local_coords: [i8; 3],
    angle: f32,
    local_position: [f32; 3],
) -> [f32; 3] {
    let axis = Vector3::new(
        local_coords[0] as f32,
        local_coords[1] as f32,
        local_coords[2] as f32,
    );
    let rotation = Rotation3::from_axis_angle(&Unit::new_normalize(axis), angle);
    let rotated = rotation * Vector3::new(local_position[0], local_position[1], local_position[2]);
    [rotated.x, rotated.y, rotated.z]
}

/// Target angle magnitude for a click, given the clicked facet's number of
/// nonzero local coordinates (1 = face, 2 = edge, 3 = corner).
pub(crate) fn base_angle(local_nonzero_count: usize) -> f32 {
    match local_nonzero_count {
        1 => FRAC_PI_2,
        2 => PI,
        3 => TAU / 3.0,
        n => unreachable!("non-actionable or malformed local_coords: {n} nonzero"),
    }
}

/// Sign to apply to a click's rotation magnitude so it turns clockwise when
/// viewed from beyond the clicked facet, looking back in toward the
/// rotation's own center - i.e. along the ray from the cell's center
/// (`local_coords = (0,0,0)`) through the facet's own position, which is a
/// positive multiple of `local_coords` itself (`FacetGeometry`'s doc: "this
/// piece's position restricted to `free_axes`"). That viewing ray is exactly
/// the rotation axis `discrete_rotation`/`rotate_local_position` use, so
/// this is the one case where clockwise is never ambiguous: every off-axis
/// point traces a clean circle around the view direction, with no
/// flip/tumble regardless of `local_coords`, `axis`, or `side_sign`.
///
/// The sign still depends on the facet: `local_coords` and the rotation
/// live in an abstract 3D frame (the ordered `free_axes` basis), and what a
/// viewer actually sees is that frame's image under `math::project_4d_to_3d`,
/// a nonlinear map (from the perspective divide) whose local Jacobian can be
/// orientation-*reversing*, flipping which direction reads as clockwise,
/// depending on which axis is fixed and the sign of the piece's position on
/// it.
///
/// Concretely, with `d[0], d[1], d[2]` the images (via finite difference, at
/// the identity 4D rotation - a fixed reference frame) of the three
/// `free_axes` basis directions:
///
/// - `visible_axis` transforms `local_coords` as a *pseudovector* (the
///   cofactor-matrix construction `local_coords[0]*(d[1]×d[2]) +
///   local_coords[1]*(d[2]×d[0]) + local_coords[2]*(d[0]×d[1])`), giving the
///   true rendered spin axis.
/// - `ordinary_direction` transforms the *same* `local_coords` as an
///   ordinary vector (`local_coords[0]*d[0] + local_coords[1]*d[1] +
///   local_coords[2]*d[2]`), giving the direction toward a viewer standing
///   further out along the same ray.
///
/// These coincide when the local map is orientation-preserving and diverge
/// when it isn't: `visible_axis` pointing toward that viewer means a
/// positive angle reads counterclockwise (so clockwise needs `-1.0`);
/// pointing away means it already reads clockwise (`1.0`).
pub(crate) fn clockwise_sign(facet: &FacetGeometry) -> f32 {
    const EPSILON: f32 = 1e-3;
    let rotation_4d = Matrix4::identity();
    let position_4d = Vector4::from(facet.position_4d);
    let base = project_4d_to_3d(position_4d, &rotation_4d, VIEWER_DISTANCE);
    let tangent = |axis: usize| -> Vector3<f32> {
        let mut offset_4d = position_4d;
        offset_4d[axis] += EPSILON;
        (project_4d_to_3d(offset_4d, &rotation_4d, VIEWER_DISTANCE) - base) / EPSILON
    };
    let local_coords = facet.local_coords.map(|c| c as f32);
    let d = facet.free_axes.map(tangent);
    let visible_axis = d[1].cross(&d[2]) * local_coords[0]
        + d[2].cross(&d[0]) * local_coords[1]
        + d[0].cross(&d[1]) * local_coords[2];
    let ordinary_direction =
        d[0] * local_coords[0] + d[1] * local_coords[1] + d[2] * local_coords[2];
    if visible_axis.dot(&ordinary_direction) > 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// Picks a uniformly random actionable facet from `FACET_TABLE` and derives
/// a legal move from it: `(side_axis, side_sign, local_coords, angle)`, with
/// `angle`'s magnitude from `base_angle` and a random sign. Unlike
/// `clockwise_sign`, which picks a sign to match what a viewer would call
/// clockwise, a random move has no visual referent to match, so this stays
/// pure puzzle logic with no dependency on `math::project_4d_to_3d`.
pub(crate) fn random_move(rng: &mut fastrand::Rng) -> (usize, i8, [i8; 3], f32) {
    let actionable: Vec<&FacetGeometry> = FACET_TABLE.iter().filter(|f| f.is_actionable).collect();
    let facet = actionable[rng.usize(..actionable.len())];
    let nonzero = facet.local_coords.iter().filter(|c| **c != 0).count();
    let sign = if rng.bool() { 1.0 } else { -1.0 };
    (
        facet.axis,
        facet.side_sign,
        facet.local_coords,
        base_angle(nonzero) * sign,
    )
}

impl Hypercube {
    /// Applies a move: `side_axis`/`side_sign` select the 27-piece affected
    /// side; `local_coords` is the clicked piece's local position (the
    /// rotation axis); `angle` is the signed target angle (its sign encodes
    /// direction, its magnitude should come from `base_angle`).
    pub(crate) fn apply_move(
        &mut self,
        side_axis: usize,
        side_sign: i8,
        local_coords: [i8; 3],
        angle: f32,
    ) {
        let axes = free_axes(side_axis);
        let (perm, sign) = discrete_rotation(local_coords, angle);

        let affected = self
            .pieces
            .iter()
            .enumerate()
            .filter(|(_, p)| p.position[side_axis] == side_sign)
            .map(|(i, _)| i);
        debug_assert_eq!(affected.clone().count(), 27);

        let snapshot: Vec<Piece> = affected.map(|i| self.pieces[i]).collect();

        for old in &snapshot {
            let mut new_position = old.position;
            let mut new_colors = old.colors;
            for slot in 0..3 {
                let dst = axes[slot];
                let src = axes[perm[slot]];
                new_position[dst] = sign[slot] * old.position[src];
                new_colors[dst] = old.colors[src];
            }
            self.pieces[index_of(new_position)] = Piece {
                position: new_position,
                colors: new_colors,
            };
        }
    }

    /// Applies `count` random legal moves in sequence, instantly (no
    /// animation) - used by the 1/2/3-random-move and Scramble UI actions.
    pub(crate) fn apply_random_moves(&mut self, count: u32, rng: &mut fastrand::Rng) {
        for _ in 0..count {
            let (side_axis, side_sign, local_coords, angle) = random_move(rng);
            self.apply_move(side_axis, side_sign, local_coords, angle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// side_axis=W(3), side_sign=+1, free_axes=[X(0),Y(1),Z(2)] throughout.
    const SIDE_AXIS: usize = 3;
    const SIDE_SIGN: i8 = 1;

    #[test]
    fn discrete_rotation_matches_worked_180_edge_example() {
        // local_coords=(1,1,0): X<->Y swap, Z negated.
        let (perm, sign) = discrete_rotation([1, 1, 0], PI);
        assert_eq!(perm, [1, 0, 2]);
        assert_eq!(sign, [1, 1, -1]);
    }

    #[test]
    fn discrete_rotation_matches_worked_120_corner_example() {
        // local_coords=(1,1,1): cyclic X<-Z<-Y<-X.
        let (perm, sign) = discrete_rotation([1, 1, 1], TAU / 3.0);
        assert_eq!(perm, [2, 0, 1]);
        assert_eq!(sign, [1, 1, 1]);
    }

    #[test]
    fn discrete_rotation_90_face_is_a_valid_signed_permutation() {
        for &local_coords in &[[1i8, 0, 0], [0, 1, 0], [0, 0, 1]] {
            let (perm, sign) = discrete_rotation(local_coords, FRAC_PI_2);
            let mut sorted = perm;
            sorted.sort_unstable();
            assert_eq!(sorted, [0, 1, 2]);
            assert!(sign.iter().all(|&s| s == 1 || s == -1));
        }
    }

    #[test]
    fn rotate_local_position_at_zero_angle_is_identity() {
        for &local_coords in &[[1i8, 0, 0], [1, 1, 0], [1, 1, 1]] {
            let position = [1.0, -1.0, 0.5];
            let rotated = rotate_local_position(local_coords, 0.0, position);
            for i in 0..3 {
                assert!((rotated[i] - position[i]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn rotate_local_position_at_snapped_angle_matches_discrete_rotation() {
        for (local_coords, angle) in [
            ([1i8, 0, 0], FRAC_PI_2),
            ([1, 1, 0], PI),
            ([1, 1, 1], TAU / 3.0),
        ] {
            let (perm, sign) = discrete_rotation(local_coords, angle);
            let position = [1.0, -1.0, 0.5];
            let expected = [
                sign[0] as f32 * position[perm[0]],
                sign[1] as f32 * position[perm[1]],
                sign[2] as f32 * position[perm[2]],
            ];
            let rotated = rotate_local_position(local_coords, angle, position);
            for i in 0..3 {
                assert!(
                    (rotated[i] - expected[i]).abs() < 1e-5,
                    "component {i}: rotated={:?} expected={:?}",
                    rotated,
                    expected
                );
            }
        }
    }

    fn apply_click(cube: &mut Hypercube, local_coords: [i8; 3], direction: i8) {
        let nonzero = local_coords.iter().filter(|c| **c != 0).count();
        let magnitude = base_angle(nonzero);
        cube.apply_move(
            SIDE_AXIS,
            SIDE_SIGN,
            local_coords,
            magnitude * direction as f32,
        );
    }

    fn colors_position_invariant_holds(cube: &Hypercube) -> bool {
        cube.pieces
            .iter()
            .all(|p| (0..4).all(|axis| p.colors[axis].is_some() == (p.position[axis] != 0)))
    }

    fn total_facet_count(cube: &Hypercube) -> usize {
        cube.pieces.iter().map(|p| p.facet_count() as usize).sum()
    }

    #[test]
    fn invariant_holds_after_each_move_type() {
        for local_coords in [[1i8, 0, 0], [1, 1, 0], [1, 1, 1]] {
            let mut cube = Hypercube::solved();
            apply_click(&mut cube, local_coords, 1);
            assert!(colors_position_invariant_holds(&cube));
        }
    }

    #[test]
    fn face_move_has_period_4() {
        let solved = Hypercube::solved();
        let mut cube = solved.clone();
        for _ in 0..4 {
            apply_click(&mut cube, [1, 0, 0], 1);
        }
        assert_eq!(cube, solved);
    }

    #[test]
    fn edge_move_has_period_2() {
        let solved = Hypercube::solved();
        let mut cube = solved.clone();
        for _ in 0..2 {
            apply_click(&mut cube, [1, 1, 0], 1);
        }
        assert_eq!(cube, solved);
    }

    #[test]
    fn corner_move_has_period_3() {
        let solved = Hypercube::solved();
        let mut cube = solved.clone();
        for _ in 0..3 {
            apply_click(&mut cube, [1, 1, 1], 1);
        }
        assert_eq!(cube, solved);
    }

    #[test]
    fn single_move_is_never_a_no_op() {
        for local_coords in [[1i8, 0, 0], [1, 1, 0], [1, 1, 1]] {
            let solved = Hypercube::solved();
            let mut cube = solved.clone();
            apply_click(&mut cube, local_coords, 1);
            assert_ne!(cube, solved);
        }
    }

    #[test]
    fn affected_set_is_27_pieces() {
        for local_coords in [[1i8, 0, 0], [1, 1, 0], [1, 1, 1]] {
            let before = Hypercube::solved();
            let mut after = before.clone();
            apply_click(&mut after, local_coords, 1);

            let affected_before = (0..81)
                .filter(|&i| before.pieces[i].position[SIDE_AXIS] == SIDE_SIGN)
                .count();
            let affected_after = (0..81)
                .filter(|&i| after.pieces[i].position[SIDE_AXIS] == SIDE_SIGN)
                .count();
            assert_eq!(affected_before, 27);
            assert_eq!(affected_after, 27);
        }
    }

    #[test]
    fn facet_count_conserved_across_scramble() {
        let mut cube = Hypercube::solved();
        let expected = total_facet_count(&cube);
        let moves = [[1i8, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]];
        for (i, local_coords) in moves.iter().cycle().take(20).enumerate() {
            apply_click(&mut cube, *local_coords, if i % 2 == 0 { 1 } else { -1 });
        }
        assert_eq!(total_facet_count(&cube), expected);
    }

    #[test]
    fn cell_centers_never_move_across_scramble() {
        let mut cube = Hypercube::solved();
        let cell_center_positions: Vec<[i8; 4]> = cube
            .pieces
            .iter()
            .filter(|p| p.facet_count() == 1)
            .map(|p| p.position)
            .collect();
        assert_eq!(cell_center_positions.len(), 8);

        let moves = [[1i8, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 0, 1]];
        for (i, local_coords) in moves.iter().cycle().take(20).enumerate() {
            apply_click(&mut cube, *local_coords, if i % 2 == 0 { 1 } else { -1 });
        }

        for position in cell_center_positions {
            let piece = &cube.pieces[index_of(position)];
            assert_eq!(piece.position, position);
            assert_eq!(piece.facet_count(), 1);
        }
    }

    #[test]
    fn is_solved_false_after_single_move_true_after_inverse() {
        let mut cube = Hypercube::solved();
        apply_click(&mut cube, [1, 1, 0], 1);
        assert!(!cube.is_solved());
        apply_click(&mut cube, [1, 1, 0], -1);
        assert!(cube.is_solved());
    }

    #[test]
    fn scramble_then_inverses_returns_to_solved() {
        let solved = Hypercube::solved();
        let mut cube = solved.clone();
        let moves: [([i8; 3], i8); 6] = [
            ([1, 0, 0], 1),
            ([1, 1, 0], 1),
            ([1, 1, 1], 1),
            ([0, 1, 0], -1),
            ([1, 0, 1], 1),
            ([1, 1, 1], -1),
        ];
        for &(local_coords, direction) in &moves {
            apply_click(&mut cube, local_coords, direction);
        }
        assert_ne!(cube, solved);
        for &(local_coords, direction) in moves.iter().rev() {
            let inverse_direction = if local_coords.iter().filter(|c| **c != 0).count() == 2 {
                direction // edge moves are their own inverse
            } else {
                -direction
            };
            apply_click(&mut cube, local_coords, inverse_direction);
        }
        assert_eq!(cube, solved);
    }

    #[test]
    fn random_move_always_targets_an_actionable_facet() {
        let mut rng = fastrand::Rng::with_seed(1);
        for _ in 0..200 {
            let (_, _, local_coords, _) = random_move(&mut rng);
            let nonzero = local_coords.iter().filter(|c| **c != 0).count();
            assert!((1..=3).contains(&nonzero));
        }
    }

    #[test]
    fn apply_random_moves_with_seed_is_reproducible() {
        let mut rng_a = fastrand::Rng::with_seed(42);
        let mut rng_b = fastrand::Rng::with_seed(42);
        let mut cube_a = Hypercube::solved();
        let mut cube_b = Hypercube::solved();
        cube_a.apply_random_moves(10, &mut rng_a);
        cube_b.apply_random_moves(10, &mut rng_b);
        assert_eq!(cube_a, cube_b);
    }

    #[test]
    fn apply_random_moves_preserves_facet_count_invariant() {
        let mut cube = Hypercube::solved();
        let expected = total_facet_count(&cube);
        let mut rng = fastrand::Rng::with_seed(7);
        cube.apply_random_moves(30, &mut rng);
        assert_eq!(total_facet_count(&cube), expected);
        assert!(colors_position_invariant_holds(&cube));
    }

    #[test]
    fn apply_random_moves_zero_is_a_no_op() {
        let solved = Hypercube::solved();
        let mut cube = solved.clone();
        let mut rng = fastrand::Rng::with_seed(3);
        cube.apply_random_moves(0, &mut rng);
        assert_eq!(cube, solved);
    }
}
