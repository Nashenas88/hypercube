//! Bridges `solver::coords::Twist` (NdSolve's own quarter-turn-of-one-outer-side
//! move) to this project's native `Move` (which also allows the 180 degree
//! and 120 degree turns a single click can perform), and merges consecutive
//! same-side twists in the raw solution into this project's richer move
//! vocabulary.

use crate::moves::{Move, ROT3_IDENTITY, Rot3, native_move_for_rotation, rot3_mul};
use crate::piece::free_axes;
use crate::solver::coords::Twist;

/// The `Rot3` a `Twist` performs, expressed in the local (`free_axes`) basis:
/// with `axes = free_axes(face_axis)`, `i`/`j` the positions of
/// `from_axis`/`to_axis` within `axes`, and `c` the remaining position, it's
/// the identity except `M[j][i] = 1` and `M[i][j] = -1` - the same rotation
/// `coords::rot90` performs, restated as a matrix on local coordinates.
fn twist_local_rotation(twist: &Twist) -> Rot3 {
    let axes = free_axes(twist.face_axis);
    let i = axes.iter().position(|&a| a == twist.from_axis).unwrap();
    let j = axes.iter().position(|&a| a == twist.to_axis).unwrap();
    let mut m = ROT3_IDENTITY;
    for row in m.iter_mut() {
        *row = [0, 0, 0];
    }
    let c = 3 - i - j;
    m[c][c] = 1;
    m[j][i] = 1;
    m[i][j] = -1;
    m
}

/// The native `Move` equivalent to a single `Twist`: a +-90 degree turn about
/// the local axis `c` not involved in the twist's `from_axis`/`to_axis` pair.
/// A right-handed +90 degree turn about local axis `c` sends local axis
/// `(c+1)%3` to `(c+2)%3`, which is `+90` exactly when `(j + 3 - i) % 3 == 1`
/// (`i`/`j` as in `twist_local_rotation`) and `-90` otherwise.
pub(crate) fn twist_to_move(twist: &Twist) -> Move {
    let axes = free_axes(twist.face_axis);
    let i = axes.iter().position(|&a| a == twist.from_axis).unwrap();
    let j = axes.iter().position(|&a| a == twist.to_axis).unwrap();
    let c = 3 - i - j;
    let mut local_coords = [0i8; 3];
    local_coords[c] = 1;
    let angle = if (j + 3 - i) % 3 == 1 {
        std::f32::consts::FRAC_PI_2
    } else {
        -std::f32::consts::FRAC_PI_2
    };
    Move {
        side_axis: twist.face_axis,
        side_sign: twist.face_sign,
        local_coords,
        angle,
    }
}

/// Composes consecutive same-side twists in `twists` into this project's
/// native moves (90/180 degree face, 180 degree edge, +-120 degree corner),
/// dropping runs that cancel to identity. Uses a stack: a twist on the same
/// side as the top entry is composed into it (popping it if the result is
/// the identity, so the next twist can merge with whatever's now on top);
/// otherwise it's pushed as a new entry. This is sound for gaps of unrelated
/// sides too - e.g. `A B B' A'` reduces to nothing - because composing is
/// associative: cancelling `B B'` down to identity, wherever it sits in the
/// stack, doesn't change the product of everything around it.
pub(crate) fn merge_stage(twists: &[Twist]) -> Vec<Move> {
    let mut stack: Vec<(usize, i8, Rot3)> = Vec::new();
    for twist in twists {
        let rot = twist_local_rotation(twist);
        if let Some(top) = stack.last_mut()
            && top.0 == twist.face_axis
            && top.1 == twist.face_sign
        {
            let composed = rot3_mul(&rot, &top.2);
            if composed == ROT3_IDENTITY {
                stack.pop();
            } else {
                top.2 = composed;
            }
            continue;
        }
        stack.push((twist.face_axis, twist.face_sign, rot));
    }
    stack
        .into_iter()
        .filter_map(|(side_axis, side_sign, rot)| {
            native_move_for_rotation(&rot).map(|(local_coords, angle)| Move {
                side_axis,
                side_sign,
                local_coords,
                angle,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::{Hypercube, Piece, index_of};
    use crate::solver::coords::{make_twist90, sticker_coords, twist90};

    /// Independent, doubled-coordinate oracle for what one `Twist` does to a
    /// `Hypercube`: rotates every piece on the twist's side by relabeling
    /// which axis each of its kinds sits on (mirroring `coords::rot90` on
    /// the piece's own position and, per-axis, its stickers), and leaves
    /// every other piece untouched. Independent of `Hypercube::apply_move`
    /// and `discrete_rotation`, so agreement with `apply(twist_to_move(t))`
    /// is real evidence the two move representations agree, not a
    /// self-fulfilling check against the same code.
    fn oracle_apply_twist(cube: &Hypercube, twist: &Twist) -> Hypercube {
        let mut new_pieces = cube.pieces.clone();
        for old in &cube.pieces {
            if old.position[twist.face_axis] * twist.face_sign <= 0 {
                continue; // not on this side
            }
            let mut new_position = old.position;
            new_position[twist.to_axis] = old.position[twist.from_axis];
            new_position[twist.from_axis] = -old.position[twist.to_axis];

            let mut new_kinds = old.kinds;
            new_kinds[twist.to_axis] = old.kinds[twist.from_axis];
            new_kinds[twist.from_axis] = old.kinds[twist.to_axis];

            // Cross-check against the doubled-coordinate formulation too:
            // rotating each live sticker's own coords should land it exactly
            // on the sticker coords implied by the new position/axis.
            for axis in 0..4 {
                if let Some(kind) = old.kinds[axis] {
                    let rotated = twist90(twist, sticker_coords(old.position, axis));
                    let new_axis = if axis == twist.from_axis {
                        twist.to_axis
                    } else if axis == twist.to_axis {
                        twist.from_axis
                    } else {
                        axis
                    };
                    assert_eq!(
                        rotated,
                        sticker_coords(new_position, new_axis),
                        "coords oracle disagrees with position/kind relabeling for kind {kind}"
                    );
                }
            }

            new_pieces[index_of(new_position)] = Piece {
                position: new_position,
                kinds: new_kinds,
            };
        }
        Hypercube { pieces: new_pieces }
    }

    fn all_twists() -> Vec<Twist> {
        let mut twists = Vec::with_capacity(48);
        for face_axis in 0..4 {
            for &face_sign in &[1i8, -1] {
                let axes = free_axes(face_axis);
                for &from_axis in &axes {
                    for &to_axis in &axes {
                        if from_axis != to_axis {
                            twists.push(make_twist90(face_axis, face_sign, from_axis, to_axis));
                        }
                    }
                }
            }
        }
        twists
    }

    #[test]
    fn all_twists_is_48_entries() {
        assert_eq!(all_twists().len(), 48);
    }

    #[test]
    fn twist_to_move_matches_oracle_from_solved() {
        let solved = Hypercube::solved();
        for twist in all_twists() {
            let expected = oracle_apply_twist(&solved, &twist);
            let mut actual = solved.clone();
            actual.apply(&twist_to_move(&twist));
            assert_eq!(actual, expected, "mismatch for {twist:?}");
        }
    }

    #[test]
    fn twist_to_move_matches_oracle_from_scrambled_states() {
        let mut rng = fastrand::Rng::with_seed(20260911);
        for seed_iter in 0..20 {
            let mut cube = Hypercube::solved();
            cube.apply_random_moves(15, &mut rng);
            for twist in all_twists() {
                let expected = oracle_apply_twist(&cube, &twist);
                let mut actual = cube.clone();
                actual.apply(&twist_to_move(&twist));
                assert_eq!(
                    actual, expected,
                    "mismatch for {twist:?} on scramble {seed_iter}"
                );
            }
        }
    }

    #[test]
    fn merge_stage_four_identical_twists_vanish() {
        let t = make_twist90(3, 1, 0, 1);
        let merged = merge_stage(&[t, t, t, t]);
        assert!(merged.is_empty());
    }

    #[test]
    fn merge_stage_cancels_across_a_gap() {
        let a = make_twist90(3, 1, 0, 1);
        let b = make_twist90(0, 1, 1, 2);
        let merged = merge_stage(&[a, b, b.reversed(), a.reversed()]);
        assert!(merged.is_empty(), "expected empty, got {merged:?}");
    }

    #[test]
    fn merge_stage_two_quarter_turns_become_one_move() {
        let t = make_twist90(3, 1, 0, 1);
        let merged = merge_stage(&[t, t]);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn merge_stage_matches_raw_application_for_random_twist_strings() {
        let mut rng = fastrand::Rng::with_seed(7);
        let axes_choices = [0usize, 1, 2, 3];
        for _ in 0..500 {
            let len = rng.usize(0..12);
            let mut twists = Vec::with_capacity(len);
            for _ in 0..len {
                let face_axis = axes_choices[rng.usize(0..4)];
                let face_sign = if rng.bool() { 1 } else { -1 };
                let axes = free_axes(face_axis);
                let from_axis = axes[rng.usize(0..3)];
                let to_axis = loop {
                    let candidate = axes[rng.usize(0..3)];
                    if candidate != from_axis {
                        break candidate;
                    }
                };
                twists.push(make_twist90(face_axis, face_sign, from_axis, to_axis));
            }

            let mut via_raw = Hypercube::solved();
            for twist in &twists {
                via_raw.apply(&twist_to_move(twist));
            }

            let merged = merge_stage(&twists);
            let mut via_merged = Hypercube::solved();
            for mv in &merged {
                via_merged.apply(mv);
            }

            assert_eq!(via_raw, via_merged, "twists={twists:?} merged={merged:?}");
        }
    }
}
