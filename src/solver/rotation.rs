//! Whole-puzzle rotation search - the port of `NdSolve.java`'s
//! `find_rotation_sequence_taking_these_coords_to_those_coords` and
//! `find_oneface_twist_sequence_taking_these_coords_to_those_coords`. Every
//! setup move the solver makes ("twist this face until that piece lands
//! there") is found by this search rather than hardcoded.

use super::coords::{Coords, Rot, Twist, make_twist90, norm_sqrd, rot90, rot90s};
use super::{SolveError, SolveResult};

/// Whether column `i` of `a` equals `sign` times column `j` of `b`, row by
/// row.
fn column_equals(a: &[Coords], i: usize, b: &[Coords], j: usize, sign: i8) -> bool {
    a.iter().zip(b).all(|(ra, rb)| ra[i] == sign * rb[j])
}

fn rotate_all(coords: &mut [Coords], rot: Rot) {
    for c in coords.iter_mut() {
        *c = rot90(rot.0, rot.1, *c);
    }
}

/// A sequence of 90 degree whole-puzzle rotations taking each of `these` to
/// the matching entry of `those`. Treating the points as rows, the search
/// works column (axis) by column: a column is "happy" once it matches
/// `those`' and "antihappy" once it matches its negation. The first pass
/// makes every column one or the other with 90 degree rotations, the second
/// fixes antihappy columns in pairs with 180 degree ones, and the third
/// fixes a last lone antihappy column with one of three small tricks.
///
/// If none of those tricks works the target is a mirror image of the source:
/// that's `Ok(None)` when `none_if_mirror` is set (the caller is asking
/// "same handedness?"), otherwise an `Internal` error, as is any target that
/// isn't a rotation at all.
pub(crate) fn find_rotation_sequence(
    these: &[Coords],
    those: &[Coords],
    none_if_mirror: bool,
) -> SolveResult<Option<Vec<Rot>>> {
    debug_assert_eq!(these.len(), those.len());
    debug_assert!(
        these
            .iter()
            .zip(those)
            .all(|(a, b)| norm_sqrd(*a) == norm_sqrd(*b))
    );
    let original = these;
    let mut these = these.to_vec();

    let mut happy = [false; 4];
    let mut antihappy = [false; 4];
    for i in 0..4 {
        happy[i] = column_equals(&these, i, those, i, 1);
        antihappy[i] = column_equals(&these, i, those, i, -1);
    }
    let mut rots = Vec::new();

    // First pass: every column happy or antihappy, via 90 degree rotations
    // (a column swap negating one of the two). First preference is a swap
    // that makes the other column happy too.
    for i in 0..4 {
        if happy[i] || antihappy[i] {
            continue;
        }
        for picky in [true, false] {
            for j in i + 1..4 {
                if happy[j] || antihappy[j] {
                    continue;
                }
                let rot = if column_equals(&these, j, those, i, 1)
                    && (!picky || column_equals(&these, i, those, j, -1))
                {
                    Some((j, i))
                } else if column_equals(&these, j, those, i, -1)
                    && (!picky || column_equals(&these, i, those, j, 1))
                {
                    Some((i, j))
                } else {
                    None
                };
                if let Some(rot) = rot {
                    rots.push(rot);
                    rotate_all(&mut these, rot);
                    happy[i] = true;
                    antihappy[i] = column_equals(&these, i, those, i, -1);
                    if picky {
                        happy[j] = true;
                    }
                    antihappy[j] = column_equals(&these, j, those, j, -1);
                    break;
                }
            }
            if happy[i] {
                break;
            }
        }
        if !happy[i] {
            return Err(SolveError::Internal("no rotation takes these points there"));
        }
    }

    // Second pass: fix antihappy-but-not-happy columns in pairs with 180
    // degree rotations.
    let mut remaining_unhappy_axis = None;
    for i in 0..4 {
        if happy[i] {
            continue;
        }
        if let Some(j) = (i + 1..4).find(|&j| !happy[j]) {
            for _ in 0..2 {
                rots.push((i, j));
                rotate_all(&mut these, (i, j));
            }
            happy[i] = true;
            antihappy[i] = false;
            happy[j] = true;
            antihappy[j] = false;
        }
        if !happy[i] {
            remaining_unhappy_axis = Some(i);
        }
    }

    // Third pass: at most one antihappy column is left.
    if let Some(i) = remaining_unhappy_axis {
        // Trick #1 (1 move): another column equal or opposite to it; a 90
        // degree rotation between the two fixes it and keeps the other one.
        for j in (0..4).filter(|&j| j != i) {
            let rot = if column_equals(&these, i, &these, j, 1) {
                Some((i, j))
            } else if column_equals(&these, i, &these, j, -1) {
                Some((j, i))
            } else {
                None
            };
            if let Some(rot) = rot {
                rots.push(rot);
                rotate_all(&mut these, rot);
                happy[i] = true;
                antihappy[i] = false;
                break;
            }
        }

        // Trick #2 (2 moves): an all-zero column; a 180 degree rotation
        // with it fixes this one and leaves the zero column zero.
        if !happy[i]
            && let Some(j) = (0..4).find(|&j| j != i && column_equals(&these, j, &these, j, -1))
        {
            for _ in 0..2 {
                rots.push((i, j));
                rotate_all(&mut these, (i, j));
            }
            happy[i] = true;
            antihappy[i] = false;
        }

        // Trick #3 (3 moves): two other columns equal or opposite to each
        // other; rotating one onto the other makes one of them antihappy,
        // then a 180 degree rotation fixes it together with this one.
        if !happy[i] {
            'search: for j in (0..4).filter(|&j| j != i) {
                for k in (0..4).filter(|&k| k != i && k != j) {
                    let sign = if column_equals(&these, j, &these, k, 1) {
                        1
                    } else if column_equals(&these, j, &these, k, -1) {
                        -1
                    } else {
                        continue;
                    };
                    rots.push((j, k));
                    rotate_all(&mut these, (j, k));
                    let other_antihappy_axis = if sign == 1 { j } else { k };
                    for _ in 0..2 {
                        rots.push((i, other_antihappy_axis));
                        rotate_all(&mut these, (i, other_antihappy_axis));
                    }
                    happy[i] = true;
                    antihappy[i] = false;
                    break 'search;
                }
            }
        }

        if !happy[i] {
            // Every trick failed: it's inside out.
            return if none_if_mirror {
                Ok(None)
            } else {
                Err(SolveError::Internal(
                    "expected a rotation, found a mirror image",
                ))
            };
        }
    }

    debug_assert!((0..4).all(|i| happy[i]
        && happy[i] == column_equals(&these, i, those, i, 1)
        && antihappy[i] == column_equals(&these, i, those, i, -1)));
    debug_assert!(
        original
            .iter()
            .zip(those)
            .all(|(from, to)| rot90s(&rots, *from) == *to)
    );
    Ok(Some(rots))
}

/// Twists of the single side `face_axis`/`face_sign` taking `these` to
/// `those`: a whole-puzzle rotation search with one extra constraint, that
/// the side's own center stays put. (As in the Java original, nothing checks
/// that the points are actually on that side - callers guarantee it.)
pub(crate) fn find_oneface_twist_sequence(
    face_axis: usize,
    face_sign: i8,
    these: &[Coords],
    those: &[Coords],
) -> SolveResult<Vec<Twist>> {
    if these.is_empty() {
        return Ok(Vec::new());
    }
    // Not necessarily the face center, but in its direction, which is all
    // that matters.
    let mut face_center = [0i8; 4];
    face_center[face_axis] = face_sign;
    let mut these = these.to_vec();
    these.push(face_center);
    let mut those = those.to_vec();
    those.push(face_center);
    let rots = find_rotation_sequence(&these, &those, false)?.ok_or(SolveError::Internal(
        "expected a rotation, found a mirror image",
    ))?;
    rots.into_iter()
        .map(|(from_axis, to_axis)| {
            // Keeping the face center fixed means no rotation found can
            // involve the face's own axis (its column only ever matches
            // itself), but check rather than build a malformed twist.
            if from_axis == face_axis || to_axis == face_axis {
                return Err(SolveError::Internal("a one-face twist moved its own face"));
            }
            Ok(make_twist90(face_axis, face_sign, from_axis, to_axis))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::coords::twist90s;

    /// A corner's four stickers, which pin down a rotation completely.
    fn corner_stickers(center: Coords) -> Vec<Coords> {
        (0..4)
            .map(|axis| {
                let mut s = center;
                s[axis] = center[axis] * 2;
                s
            })
            .collect()
    }

    fn random_rots(rng: &mut fastrand::Rng, len: usize) -> Vec<Rot> {
        (0..len)
            .map(|_| {
                let from = rng.usize(0..4);
                let to = loop {
                    let to = rng.usize(0..4);
                    if to != from {
                        break to;
                    }
                };
                (from, to)
            })
            .collect()
    }

    #[test]
    fn round_trips_random_rotations() {
        let mut rng = fastrand::Rng::with_seed(11);
        for _ in 0..500 {
            let these = corner_stickers([2, -2, 2, 2]);
            let len = rng.usize(0..8);
            let rots = random_rots(&mut rng, len);
            let those: Vec<Coords> = these.iter().map(|&c| rot90s(&rots, c)).collect();
            let found = find_rotation_sequence(&these, &those, false)
                .unwrap()
                .unwrap();
            for (from, to) in these.iter().zip(&those) {
                assert_eq!(rot90s(&found, *from), *to);
            }
        }
    }

    #[test]
    fn round_trips_partial_constraints() {
        // Fewer points than axes leave the search freedom - it just has to
        // hit these targets, not reproduce the original rotation.
        let mut rng = fastrand::Rng::with_seed(12);
        for _ in 0..500 {
            let these = vec![[2, 2, 0, 0], [4, 2, 2, 0]];
            let len = rng.usize(0..8);
            let rots = random_rots(&mut rng, len);
            let those: Vec<Coords> = these.iter().map(|&c| rot90s(&rots, c)).collect();
            let found = find_rotation_sequence(&these, &those, false)
                .unwrap()
                .unwrap();
            for (from, to) in these.iter().zip(&those) {
                assert_eq!(rot90s(&found, *from), *to);
            }
        }
    }

    #[test]
    fn mirror_image_is_none_or_error() {
        let these = corner_stickers([2, 2, 2, 2]);
        // Reflect in axis 0: a determinant -1 map, no rotation does this.
        let those: Vec<Coords> = these.iter().map(|c| [-c[0], c[1], c[2], c[3]]).collect();
        assert_eq!(find_rotation_sequence(&these, &those, true), Ok(None));
        assert!(matches!(
            find_rotation_sequence(&these, &those, false),
            Err(SolveError::Internal(_))
        ));
    }

    #[test]
    fn oneface_twists_stay_on_their_face_and_hit_the_target() {
        // A 180 degree turn of side +X taking one edge cubie to another.
        let twists = find_oneface_twist_sequence(0, 1, &[[2, 2, 2, 0]], &[[2, -2, -2, 0]]).unwrap();
        assert_eq!(twists.len(), 2);
        for t in &twists {
            assert_eq!((t.face_axis, t.face_sign), (0, 1));
        }
        assert_eq!(twist90s(&twists, [2, 2, 2, 0]), [2, -2, -2, 0]);
        assert!(
            find_oneface_twist_sequence(0, 1, &[], &[])
                .unwrap()
                .is_empty()
        );
    }
}
