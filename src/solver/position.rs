//! Positioning - the port of `NdSolve.java`'s `position_ksticker_cubies` and
//! the functions under it. Moves every k-sticker piece to its home without
//! disturbing any piece with fewer stickers (pieces with more are fair
//! game), by splitting the permutation into 3-cycles and doing each one as
//! setup-into-an-L-shape, a fixed recipe, then undoing the setup.

use super::coords::{
    Coords, SLOT_COUNT, Twist, average, coords_of_slot, find_axis, is_cubie_center,
    make_signed_twist90, minus, n_indices_different, nonzero_count, norm_sqrd, plus,
    reverse_twists, sign_of, slot_of, twist90s,
};
use super::grid::WantsGrid;
use super::rotation::find_oneface_twist_sequence;
use super::{SolveError, SolveResult};

/// Twists that put every k-sticker piece in its home slot, leaving pieces
/// with fewer stickers untouched. Assumes the permutation on the k-sticker
/// pieces is even (the parity stage and the solvability check see to that).
pub(crate) fn position_ksticker_cubies(k: u8, grid: &WantsGrid) -> SolveResult<Vec<Twist>> {
    // Decompose the permutation on the k-sticker cubies into cycles,
    // omitting cycles of length 1.
    let mut cycles = Vec::new();
    let mut seen = [false; SLOT_COUNT];
    for slot in 0..SLOT_COUNT {
        let start = coords_of_slot(slot);
        if !is_cubie_center(start) || nonzero_count(start) != k {
            continue;
        }
        let mut cycle = Vec::new();
        let mut at = start;
        while !seen[slot_of(at)] {
            seen[slot_of(at)] = true;
            cycle.push(at);
            at = grid.target(at)?;
        }
        if cycle.len() > 1 {
            cycles.push(Some(cycle));
        }
    }

    let mut solution = Vec::new();
    for tricycle in split_into_tricycles(cycles)? {
        solution.extend(cycle_3_ksticker_cubies(k, &tricycle)?);
    }
    Ok(solution)
}

/// Splits a cycle decomposition (`a` wants to go to `b`, `b` to `c`, ...)
/// into 3-cycles to perform in order: `(a b c d e)` -> `(a b c)(a d e)`, and a
/// 2-cycle borrows the first element of a later even-length cycle,
/// `(a b)(c d e f)` -> `(a b c)(c a d e f)`. There's always one to borrow,
/// because the whole permutation is even.
fn split_into_tricycles(mut cycles: Vec<Option<Vec<Coords>>>) -> SolveResult<Vec<[Coords; 3]>> {
    let mut tricycles = Vec::new();
    for i in 0..cycles.len() {
        let Some(mut cycle) = cycles[i].take() else {
            continue; // already used up by an earlier 2-cycle
        };
        while cycle.len() != 1 {
            if cycle.len() == 2 {
                let other = (i + 1..cycles.len())
                    .find_map(|j| {
                        if cycles[j].as_ref().is_some_and(|c| c.len() % 2 == 0) {
                            cycles[j].take()
                        } else {
                            None
                        }
                    })
                    .ok_or(SolveError::Internal(
                        "an odd permutation reached the positioning stage",
                    ))?;
                tricycles.push([cycle[0], cycle[1], other[0]]);
                let mut next = other;
                next.insert(1, cycle[0]);
                cycle = next;
            } else {
                tricycles.push([cycle[0], cycle[1], cycle[2]]);
                cycle.drain(1..3);
            }
        }
    }
    Ok(tricycles)
}

/// Twists moving the k-sticker cubie at `tricycle[0]` to `tricycle[1]`,
/// `[1]` to `[2]` and `[2]` to `[0]`, without disturbing any other cubie with
/// k or fewer stickers.
pub(crate) fn cycle_3_ksticker_cubies(k: u8, tricycle: &[Coords; 3]) -> SolveResult<Vec<Twist>> {
    let (to_l, b, c) = take_tricycle_to_l(tricycle[0], tricycle[1], Some(tricycle[2]))?;
    let c = c.ok_or(SolveError::Internal("lost the third cubie of a 3-cycle"))?;
    let mut solution = to_l.clone();
    solution.extend(cycle_l_of_ksticker_cubies(k, tricycle[0], b, c)?);
    solution.extend(reverse_twists(&to_l));
    Ok(solution)
}

/// Twists 3-cycling k-sticker cubies `a -> b -> c` that form an L - all in
/// one 2-plane, with `b` the knee - without disturbing any other cubie with
/// k or fewer stickers.
fn cycle_l_of_ksticker_cubies(k: u8, a: Coords, b: Coords, c: Coords) -> SolveResult<Vec<Twist>> {
    debug_assert_eq!(norm_sqrd(a), norm_sqrd(b));
    debug_assert_eq!(norm_sqrd(b), norm_sqrd(c));
    if k != 2 {
        return cycle_l_of_kslabs(4 - k, a, b, c);
    }
    // Special case: the slab recursion can't cycle an L of 2-slabs, but we
    // don't need it to - cycling 2-sticker cubies is easy when the cubies
    // with more stickers may be messed up. On the standard Rubik's cube,
    // (BU FU FD) is U U F F U U F F.
    let ab_center = average(a, b); // the face containing a and b, not c
    let ab_face_axis = find_axis("the face of an L's first leg", |axis| ab_center[axis] != 0)?;
    let ab_face_sign = sign_of(a[ab_face_axis]);
    let bc_center = average(b, c); // the face containing b and c, not a
    let bc_face_axis = find_axis("the face of an L's second leg", |axis| bc_center[axis] != 0)?;
    let bc_face_sign = sign_of(b[bc_face_axis]);

    let ab = find_oneface_twist_sequence(ab_face_axis, ab_face_sign, &[a], &[b])?;
    let bc = find_oneface_twist_sequence(bc_face_axis, bc_face_sign, &[b], &[c])?;
    if ab.len() != 2 || bc.len() != 2 {
        return Err(SolveError::Internal("expected an L leg to be a half turn"));
    }
    Ok([ab.as_slice(), &bc, &ab, &bc].concat())
}

/// Twists 3-cycling an L of k-slabs `a -> b -> c` (a 0-slab is a corner
/// cubie, a 1-slab the row of cubies along an edge, ...; each represented
/// by its center, zero in its k slab directions), keeping each slab's
/// orientation consistent in the slab directions and disturbing nothing
/// else. Only works for `k <= d - 3`, which for this puzzle means k is 0
/// (corners) or 1 (the 3-sticker edges, extruded).
fn cycle_l_of_kslabs(k: u8, a: Coords, b: Coords, c: Coords) -> SolveResult<Vec<Twist>> {
    match k {
        1 => {
            // Base case: cycling three corners of a 3D Rubik's cube, in 8
            // moves. From Tom Davis's "Permutation Groups and Rubik's Cube":
            // (LUF RUB LUB) is U R U' L' U R' U' L. Seen from above:
            //     b = LUB <- RUB = a
            //         |     /
            //         v    /
            //     c = LUF
            let u_axis = find_axis("the face containing a, b and c", |axis| {
                a[axis] != 0 && a[axis] == b[axis] && b[axis] == c[axis]
            })?;
            let u_sign = sign_of(a[u_axis]);
            let l_axis = find_axis("the face containing b and c but not a", |axis| {
                a[axis] != 0 && a[axis] != b[axis] && b[axis] == c[axis]
            })?;
            let l_sign = sign_of(b[l_axis]);
            let f_axis = find_axis("the face containing c but not a or b", |axis| {
                a[axis] != 0 && a[axis] == b[axis] && b[axis] != c[axis]
            })?;
            let f_sign = sign_of(c[f_axis]);
            let (r_axis, r_sign) = (l_axis, -l_sign);

            let u = make_signed_twist90(u_axis, u_sign, f_axis, f_sign, l_axis, l_sign);
            let l = make_signed_twist90(l_axis, l_sign, u_axis, u_sign, f_axis, f_sign);
            let r = make_signed_twist90(r_axis, r_sign, f_axis, f_sign, u_axis, u_sign);
            Ok(vec![
                u,
                r,
                u.reversed(),
                l.reversed(),
                u,
                r.reversed(),
                u.reversed(),
                l,
            ])
        }
        0 => {
            // Inductive step. With d = a - b + c the square's fourth corner,
            // (a b c) = (d a b) (a b c d) (d a b)^-1 (a b c d)^-1. Do each
            // (d a b) on the (k+1)-slabs D, A, B (a, b, c, d extruded along
            // the face axis they share) recursively, and each (a b c d) as a
            // single quarter turn of that face.
            let d = plus(minus(a, b), c);
            let face_axis = find_axis("the face containing a, b and c", |axis| {
                a[axis] != 0 && a[axis] == b[axis] && b[axis] == c[axis]
            })?;
            let face_sign = sign_of(a[face_axis]);
            let extrude = |mut x: Coords| {
                x[face_axis] = 0;
                x
            };

            let inner = cycle_l_of_kslabs(k + 1, extrude(d), extrude(a), extrude(b))?;
            let turn = find_oneface_twist_sequence(face_axis, face_sign, &[a, b], &[b, c])?;
            if turn.len() != 1 {
                return Err(SolveError::Internal("expected a single quarter turn"));
            }
            debug_assert_eq!(twist90s(&turn, a), b);
            debug_assert_eq!(twist90s(&turn, b), c);
            debug_assert_eq!(twist90s(&turn, c), d);
            debug_assert_eq!(twist90s(&turn, d), a);
            Ok([
                inner.as_slice(),
                &turn,
                &reverse_twists(&inner),
                &reverse_twists(&turn),
            ]
            .concat())
        }
        _ => Err(SolveError::Internal(
            "cycling an L of slabs this big isn't possible",
        )),
    }
}

/// Twists bringing `mover` next to `anchor` (differing on exactly one axis)
/// without moving `anchor` or anything else whose position rules out the
/// allowed faces. The shared body of the two near-identical blocks in
/// `take_tricycle_to_L_of_ksticker_cubies_INDICES`: `target_face_ok` picks
/// which of `anchor`'s faces `mover` may land on the far side of, and
/// `helper_face_ok(axis, mover)` which of `mover`'s own faces may carry it
/// there.
fn bring_next_to(
    anchor: Coords,
    mut mover: Coords,
    target_face_ok: impl Fn(usize) -> bool,
    helper_face_ok: impl Fn(usize, Coords) -> bool,
) -> SolveResult<(Vec<Twist>, Coords)> {
    let mut solution = Vec::new();
    if n_indices_different(anchor, mover) > 1 {
        let target_face_axis = find_axis("a face to bring a cubie next to", &target_face_ok)?;
        let target_face_sign = -sign_of(anchor[target_face_axis]);
        let mut target = anchor;
        target[target_face_axis] = -anchor[target_face_axis];

        // Get mover onto the target face, if it isn't already there.
        if mover[target_face_axis] != target[target_face_axis] {
            let helper_face_axis = find_axis("a face to carry a cubie", |axis| {
                mover[axis] != 0 && helper_face_ok(axis, mover)
            })?;
            let helper_face_sign = sign_of(mover[helper_face_axis]);
            if helper_face_axis == target_face_axis {
                return Err(SolveError::Internal("the carrying face is the target face"));
            }
            // A point of the right type on both faces, as close as possible
            // to mover: shoot straight to the target face, and if that made
            // a zero coordinate nonzero, zero out another one of the same
            // size to restore the type (never the helper face's own axis).
            let mut waystation = mover;
            waystation[target_face_axis] = target[target_face_axis];
            if mover[target_face_axis] == 0 {
                let axis_to_zero_out = find_axis("an axis to zero out", |axis| {
                    axis != helper_face_axis
                        && axis != target_face_axis
                        && waystation[axis] != 0
                        && waystation[axis].abs() == waystation[target_face_axis].abs()
                })?;
                waystation[axis_to_zero_out] = 0;
            }
            solution.extend(find_oneface_twist_sequence(
                helper_face_axis,
                helper_face_sign,
                &[mover],
                &[waystation],
            )?);
            mover = waystation;
        }

        // Now twist the target face to take it the rest of the way.
        solution.extend(find_oneface_twist_sequence(
            target_face_axis,
            target_face_sign,
            &[mover],
            &[target],
        )?);
        mover = target;
    }
    Ok((solution, mover))
}

/// Setup twists taking the cubie centers `a`, `b`, `c` to an L (`a` fixed,
/// `b` the knee), freely messing up everything else. Returns the twists and
/// where `b` and `c` end up. `c` may be `None`, which just takes `a`, `b` to
/// an I - adjacent, differing on one axis.
pub(crate) fn take_tricycle_to_l(
    a: Coords,
    b: Coords,
    c: Option<Coords>,
) -> SolveResult<(Vec<Twist>, Coords, Option<Coords>)> {
    debug_assert_ne!(a, b);
    debug_assert!(c.is_none_or(|c| c != a && c != b));

    // Move b next to a, via any face of a's (landing opposite a) and any
    // face containing b but not a.
    let (mut solution, b) = bring_next_to(a, b, |axis| a[axis] != 0, |axis, m| a[axis] != m[axis])?;
    debug_assert_eq!(n_indices_different(a, b), 1);
    debug_assert_eq!(norm_sqrd(a), norm_sqrd(b));

    let c = match c {
        None => None,
        Some(c) => {
            // Move c next to b, not along the axis a and b differ on, via a
            // face containing c but neither of the others.
            let c = twist90s(&solution, c);
            let (more, c) = bring_next_to(
                b,
                c,
                |axis| b[axis] != 0 && b[axis] == a[axis],
                |axis, m| b[axis] != m[axis] && a[axis] != m[axis],
            )?;
            solution.extend(more);
            debug_assert_eq!(n_indices_different(b, c), 1);
            debug_assert_eq!(n_indices_different(c, a), 2);
            debug_assert_eq!(norm_sqrd(b), norm_sqrd(c));
            Some(c)
        }
    };
    Ok((solution, b, c))
}

/// Setup twists taking two cubie centers to an I (see `take_tricycle_to_l`),
/// returning the twists and where `b` ends up.
pub(crate) fn take_two_ksticker_cubies_to_i(
    a: Coords,
    b: Coords,
) -> SolveResult<(Vec<Twist>, Coords)> {
    let (solution, b, _) = take_tricycle_to_l(a, b, None)?;
    Ok((solution, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::{Hypercube, index_of};
    use crate::solver::apply_twists;
    use crate::solver::test_support::{
        assert_untouched_except, home_of, kcubie_centers, position_of_coords, sample,
    };

    /// Every ordered triple of distinct k-sticker cubie centers.
    fn all_tricycles(k: u8) -> Vec<[Coords; 3]> {
        let centers = kcubie_centers(k);
        let mut out = Vec::new();
        for &a in &centers {
            for &b in &centers {
                for &c in &centers {
                    if a != b && b != c && c != a {
                        out.push([a, b, c]);
                    }
                }
            }
        }
        out
    }

    /// Applies `cycle_3_ksticker_cubies` to a solved cube and checks it does
    /// exactly what it promises: the cubie from `[0]` now sits at `[1]` and
    /// so on, and every other cubie with k or fewer stickers is untouched.
    fn check_cycle_3(k: u8, tricycle: [Coords; 3]) {
        let twists = cycle_3_ksticker_cubies(k, &tricycle).unwrap();
        let mut cube = Hypercube::solved();
        apply_twists(&mut cube, &twists);
        for i in 0..3 {
            let piece = cube.pieces[index_of(position_of_coords(tricycle[(i + 1) % 3]))];
            assert_eq!(
                home_of(&piece),
                position_of_coords(tricycle[i]),
                "k={k} {tricycle:?}: wrong cubie at slot {}",
                (i + 1) % 3
            );
        }
        let exempt = tricycle.map(position_of_coords);
        assert_untouched_except(&cube, k, &exempt, &format!("cycle_3 k={k} {tricycle:?}"));
    }

    #[test]
    fn tricycle_counts() {
        assert_eq!(all_tricycles(2).len(), 24 * 23 * 22);
        assert_eq!(all_tricycles(3).len(), 32 * 31 * 30);
        assert_eq!(all_tricycles(4).len(), 16 * 15 * 14);
    }

    #[test]
    fn cycle_3_contract_sample() {
        for k in 2..=4 {
            for tricycle in sample(all_tricycles(k), 97) {
                check_cycle_3(k, tricycle);
            }
        }
    }

    #[test]
    #[ignore = "exhaustive: every tricycle of every piece type (~45k); run with --ignored"]
    fn cycle_3_contract_exhaustive() {
        for k in 2..=4 {
            for tricycle in all_tricycles(k) {
                check_cycle_3(k, tricycle);
            }
        }
    }

    #[test]
    fn take_tricycle_to_l_makes_an_l_and_leaves_a_alone() {
        for k in 2..=4 {
            for [a, b, c] in sample(all_tricycles(k), 13) {
                let (twists, b2, c2) = take_tricycle_to_l(a, b, Some(c)).unwrap();
                let c2 = c2.unwrap();
                assert_eq!(twist90s(&twists, a), a);
                assert_eq!(twist90s(&twists, b), b2);
                assert_eq!(twist90s(&twists, c), c2);
                assert_eq!(n_indices_different(a, b2), 1);
                assert_eq!(n_indices_different(b2, c2), 1);
                assert_eq!(n_indices_different(c2, a), 2);
            }
        }
    }

    /// Composes 3-cycles as "whatever's at `x[0]` moves to `x[1]`", in order,
    /// on labels `0..n`.
    fn compose(n: usize, tricycles: &[[Coords; 3]]) -> Vec<usize> {
        // where[label] = slot the thing originally at `label` is now in
        let mut at: Vec<usize> = (0..n).collect();
        for t in tricycles {
            let t = t.map(|c| c[0] as usize);
            for slot in at.iter_mut() {
                if let Some(i) = t.iter().position(|&x| x == *slot) {
                    *slot = t[(i + 1) % 3];
                }
            }
        }
        at
    }

    #[test]
    fn split_into_tricycles_reproduces_even_permutations() {
        let mut rng = fastrand::Rng::with_seed(5);
        let n = 12;
        let mut tested = 0;
        while tested < 300 {
            let mut perm: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut perm);
            let mut seen = vec![false; n];
            let mut cycles = Vec::new();
            let mut odd = false;
            for start in 0..n {
                let mut cycle = Vec::new();
                let mut at = start;
                while !seen[at] {
                    seen[at] = true;
                    cycle.push([at as i8, 0, 0, 0]);
                    at = perm[at];
                }
                if cycle.len() % 2 == 0 && !cycle.is_empty() {
                    odd = !odd;
                }
                if cycle.len() > 1 {
                    cycles.push(Some(cycle));
                }
            }
            if odd {
                continue;
            }
            tested += 1;
            let tricycles = split_into_tricycles(cycles).unwrap();
            assert_eq!(compose(n, &tricycles), perm);
        }
    }
}
