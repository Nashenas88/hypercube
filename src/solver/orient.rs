//! Orienting - the port of `NdSolve.java`'s `orient_ksticker_cubies` and the
//! functions under it. Once the k-sticker pieces are all home, what's left
//! is sticker cycles within each piece. Those are regrouped into pairs on
//! two different pieces - "flip" (swap two stickers) for non-corners,
//! "twirl" (cycle three stickers, the two pieces in opposite directions) for
//! corners - and each pair is done as setup-into-a-flat-canonical-shape, a
//! fixed recipe, then undoing the setup.

use super::coords::{
    Coords, Rot, SLOT_COUNT, Twist, are_on_same_cubie, coords_of_slot, cubie_of, find_axis,
    is_sticker, legs_sum, make_signed_twist90, minus, n_indices_different, nonzero_count,
    reverse_twists, rot90, sign_of, slot_of, twist90s,
};
use super::grid::WantsGrid;
use super::position::take_two_ksticker_cubies_to_i;
use super::rotation::{find_oneface_twist_sequence, find_rotation_sequence};
use super::{SolveError, SolveResult};

/// A flip: two stickers on one piece to swap.
pub(crate) type Flip = [Coords; 2];
/// A twirl: three stickers on one corner to cycle, `[0] -> [1] -> [2]`.
pub(crate) type Twirl = [Coords; 3];

/// Twists that orient every (already positioned) k-sticker piece, leaving
/// pieces with fewer stickers untouched.
pub(crate) fn orient_ksticker_cubies(k: u8, grid: &WantsGrid) -> SolveResult<Vec<Twist>> {
    let cycles = sticker_cycles(k, grid)?;
    let mut solution = Vec::new();
    if k < 4 {
        for (a, b) in flip_pairs(cycles)? {
            solution.extend(flip_two_noncorner_cubies(k, a, b)?);
        }
    } else {
        for (a, b) in twirl_pairs(cycles)? {
            solution.extend(twirl_two_corner_cubies(a, b)?);
        }
    }
    Ok(solution)
}

/// The permutation on the k-sticker pieces' stickers, as cycles (omitting
/// cycles of length 1), each of which must stay on a single piece.
fn sticker_cycles(k: u8, grid: &WantsGrid) -> SolveResult<Vec<Vec<Coords>>> {
    let mut cycles = Vec::new();
    let mut seen = [false; SLOT_COUNT];
    for slot in 0..SLOT_COUNT {
        let start = coords_of_slot(slot);
        if !is_sticker(start) || nonzero_count(start) != k {
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
            if !cycle.iter().all(|&s| are_on_same_cubie(s, cycle[0])) {
                return Err(SolveError::Internal(
                    "a sticker cycle spans pieces after positioning",
                ));
            }
            cycles.push(cycle);
        }
    }
    Ok(cycles)
}

/// A rotation taking a cubie to a different cubie of the same type: the
/// rotation from axis 0 to the first later axis where the cubie isn't zero
/// on both (so it isn't rotated onto itself).
///
/// The Java original tests `cubieCenter[axis] == 0` on an *index*-space
/// clamp (always 1..n, never 0), so it always rotates axes 0 -> 1 - and when
/// the cubie is zero on both of those, the "helper" it picks is the very
/// cubie being fixed. Testing the coordinates, as its comment describes,
/// avoids that.
fn happy_helper_rot(cubie_center: Coords) -> SolveResult<Rot> {
    (1..4)
        .find(|&to| cubie_center[0] != 0 || cubie_center[to] != 0)
        .map(|to| (0, to))
        .ok_or(SolveError::Internal("no rotation moves this cubie"))
}

/// Regroups non-corner sticker cycles into flip pairs `(a b) (c d)` on two
/// different pieces: `(a b c d e)` gives up `(a b)` leaving `(a c d e)`,
/// paired with a swap taken from a later cycle on another piece. If every
/// cycle left is on one piece, a solved "happy helper" piece of the same
/// type lends a flip for all of them (used an even number of times, so it
/// ends up solved again).
fn flip_pairs(mut cycles: Vec<Vec<Coords>>) -> SolveResult<Vec<(Flip, Flip)>> {
    let mut pairs = Vec::new();
    let mut happy_helper_flip: Option<Flip> = None;
    for i in 0..cycles.len() {
        let mut cycle = cycles[i].clone();
        let cubie_center = cubie_of(cycle[0]);
        while cycle.len() != 1 {
            if let Some(helper) = happy_helper_flip {
                pairs.push(([cycle[0], cycle[1]], helper));
                cycle.remove(1);
            } else if let Some(h) = (i + 1..cycles.len())
                .find(|&h| cycles[h].len() != 1 && !are_on_same_cubie(cycles[h][0], cycle[0]))
            {
                pairs.push(([cycle[0], cycle[1]], [cycles[h][0], cycles[h][1]]));
                cycle.remove(1);
                cycles[h].remove(1);
            } else {
                // Only the rotated images of two of this cubie's stickers are
                // needed, not the helper cubie's center.
                let (from, to) = happy_helper_rot(cubie_center)?;
                happy_helper_flip = Some([rot90(from, to, cycle[0]), rot90(from, to, cycle[1])]);
            }
        }
    }
    Ok(pairs)
}

/// Whether some rotation takes the stickers `x` onto `y` - for corners in 4
/// dimensions, exactly when the two twirls turn the same way.
fn same_handedness(x: &Twirl, y: &Twirl) -> SolveResult<bool> {
    Ok(find_rotation_sequence(x, y, true)?.is_some())
}

/// Regroups corner sticker cycles into twirl pairs on two different corners
/// turning in opposite directions. Even-length cycles on the same corner are
/// first merged pairwise into odd ones (`(a b)(c d)` -> `(a b c)(c a d)`);
/// then `(a b c d e)` gives up `(a b c)` leaving `(a d e)`, paired with an
/// opposite twirl taken from a later cycle on another corner - reversing
/// that one's first three if needed - or from a happy helper corner, as in
/// `flip_pairs`.
fn twirl_pairs(mut cycles: Vec<Vec<Coords>>) -> SolveResult<Vec<(Twirl, Twirl)>> {
    for i in 0..cycles.len() {
        if cycles[i].len().is_multiple_of(2)
            && let Some(j) = (i + 1..cycles.len()).find(|&j| {
                cycles[j].len().is_multiple_of(2) && are_on_same_cubie(cycles[i][0], cycles[j][0])
            })
        {
            let first_of_j = cycles[j][0];
            cycles[i].push(first_of_j);
            let first_of_i = cycles[i][0];
            cycles[j].insert(1, first_of_i);
        }
        if cycles[i].len() % 2 != 1 {
            return Err(SolveError::Internal(
                "a corner has an odd sticker cycle left over",
            ));
        }
    }

    let mut pairs = Vec::new();
    let mut happy_helper: Option<(Twirl, Twirl)> = None; // (twirl, untwirl)
    let mut times_happy_helper_twirl_helped = 0i32;
    for i in 0..cycles.len() {
        let mut cycle = cycles[i].clone();
        let cubie_center = cubie_of(cycle[0]);
        while cycle.len() != 1 {
            if cycle.len() < 3 {
                return Err(SolveError::Internal("a corner twirl needs three stickers"));
            }
            let abc = [cycle[0], cycle[1], cycle[2]];
            if let Some((twirl, untwirl)) = happy_helper {
                let same_as_twirl = same_handedness(&abc, &twirl)?;
                if same_as_twirl == same_handedness(&abc, &untwirl)? {
                    return Err(SolveError::Internal("a twirl is both handed and not"));
                }
                let partner = if same_as_twirl {
                    times_happy_helper_twirl_helped -= 1;
                    untwirl
                } else {
                    times_happy_helper_twirl_helped += 1;
                    twirl
                };
                pairs.push((abc, partner));
                cycle.drain(1..3);
            } else if let Some(h) = (i + 1..cycles.len())
                .find(|&h| cycles[h].len() != 1 && !are_on_same_cubie(cycles[h][0], cycle[0]))
            {
                let helper = &mut cycles[h];
                let hij = [helper[0], helper[1], helper[2]];
                let jih = [helper[2], helper[1], helper[0]];
                let same_as_jih = same_handedness(&abc, &jih)?;
                if same_as_jih == same_handedness(&abc, &hij)? {
                    return Err(SolveError::Internal("a twirl is both handed and not"));
                }
                if same_as_jih {
                    // (a b c d e f g)(h i j k l) -> (a b c)(a d e f g)(h i j)(h k l)
                    pairs.push((abc, hij));
                    helper.drain(1..3);
                } else {
                    // (a b c d e f g)(h i j k l) -> (a b c)(a d e f g)(j i h)(h j i k l)
                    pairs.push((abc, jih));
                    helper.swap(1, 2);
                }
                cycle.drain(1..3);
            } else {
                let (from, to) = happy_helper_rot(cubie_center)?;
                let twirl = [
                    rot90(from, to, cycle[0]),
                    rot90(from, to, cycle[1]),
                    rot90(from, to, cycle[2]),
                ];
                happy_helper = Some((twirl, [twirl[2], twirl[1], twirl[0]]));
            }
        }
    }
    // In 5+ dimensions the helper can end up twirled a nonzero amount (mod
    // 3), needing a fix-up the Java original has; in 4 the twirl parity
    // check rules that out.
    if times_happy_helper_twirl_helped % 3 != 0 {
        return Err(SolveError::Internal(
            "the happy helper corner was left twirled",
        ));
    }
    Ok(pairs)
}

/// Twists flipping (swapping two stickers on each of) two k-sticker
/// non-corner pieces, without disturbing any other piece with k or fewer
/// stickers.
pub(crate) fn flip_two_noncorner_cubies(k: u8, a: Flip, b: Flip) -> SolveResult<Vec<Twist>> {
    let solution = if k == 2 {
        // Special case: the slab approach can't do it, but flipping two
        // 2-sticker pieces isn't too hard directly.
        let (to_i, b_center) = take_two_ksticker_cubies_to_i(cubie_of(a[0]), cubie_of(b[0]))?;
        let mut solution = to_i.clone();
        solution.extend(flip_i_of_2sticker_cubies(cubie_of(a[0]), b_center)?);
        solution.extend(reverse_twists(&to_i));
        solution
    } else {
        // Flatten (twists making the four stickers and two cubie centers
        // lie in one 2-plane with the cubies in an I), flip that simpler
        // canonical pair, then unflatten.
        let flatten = flatten_flip_pair_or_twirl_pair(&a, &b)?;
        let a1 = a.map(|s| twist90s(&flatten, s));
        let b1 = b.map(|s| twist90s(&flatten, s));
        let flip = flip_two_canonical_noncorner_cubies(a1, b1)?;
        let mut solution = flatten.clone();
        solution.extend(flip);
        solution.extend(reverse_twists(&flatten));
        solution
    };
    debug_assert_eq!(twist90s(&solution, a[0]), a[1]);
    debug_assert_eq!(twist90s(&solution, a[1]), a[0]);
    debug_assert_eq!(twist90s(&solution, b[0]), b[1]);
    debug_assert_eq!(twist90s(&solution, b[1]), b[0]);
    Ok(solution)
}

/// `flip_two_noncorner_cubies` for a pair already flattened into canonical
/// position (see `flatten_flip_pair_or_twirl_pair`).
fn flip_two_canonical_noncorner_cubies(a: Flip, b: Flip) -> SolveResult<Vec<Twist>> {
    let original_axis_that_was_zero =
        find_axis("an axis a flipped cubie is zero on", |axis| a[0][axis] == 0)?;
    flip_two_canonical_kslabs(a, b, original_axis_that_was_zero)
}

/// Flips two canonical, I-shaped 1-slabs - for this puzzle, the base case of
/// the Java original's recursion, flipping two 3-sticker pieces. (Its
/// "/"-shaped base case and recursive step only arise in 5+ dimensions.)
///
/// To flip the front-up and front-back edges of the center face (faces
/// L, R, F, B, U, D, and I, O for inner/outer): slide = slide front-up to
/// the down face using the front face; flip = flip it using the down face;
/// slide^-1; exchange = twist the up face exchanging front-up with
/// front-back; slide; flip^-1; slide^-1; exchange^-1. That reverses the slabs
/// in the R direction, so R is chosen along `original_axis_that_was_zero`,
/// where nobody minds.
fn flip_two_canonical_kslabs(
    a: Flip,
    b: Flip,
    original_axis_that_was_zero: usize,
) -> SolveResult<Vec<Twist>> {
    let a_center = cubie_of(a[0]);
    let b_center = cubie_of(b[0]);
    if n_indices_different(a_center, b_center) != 1 {
        return Err(SolveError::Internal(
            "flipping a /-shaped pair only arises in 5+ dimensions",
        ));
    }
    let stickers = [a[0], a[1], b[0], b[1]];

    // I = the face containing a and b but none of the four stickers.
    let i_axis = find_axis("the face holding none of the flipped stickers", |axis| {
        a_center[axis] != 0
            && a_center[axis] == b_center[axis]
            && stickers.iter().all(|s| s[axis].abs() != 4)
    })?;
    let i_sign = sign_of(a_center[i_axis]);
    // U = the other face containing a and b.
    let u_axis = find_axis("the other face containing both flipped cubies", |axis| {
        axis != i_axis && a_center[axis] != 0 && a_center[axis] == b_center[axis]
    })?;
    let u_sign = sign_of(a_center[u_axis]);
    // F = the third face containing a.
    let f_axis = find_axis("the third face containing a flipped cubie", |axis| {
        axis != i_axis && axis != u_axis && a_center[axis] != 0
    })?;
    let f_sign = sign_of(a_center[f_axis]);
    let r_axis = original_axis_that_was_zero;
    if r_axis == i_axis || r_axis == u_axis || r_axis == f_axis {
        return Err(SolveError::Internal("the flip's spare axis isn't spare"));
    }
    let r_sign = 1; // arbitrarily
    let (d_axis, d_sign) = (u_axis, -u_sign);

    let slide = [make_signed_twist90(
        f_axis, f_sign, u_axis, u_sign, i_axis, i_sign,
    )];
    // A single click on that edge in the 4D puzzle, as three quarter turns.
    let flip = [
        make_signed_twist90(d_axis, d_sign, i_axis, i_sign, r_axis, r_sign),
        make_signed_twist90(d_axis, d_sign, f_axis, f_sign, i_axis, i_sign),
        make_signed_twist90(d_axis, d_sign, r_axis, r_sign, f_axis, f_sign),
    ];
    let exchange = [make_signed_twist90(u_axis, u_sign, f_axis, f_sign, r_axis, r_sign); 2];
    let (unslide, unflip, unexchange) = (
        reverse_twists(&slide),
        reverse_twists(&flip),
        reverse_twists(&exchange),
    );
    Ok([
        slide.as_slice(),
        &flip,
        &unslide,
        &exchange,
        &slide,
        &unflip,
        &unslide,
        &unexchange,
    ]
    .concat())
}

/// Twists flipping two 2-sticker pieces that form an I (differing on exactly
/// one axis, opposite on it), disturbing no other 2-sticker piece. Uses the
/// standard Rubik's cube edge flip of UR and UB, R U D B2 U2 B' U B U B2 D'
/// R' U', conjugated by F R (which takes UF to UR) to flip UF and UB.
fn flip_i_of_2sticker_cubies(a: Coords, b: Coords) -> SolveResult<Vec<Twist>> {
    let u_axis = find_axis("the face containing both flipped cubies", |axis| {
        a[axis] != 0 && a[axis] == b[axis]
    })?;
    let u_sign = sign_of(a[u_axis]);
    let f_axis = find_axis("the face containing one flipped cubie", |axis| {
        a[axis] != b[axis]
    })?;
    let f_sign = sign_of(a[f_axis]);
    let r_axis = find_axis("a face next to both", |axis| {
        axis != u_axis && axis != f_axis
    })?;
    let r_sign = 1; // arbitrary

    let f = make_signed_twist90(f_axis, f_sign, u_axis, u_sign, r_axis, r_sign);
    let r = make_signed_twist90(r_axis, r_sign, f_axis, f_sign, u_axis, u_sign);
    let u = make_signed_twist90(u_axis, u_sign, r_axis, r_sign, f_axis, f_sign);
    let (f_, r_, u_) = (f.reversed(), r.reversed(), u.reversed());
    // B and D turn the faces opposite F and U the same way as F' and U'
    // (seen from outside, that's the matching clockwise turn).
    let opposite = |t: Twist| Twist {
        face_sign: -t.face_sign,
        ..t
    };
    let back = opposite(f_);
    let down = opposite(u_);
    let (back_, down_) = (back.reversed(), down.reversed());
    Ok(vec![
        f, r, r, u, down, back, back, u, u, back_, u, back, u, back, back, down_, r_, u_, r_, f_,
    ])
}

/// Twists twirling two corners (cycling three stickers on each, in opposite
/// directions), disturbing nothing else.
pub(crate) fn twirl_two_corner_cubies(a: Twirl, b: Twirl) -> SolveResult<Vec<Twist>> {
    // Flatten (twists making the six stickers and two cubie centers lie in
    // one 3-space with the cubies in an I), twirl that simpler canonical
    // pair, then unflatten.
    let flatten = flatten_flip_pair_or_twirl_pair(&a, &b)?;
    let a1 = a.map(|s| twist90s(&flatten, s));
    let b1 = b.map(|s| twist90s(&flatten, s));
    let mut solution = flatten.clone();
    solution.extend(twirl_two_canonical_corner_kslabs(0, a1, b1)?);
    solution.extend(reverse_twists(&flatten));
    Ok(solution)
}

/// Rotates a twirl's stickers (cyclically, so it's the same twirl) until the
/// one sticking out from `center` along `axis` comes first.
fn rotate_to_front(x: Twirl, center: Coords, axis: usize) -> SolveResult<Twirl> {
    (0..3)
        .find(|&i| x[i][axis] - center[axis] != 0)
        .map(|i| [x[i], x[(i + 1) % 3], x[(i + 2) % 3]])
        .ok_or(SolveError::Internal(
            "no twirled sticker faces the other corner",
        ))
}

/// `twirl_two_corner_cubies` for a pair already flattened into canonical
/// position. Works on k-slabs rather than just corners: in each axis where
/// the stickers' coordinates are non-extreme, everything is extruded into
/// slabs and the moves act on those.
fn twirl_two_canonical_corner_kslabs(k: u8, a: Twirl, b: Twirl) -> SolveResult<Vec<Twist>> {
    let a_center = cubie_of(a[0]);
    let b_center = cubie_of(b[0]);
    let a_legs = legs_sum(a_center, &a);
    let b_legs = legs_sum(b_center, &b);

    let solution = match k {
        1 => {
            // Base case: twirl two corners of a 3D Rubik's cube. From Tom
            // Davis's "Permutation Groups and Rubik's Cube": URB,URF ->
            // RBU,RFU is F D D F' R' D D R U R' D D R F D D F' U', twirling
            // URB counterclockwise and URF clockwise.
            //         B
            //     +-------b
            //     |       |
            //     |   U   |R
            //     |       |
            //     +-------a
            //         F
            let f_axis = find_axis("the face containing a but not b", |axis| {
                a_center[axis] != b_center[axis]
            })?;
            let f_sign = sign_of(a_center[f_axis]);
            // List the stickers facing F (and away from it) first.
            let a = rotate_to_front(a, a_center, f_axis)?;
            let b = rotate_to_front(b, b_center, f_axis)?;
            debug_assert_eq!(a[0][f_axis], f_sign * 4);
            debug_assert_eq!(b[0][f_axis], -f_sign * 4);
            // U = the direction of a[1], R = the direction of a[2].
            let u_axis = find_axis("a twirled sticker's direction", |axis| {
                a[1][axis] - a_center[axis] != 0
            })?;
            let u_sign = sign_of(a_center[u_axis]);
            let r_axis = find_axis("a twirled sticker's direction", |axis| {
                a[2][axis] - a_center[axis] != 0
            })?;
            let r_sign = sign_of(a_center[r_axis]);
            debug_assert_eq!(b[1][u_axis], u_sign * 4);
            debug_assert_eq!(b[2][r_axis], r_sign * 4);
            let (d_axis, d_sign) = (u_axis, -u_sign);

            let u = make_signed_twist90(u_axis, u_sign, r_axis, r_sign, f_axis, f_sign);
            let f = make_signed_twist90(f_axis, f_sign, u_axis, u_sign, r_axis, r_sign);
            let r = make_signed_twist90(r_axis, r_sign, f_axis, f_sign, u_axis, u_sign);
            let d = make_signed_twist90(d_axis, d_sign, f_axis, f_sign, r_axis, r_sign);
            let (u_, f_, r_) = (u.reversed(), f.reversed(), r.reversed());
            vec![f, d, d, f_, r_, d, d, r, u, r_, d, d, r, f, d, d, f_, u_]
        }
        0 => {
            // An axis where all six stickers agree (and aren't zero).
            let extrusion_axis = find_axis("an axis to extrude the twirl along", |axis| {
                a[0][axis] != 0
                    && [a[1], a[2], b[0], b[1], b[2]]
                        .iter()
                        .all(|s| s[axis] == a[0][axis])
            })?;
            // bc = the face containing b (and c) but not a.
            let bc_face_axis = find_axis("the face containing b but not a", |axis| {
                b_center[axis] != a_center[axis]
            })?;
            let bc_face_sign = sign_of(b_center[bc_face_axis]);
            // c is b reflected along one of its legs other than towards a
            // or along the extrusion axis; cFace = the face containing c but
            // not a or b.
            let c_face_axis = find_axis("the face containing c", |axis| {
                axis != bc_face_axis && axis != extrusion_axis && b_legs[axis] != 0
            })?;
            let c_face_sign = sign_of(b_legs[c_face_axis]);
            // abc = a sticker direction that's none of the above.
            let abc_face_axis = find_axis("the face containing a, b and c", |axis| {
                axis != extrusion_axis
                    && axis != bc_face_axis
                    && axis != c_face_axis
                    && a_legs[axis] != 0
            })?;
            let abc_face_sign = sign_of(a_center[abc_face_axis]);
            let extrusion_sign = sign_of(a_center[extrusion_axis]);
            let extrude = |x: Twirl| {
                x.map(|mut s| {
                    s[extrusion_axis] = 0;
                    s
                })
            };

            // b_to_c = twist bcFace so b goes to c; inner = twirl A,
            // untwirl B (recursively); c_to_a = twist the extrusion face so
            // c goes to a and b stays put; then undo inner, c_to_a, b_to_c.
            let b_to_c = [make_signed_twist90(
                bc_face_axis,
                bc_face_sign,
                abc_face_axis,
                abc_face_sign,
                c_face_axis,
                c_face_sign,
            )];
            let inner = twirl_two_canonical_corner_kslabs(k + 1, extrude(a), extrude(b))?;
            let c_to_a = [
                make_signed_twist90(
                    extrusion_axis,
                    extrusion_sign,
                    bc_face_axis,
                    bc_face_sign,
                    c_face_axis,
                    c_face_sign,
                ),
                make_signed_twist90(
                    extrusion_axis,
                    extrusion_sign,
                    c_face_axis,
                    c_face_sign,
                    abc_face_axis,
                    abc_face_sign,
                ),
            ];
            [
                b_to_c.as_slice(),
                &inner,
                &c_to_a,
                &reverse_twists(&inner),
                &reverse_twists(&c_to_a),
                &reverse_twists(&b_to_c),
            ]
            .concat()
        }
        _ => {
            return Err(SolveError::Internal(
                "twirling slabs this big isn't possible",
            ));
        }
    };
    debug_assert_eq!(twist90s(&solution, a[0]), a[1]);
    debug_assert_eq!(twist90s(&solution, a[1]), a[2]);
    debug_assert_eq!(twist90s(&solution, a[2]), a[0]);
    debug_assert_eq!(twist90s(&solution, b[0]), b[1]);
    debug_assert_eq!(twist90s(&solution, b[1]), b[2]);
    debug_assert_eq!(twist90s(&solution, b[2]), b[0]);
    Ok(solution)
}

/// Setup twists putting a flip pair or twirl pair (two or three stickers on
/// each of two same-type cubies `a`, `b`) in canonical relation: cubie
/// centers in an I, and the listed stickers plus both centers lying in as
/// low-dimensional a space as possible - 2D for a flip pair, 3D for a twirl
/// pair - with `a`'s stickers lined up with `b`'s.
pub(crate) fn flatten_flip_pair_or_twirl_pair(
    a: &[Coords],
    b: &[Coords],
) -> SolveResult<Vec<Twist>> {
    let two_or_three = a.len();
    debug_assert!(two_or_three == 2 || two_or_three == 3);
    debug_assert_eq!(b.len(), two_or_three);

    let (to_i, _) = take_two_ksticker_cubies_to_i(cubie_of(a[0]), cubie_of(b[0]))?;
    let mut solution = to_i.clone();
    let a: Vec<Coords> = a.iter().map(|&s| twist90s(&to_i, s)).collect();
    let mut b: Vec<Coords> = b.iter().map(|&s| twist90s(&to_i, s)).collect();
    let a_center = cubie_of(a[0]);
    let mut b_center = cubie_of(b[0]);

    // Bface = the face now containing b but not a.
    let mut b_face_axis = find_axis("the face containing b but not a", |axis| {
        a_center[axis] != 0 && a_center[axis] == -b_center[axis]
    })?;
    let mut b_face_sign = sign_of(b_center[b_face_axis]);

    // Both need a "leg" (a direction directly opposite a sticker) on the
    // axis connecting them.
    let a_legs = legs_sum(a_center, &a);
    let b_legs = legs_sum(b_center, &b);
    if a_legs[b_face_axis] == 0 || b_legs[b_face_axis] == 0 {
        // Find legs of a and of b, aligned neither with the b-a direction
        // nor with each other. If one of them already has a leg along the
        // b-a direction, that leg's ineligible, so it chooses first (so the
        // other can't steal its only eligible direction).
        let (a_leg_axis, b_leg_axis) = if a_legs[b_face_axis] != 0 {
            let a_leg = find_axis("a leg of a", |axis| {
                axis != b_face_axis && a_legs[axis] != 0
            })?;
            let b_leg = find_axis("a leg of b", |axis| {
                axis != b_face_axis && axis != a_leg && b_legs[axis] != 0
            })?;
            (a_leg, b_leg)
        } else {
            let b_leg = find_axis("a leg of b", |axis| {
                axis != b_face_axis && b_legs[axis] != 0
            })?;
            let a_leg = find_axis("a leg of a", |axis| {
                axis != b_face_axis && axis != b_leg && a_legs[axis] != 0
            })?;
            (a_leg, b_leg)
        };
        let a_leg_sign = sign_of(a_legs[a_leg_axis]);
        let b_leg_sign = sign_of(b_legs[b_leg_axis]);

        // Twist Bface so b ends up where b + aLeg was and b + bLeg where b
        // was; then b is diagonally across from a, and twisting the face
        // containing b along aLeg takes b to the end of aLeg and bLeg's end
        // to a. Neither twist moves a.
        let more = [
            make_signed_twist90(
                b_face_axis,
                b_face_sign,
                a_leg_axis,
                a_leg_sign,
                b_leg_axis,
                b_leg_sign,
            ),
            make_signed_twist90(
                a_leg_axis,
                a_leg_sign,
                b_leg_axis,
                b_leg_sign,
                b_face_axis,
                b_face_sign,
            ),
        ];
        solution.extend(more);
        b = b.iter().map(|&s| twist90s(&more, s)).collect();
        b_center = cubie_of(b[0]);

        // b moved, so the face containing it but not a did too.
        b_face_axis = a_leg_axis;
        b_face_sign = a_leg_sign;
        debug_assert_ne!(b_center[b_face_axis], 0);
        debug_assert_eq!(b_center[b_face_axis], -a_center[b_face_axis]);
        debug_assert_eq!(n_indices_different(a_center, b_center), 1);
    }

    // Now twist Bface, keeping b's center fixed, so all of b's remaining
    // legs line up with a's: the sum of b's remaining leg directions has to
    // rotate onto the sum of a's.
    {
        let mut a_remaining_legs = legs_sum(a_center, &a);
        let mut b_remaining_legs = legs_sum(b_center, &b);
        debug_assert_ne!(b_remaining_legs[b_face_axis], 0);
        debug_assert_eq!(
            b_remaining_legs[b_face_axis],
            -a_remaining_legs[b_face_axis]
        );
        a_remaining_legs[b_face_axis] = 0;
        b_remaining_legs[b_face_axis] = 0;
        let final_twists = find_oneface_twist_sequence(
            b_face_axis,
            b_face_sign,
            &[b_remaining_legs, b_center],
            &[a_remaining_legs, b_center],
        )?;
        solution.extend(&final_twists);
        b = b.iter().map(|&s| twist90s(&final_twists, s)).collect();
        debug_assert_eq!(cubie_of(b[0]), b_center);
    }

    debug_assert_eq!(
        (0..4)
            .filter(|&axis| {
                a.iter().any(|s| s[axis] != a_center[axis])
                    || b.iter().any(|s| s[axis] != b_center[axis])
            })
            .count(),
        two_or_three
    );
    debug_assert_eq!(n_indices_different(a_center, b_center), 1);

    // Finally a's stickers, in order, have to line up with b's (up to a
    // cyclic shift). Fixing that means reversing b's twirl, which is only
    // possible in 5+ dimensions.
    if two_or_three == 3 {
        let ai = (0..3)
            .find(|&i| a[i][b_face_axis] != a_center[b_face_axis])
            .ok_or(SolveError::Internal(
                "no twirled sticker faces the other corner",
            ))?;
        let bi = (0..3)
            .find(|&i| b[i][b_face_axis] != b_center[b_face_axis])
            .ok_or(SolveError::Internal(
                "no twirled sticker faces the other corner",
            ))?;
        debug_assert_eq!(minus(a[ai], a_center), minus(b_center, b[bi]));
        if minus(a[(ai + 1) % 3], a_center) != minus(b[(bi + 1) % 3], b_center) {
            return Err(SolveError::Internal(
                "reversing a twirl only arises in 5+ dimensions",
            ));
        }
    }
    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::{Hypercube, index_of};
    use crate::solver::apply_twists;
    use crate::solver::test_support::{
        assert_untouched_except, kcubie_centers, position_of_coords, sample, sticker_axis,
        stickers_on,
    };

    /// Every flip (an ordered pair of stickers) on every k-sticker piece;
    /// swapping is symmetric, so one order per pair.
    fn all_flips(k: u8) -> Vec<Flip> {
        let mut out = Vec::new();
        for center in kcubie_centers(k) {
            let stickers = stickers_on(center);
            for i in 0..stickers.len() {
                for j in i + 1..stickers.len() {
                    out.push([stickers[i], stickers[j]]);
                }
            }
        }
        out
    }

    fn all_flip_pairs(k: u8) -> Vec<(Flip, Flip)> {
        let flips = all_flips(k);
        let mut out = Vec::new();
        for &a in &flips {
            for &b in &flips {
                if !are_on_same_cubie(a[0], b[0]) {
                    out.push((a, b));
                }
            }
        }
        out
    }

    /// Every twirl on every corner: three of its four stickers, in either
    /// cyclic direction.
    fn all_twirls() -> Vec<Twirl> {
        let mut out = Vec::new();
        for center in kcubie_centers(4) {
            let s = stickers_on(center);
            for skip in 0..4 {
                let three: Vec<Coords> = (0..4).filter(|&i| i != skip).map(|i| s[i]).collect();
                out.push([three[0], three[1], three[2]]);
                out.push([three[2], three[1], three[0]]);
            }
        }
        out
    }

    /// Every pair of twirls on different corners turning opposite ways -
    /// the only pairs `twirl_pairs` produces.
    fn all_twirl_pairs() -> Vec<(Twirl, Twirl)> {
        let twirls = all_twirls();
        let mut out = Vec::new();
        for &a in &twirls {
            for &b in &twirls {
                if !are_on_same_cubie(a[0], b[0])
                    && same_handedness(&a, &[b[2], b[1], b[0]]).unwrap()
                {
                    out.push((a, b));
                }
            }
        }
        out
    }

    /// The solved piece at `stickers`' cubie, with the kind on each
    /// `stickers[i]`'s axis moved to `stickers[i + 1]`'s.
    fn cycled_piece(stickers: &[Coords]) -> crate::piece::Piece {
        let solved = Hypercube::solved();
        let original = solved.pieces[index_of(position_of_coords(stickers[0]))];
        let mut piece = original;
        for i in 0..stickers.len() {
            let from = sticker_axis(stickers[i]);
            let to = sticker_axis(stickers[(i + 1) % stickers.len()]);
            piece.kinds[to] = original.kinds[from];
        }
        piece
    }

    /// Applies a flip/twirl pair's twists to a solved cube and checks that
    /// exactly the listed stickers moved, as listed.
    fn check_sticker_pair(k: u8, twists: &[Twist], a: &[Coords], b: &[Coords], what: &str) {
        let mut cube = Hypercube::solved();
        apply_twists(&mut cube, twists);
        for stickers in [a, b] {
            let position = position_of_coords(stickers[0]);
            assert_eq!(
                cube.pieces[index_of(position)],
                cycled_piece(stickers),
                "{what}: wrong stickers on {position:?}"
            );
        }
        let exempt = [position_of_coords(a[0]), position_of_coords(b[0])];
        assert_untouched_except(&cube, k, &exempt, what);
    }

    #[test]
    fn pair_counts() {
        assert_eq!(all_flip_pairs(2).len(), 24 * 23);
        assert_eq!(all_flip_pairs(3).len(), 96 * 93);
        // Half of each other corner's 8 twirls turn the opposite way.
        assert_eq!(all_twirl_pairs().len(), 128 * 15 * 4);
    }

    #[test]
    fn flip_contract_2_sticker_exhaustive() {
        for (a, b) in all_flip_pairs(2) {
            let twists = flip_two_noncorner_cubies(2, a, b).unwrap();
            check_sticker_pair(2, &twists, &a, &b, &format!("flip k=2 {a:?} {b:?}"));
        }
    }

    fn check_flip_3(a: Flip, b: Flip) {
        let twists = flip_two_noncorner_cubies(3, a, b).unwrap();
        check_sticker_pair(3, &twists, &a, &b, &format!("flip k=3 {a:?} {b:?}"));
    }

    #[test]
    fn flip_contract_3_sticker_sample() {
        for (a, b) in sample(all_flip_pairs(3), 17) {
            check_flip_3(a, b);
        }
    }

    #[test]
    #[ignore = "exhaustive: every 3-sticker flip pair (~9k); run with --ignored"]
    fn flip_contract_3_sticker_exhaustive() {
        for (a, b) in all_flip_pairs(3) {
            check_flip_3(a, b);
        }
    }

    fn check_twirl(a: Twirl, b: Twirl) {
        let twists = twirl_two_corner_cubies(a, b).unwrap();
        check_sticker_pair(4, &twists, &a, &b, &format!("twirl {a:?} {b:?}"));
    }

    #[test]
    fn twirl_contract_sample() {
        for (a, b) in sample(all_twirl_pairs(), 17) {
            check_twirl(a, b);
        }
    }

    #[test]
    #[ignore = "exhaustive: every opposite-handed twirl pair (~8k); run with --ignored"]
    fn twirl_contract_exhaustive() {
        for (a, b) in all_twirl_pairs() {
            check_twirl(a, b);
        }
    }

    #[test]
    fn twirls_on_one_corner_split_by_handedness() {
        // Of a corner's 8 twirls, 4 turn each way; a twirl and its reverse
        // never match.
        let twirls = all_twirls();
        let first = twirls[0];
        let same = twirls[..8]
            .iter()
            .filter(|t| same_handedness(&first, t).unwrap())
            .count();
        assert_eq!(same, 4);
        assert!(!same_handedness(&first, &[first[2], first[1], first[0]]).unwrap());
    }

    #[test]
    fn happy_helper_rot_moves_every_cubie() {
        for k in 2..=4 {
            for center in kcubie_centers(k) {
                let (from, to) = happy_helper_rot(center).unwrap();
                assert_ne!(rot90(from, to, center), center, "{center:?}");
            }
        }
    }
}
