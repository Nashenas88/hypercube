//! NdSolve's own coordinate system, kept alongside `piece.rs`'s undoubled
//! `[i8;4]` positions because later solving stages (not yet ported) need to
//! name a single sticker, not just a piece: a piece's cubie center sits at
//! `2 * position`, and a sticker facing axis `A` with sign `S` sits at that
//! same point except with its `A` coordinate pushed out to `4 * S`. Doubling
//! everything first is what makes "sticker" and "cubie center" land on two
//! different, disjoint grids without needing separate coordinate types -
//! exactly as in `NdSolve.java`.

/// A point in the doubled coordinate system: cubie centers have every
/// coordinate in `{-2, 0, 2}`; a sticker additionally has one coordinate
/// (the axis it faces) in `{-4, 4}`.
pub(crate) type Coords = [i8; 4];

/// The doubled cubie-center coordinates for a piece at `position`.
pub(crate) fn cubie_center_coords(position: [i8; 4]) -> Coords {
    position.map(|c| c * 2)
}

/// The doubled sticker coordinates for the facet of `position` facing
/// `axis` (whose sign is `position[axis]`, which must be nonzero).
pub(crate) fn sticker_coords(position: [i8; 4], axis: usize) -> Coords {
    let mut coords = cubie_center_coords(position);
    debug_assert_ne!(
        position[axis], 0,
        "sticker axis must be one this piece has a facet on"
    );
    coords[axis] = position[axis] * 4;
    coords
}

/// Rotates `coords` by the 90 degree rotation that takes `+from_axis` to
/// `+to_axis` (and `+to_axis` to `-from_axis`), leaving every other
/// coordinate unchanged.
pub(crate) fn rot90(from_axis: usize, to_axis: usize, coords: Coords) -> Coords {
    let mut result = coords;
    result[to_axis] = coords[from_axis];
    result[from_axis] = -coords[to_axis];
    result
}

/// Applies a sequence of `(from_axis, to_axis)` 90 degree rotations in order.
pub(crate) fn rot90s(rots: &[(usize, usize)], mut coords: Coords) -> Coords {
    for &(from_axis, to_axis) in rots {
        coords = rot90(from_axis, to_axis, coords);
    }
    coords
}

/// A single 90 degree turn of one outer side: `face_axis`/`face_sign` select
/// the side (`coords[face_axis]` matching `face_sign`'s sign is the
/// condition for being on it), `from_axis`/`to_axis` the rotation, exactly
/// like `rot90`. Unlike `NdSolve.java`'s `twist`, there's no `slices_mask`
/// field: every twist in this project turns a whole outer side (the puzzle
/// has no independently-turnable inner slices), so the mask would always be
/// the single bit selecting that outer layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Twist {
    pub(crate) face_axis: usize,
    pub(crate) face_sign: i8,
    pub(crate) from_axis: usize,
    pub(crate) to_axis: usize,
}

/// Builds a `Twist`, asserting its axes are well-formed (in range, distinct,
/// and `face_sign` is +-1) - the same sanity check `NdSolve.java`'s
/// `makeTwist90` performs.
pub(crate) fn make_twist90(
    face_axis: usize,
    face_sign: i8,
    from_axis: usize,
    to_axis: usize,
) -> Twist {
    debug_assert!(face_axis < 4 && from_axis < 4 && to_axis < 4);
    debug_assert!(face_sign == 1 || face_sign == -1);
    debug_assert_ne!(face_axis, from_axis);
    debug_assert_ne!(face_axis, to_axis);
    debug_assert_ne!(from_axis, to_axis);
    Twist {
        face_axis,
        face_sign,
        from_axis,
        to_axis,
    }
}

impl Twist {
    /// The twist that undoes this one: rotating `to_axis` back to
    /// `from_axis` exactly cancels rotating `from_axis` to `to_axis`.
    pub(crate) fn reversed(&self) -> Twist {
        Twist {
            face_axis: self.face_axis,
            face_sign: self.face_sign,
            from_axis: self.to_axis,
            to_axis: self.from_axis,
        }
    }
}

/// Applies a single twist to one point: rotates it iff it's on the twist's
/// side (`coords[face_axis]` has the same sign as `face_sign`), otherwise
/// leaves it unchanged.
pub(crate) fn twist90(twist: &Twist, coords: Coords) -> Coords {
    let on_this_side = coords[twist.face_axis] as i32 * twist.face_sign as i32 > 0;
    if on_this_side {
        rot90(twist.from_axis, twist.to_axis, coords)
    } else {
        coords
    }
}

/// Applies a sequence of twists in order.
pub(crate) fn twist90s(twists: &[Twist], mut coords: Coords) -> Coords {
    for twist in twists {
        coords = twist90(twist, coords);
    }
    coords
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rot90_then_its_reverse_is_identity() {
        let coords: Coords = [2, -2, 4, 0];
        let rotated = rot90(0, 1, coords);
        let back = rot90(1, 0, rotated);
        assert_eq!(back, coords);
    }

    #[test]
    fn rot90_leaves_other_axes_untouched() {
        let coords: Coords = [2, -2, 4, -4];
        let rotated = rot90(0, 2, coords);
        assert_eq!(rotated[1], coords[1]);
        assert_eq!(rotated[3], coords[3]);
    }

    #[test]
    fn rot90_matches_worked_example() {
        // +X -> +Y, +Y -> -X.
        assert_eq!(rot90(0, 1, [1, 0, 0, 0]), [0, 1, 0, 0]);
        assert_eq!(rot90(0, 1, [0, 1, 0, 0]), [-1, 0, 0, 0]);
    }

    #[test]
    fn twist_reversed_swaps_from_and_to() {
        let t = make_twist90(3, 1, 0, 1);
        let r = t.reversed();
        assert_eq!(r.face_axis, t.face_axis);
        assert_eq!(r.face_sign, t.face_sign);
        assert_eq!(r.from_axis, t.to_axis);
        assert_eq!(r.to_axis, t.from_axis);
    }

    #[test]
    fn twist_then_its_reverse_is_identity() {
        let t = make_twist90(3, 1, 0, 2);
        let coords: Coords = [2, -2, 2, 2];
        let once = twist90(&t, coords);
        let back = twist90(&t.reversed(), once);
        assert_eq!(back, coords);
    }

    #[test]
    fn twist_leaves_other_sides_untouched() {
        let t = make_twist90(3, 1, 0, 1);
        let coords: Coords = [2, -2, 2, -2]; // face_axis=3 coord is -2, wrong sign
        assert_eq!(twist90(&t, coords), coords);
    }

    #[test]
    fn twist_leaves_center_slice_untouched() {
        let t = make_twist90(3, 1, 0, 1);
        let coords: Coords = [2, -2, 2, 0]; // face_axis coord is 0
        assert_eq!(twist90(&t, coords), coords);
    }

    #[test]
    fn sticker_coords_extends_only_the_facing_axis() {
        let position = [1i8, -1, 0, 1];
        let coords = sticker_coords(position, 3);
        assert_eq!(coords, [2, -2, 0, 4]);
        let coords = sticker_coords(position, 0);
        assert_eq!(coords, [4, -2, 0, 2]);
    }
}
