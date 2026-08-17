# moves.rs

Move application. A move rotates one "side" (27 pieces sharing a fixed coordinate on one axis) as a rigid 3×3×3 subcube; the rotation axis comes from the clicked piece's local coordinates on the 3 free axes, and turn angle (90°/180°/120°) depends on how many of those are nonzero. `discrete_rotation()` snaps a continuous rotation matrix to an exact signed permutation.
