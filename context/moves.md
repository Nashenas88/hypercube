# moves.rs

Move application. A move rotates one "side" (27 pieces sharing a fixed coordinate on one axis) as a rigid 3×3×3 subcube; the rotation axis comes from the clicked piece's local coordinates on the 3 free axes, and turn angle (90°/180°/120°) depends on how many of those are nonzero. `discrete_rotation()` snaps a continuous rotation matrix to an exact signed permutation.

`random_move()` picks a uniformly random actionable facet from `FACET_TABLE` and a random turn direction to derive a legal move; `Hypercube::apply_random_moves()` applies a run of these instantly (no animation), taking an explicit `&mut fastrand::Rng` for testability. Backs the UI's random-move/Scramble buttons.
