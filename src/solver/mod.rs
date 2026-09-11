//! A solver for the puzzle, ported from Don Hatch's `NdSolve.java` (Magic
//! Cube 4D, by Melinda Green & Don Hatch - http://superliminal.com/cube/cube.htm).
//!
//! NdSolve groups pieces by sticker count (2, 3, 4 - this project's face,
//! edge and corner pieces) and, for each group smallest first, positions
//! them and then orients them without disturbing any group already done.
//! Positioning decomposes the needed permutation into 3-cycles and performs
//! each as setup-turns-into-a-known-shape, a fixed move recipe, then undo
//! the setup; orienting pairs up the remaining sticker cycles ("flip" for
//! 2/3-sticker pieces, "twirl" for opposite-handed corner pairs) and applies
//! the same setup/recipe/undo pattern. NdSolve only ever outputs 90 degree
//! turns of one outer side; this project's 180 degree and 120 degree moves
//! are exactly compositions of those, so `native::merge_stage` folds
//! consecutive same-side turns back into this project's native move
//! vocabulary before they're played back.
//!
//! This module is under construction: `coords` and `native` establish the
//! coordinate system and move-vocabulary bridge the position/orientation
//! algorithm (not yet ported) will build on.
#![allow(dead_code)]

mod coords;
mod native;
