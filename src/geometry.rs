//! 4D hypercube rendering-mesh geometry.
//!
//! Static, puzzle-state-independent tables: tesseract side positions, cube
//! mesh vertices, and winding-index tables used by the renderer and ray
//! caster. Puzzle state itself lives in `piece.rs`.

use nalgebra::Vector4;
use serde::{Deserialize, Serialize};

/// Face centers for the 8 faces of the tesseract
pub(crate) const FACE_CENTERS: [Vector4<f32>; 8] = [
    Vector4::new(0.0, 0.0, 0.0, -1.0), // Face 0: W = -1
    Vector4::new(0.0, 0.0, -1.0, 0.0), // Face 1: Z = -1
    Vector4::new(0.0, -1.0, 0.0, 0.0), // Face 2: Y = -1
    Vector4::new(-1.0, 0.0, 0.0, 0.0), // Face 3: X = -1
    Vector4::new(1.0, 0.0, 0.0, 0.0),  // Face 4: X = +1
    Vector4::new(0.0, 1.0, 0.0, 0.0),  // Face 5: Y = +1
    Vector4::new(0.0, 0.0, 1.0, 0.0),  // Face 6: Z = +1
    Vector4::new(0.0, 0.0, 0.0, 1.0),  // Face 7: W = +1
];

/// Fixed dimensions for each face (0=X,  1=Y, 2=Z, 3=W)
pub(crate) const FIXED_DIMS: [usize; 8] = [3, 2, 1, 0, 0, 1, 2, 3];

/// Colors for the 8 sides of the 4D hypercube.
///
/// Uses standard Rubik's cube colors for the first 6 sides, with two additional
/// colors (Purple and Brown) for the extra dimensions in 4D space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum Color {
    // Using standard Rubik's cube colors for the first 6
    White,
    Yellow,
    Blue,
    Green,
    Red,
    Orange,
    // Two more for the 4D aspect
    Purple,
    Cyan,
}

impl From<Color> for Vector4<f32> {
    /// Converts a color enum to RGBA color values.
    ///
    /// Maps each hypercube color to its corresponding RGBA representation
    /// for rendering purposes.
    ///
    /// # Arguments
    /// * `color` - The color enum value to convert
    #[rustfmt::skip]
    fn from(color: Color) -> Self {
        match color {
            // center
            Color::Cyan => Vector4::new(0.0, 1.0, 1.0, 1.0),    // #00FFFF
            // left
            Color::Green => Vector4::new(0.0, 1.0, 0.0, 1.0),   // #00FF00
            // bottom
            Color::Yellow => Vector4::new(1.0, 1.0, 0.0, 1.0),  // #FFFF00
            // front
            Color::Red => Vector4::new(1.0, 0.0, 0.0, 1.0),     // #FF0000
            // back
            Color::Orange => Vector4::new(1.0, 0.65, 0.0, 1.0), // #FFA500
            // top
            Color::White => Vector4::new(1.0, 1.0, 1.0, 1.0),   // #FFFFFF
            // right
            Color::Blue => Vector4::new(0.1, 0.1, 1.0, 1.0),    // #1A1AFF
            // void
            Color::Purple => Vector4::new(0.5, 0.0, 1.0, 1.0),  // #8000FF
        }
    }
}

/// 36 vertices for a cube (6 faces × 6 vertices per face using 2 triangles each).
///
/// Each face is defined by 2 triangles (6 vertices total).
/// Vertices are arranged by face: front, right, back, left, top, bottom.
/// Scaled to 1/3 size to match the original sticker scale.
#[rustfmt::skip]
pub(crate) const BASE_CUBE_VERTICES: [[f32; 3]; 8] = [
    [-1.0, -1.0, -1.0], // 0
    [ 1.0, -1.0, -1.0], // 1
    [ 1.0,  1.0, -1.0], // 2
    [-1.0,  1.0, -1.0], // 3
    [-1.0, -1.0,  1.0], // 4
    [ 1.0, -1.0,  1.0], // 5
    [ 1.0,  1.0,  1.0], // 6
    [-1.0,  1.0,  1.0], // 7
];
#[rustfmt::skip]
pub(crate) const CUBE_VERTICES: [[f32; 3]; 36] = [
    // Front face (2 triangles: 0,1,2 and 2,3,0)
    BASE_CUBE_VERTICES[0],
    BASE_CUBE_VERTICES[1],
    BASE_CUBE_VERTICES[2],
    BASE_CUBE_VERTICES[2],
    BASE_CUBE_VERTICES[3],
    BASE_CUBE_VERTICES[0],
    // Right face (2 triangles: 1,5,6 and 6,2,1)
    BASE_CUBE_VERTICES[1],
    BASE_CUBE_VERTICES[5],
    BASE_CUBE_VERTICES[6],
    BASE_CUBE_VERTICES[6],
    BASE_CUBE_VERTICES[2],
    BASE_CUBE_VERTICES[1],
    // Back face (2 triangles: 5,4,7 and 7,6,5)
    BASE_CUBE_VERTICES[5],
    BASE_CUBE_VERTICES[4],
    BASE_CUBE_VERTICES[7],
    BASE_CUBE_VERTICES[7],
    BASE_CUBE_VERTICES[6],
    BASE_CUBE_VERTICES[5],
    // Left face (2 triangles: 4,0,3 and 3,7,4)
    BASE_CUBE_VERTICES[4],
    BASE_CUBE_VERTICES[0],
    BASE_CUBE_VERTICES[3],
    BASE_CUBE_VERTICES[3],
    BASE_CUBE_VERTICES[7],
    BASE_CUBE_VERTICES[4],
    // Top face (2 triangles: 3,2,6 and 6,7,3)
    BASE_CUBE_VERTICES[3],
    BASE_CUBE_VERTICES[2],
    BASE_CUBE_VERTICES[6],
    BASE_CUBE_VERTICES[6],
    BASE_CUBE_VERTICES[7],
    BASE_CUBE_VERTICES[3],
    // Bottom face (2 triangles: 4,5,1 and 1,0,4)
    BASE_CUBE_VERTICES[4],
    BASE_CUBE_VERTICES[5],
    BASE_CUBE_VERTICES[1],
    BASE_CUBE_VERTICES[1],
    BASE_CUBE_VERTICES[0],
    BASE_CUBE_VERTICES[4],
];

/// Used to manage winding issues that occur when rotating in 4D. Copied for each 4d face, and
/// each face can swap indices if there's a winding issue.
#[rustfmt::skip]
pub(crate) const VERTEX_NORMAL_INDICES: [u16; 36] = [
    0, 1, 2, 3, 4, 5,       // face 0
    6, 7, 8, 9, 10, 11,     // face 1
    12, 13, 14, 15, 16, 17, // face 2
    18, 19, 20, 21, 22, 23, // face 3
    24, 25, 26, 27, 28, 29, // face 4
    30, 31, 32, 33, 34, 35, // face 5
];

pub(crate) const NORMAL_TO_BASE_INDICES: [usize; 36] = [
    0, 1, 2, 2, 3, 0, // Front face (2 triangles: 0,1,2 and 2,3,0)
    1, 5, 6, 6, 2, 1, // Right face (2 triangles: 1,5,6 and 6,2,1)
    5, 4, 7, 7, 6, 5, // Back face (2 triangles: 5,4,7 and 7,6,5)
    4, 0, 3, 3, 7, 4, // Left face (2 triangles: 4,0,3 and 3,7,4)
    3, 2, 6, 6, 7, 3, // Top face (2 triangles: 3,2,6 and 6,7,3)
    4, 5, 1, 1, 0, 4, // Bottom face (2 triangles: 4,5,1 and 1,0,4)
];
