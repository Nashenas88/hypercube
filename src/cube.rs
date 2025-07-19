//! 4D hypercube data structures and geometry.
//!
//! This module defines the core data structures for representing a 4D Rubik's cube,
//! including colors, individual stickers, 3D sides, and the complete hypercube.

use nalgebra::Vector4;

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    Brown,
}

/// Individual sticker on the hypercube surface.
///
/// Each sticker represents one colored square that would be visible on the surface
/// of the 4D hypercube. Contains color and 4D position information.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sticker {
    /// Color of this sticker
    pub(crate) color: Color,
    /// Position within the 4D hypercube coordinate system
    pub(crate) position: Vector4<f32>,
}

/// One 3D face of the 4D hypercube.
///
/// Each face is a 3x3x3 arrangement of stickers, representing one of the 8 cubic
/// cells that make up the tesseract (4D hypercube).
#[derive(Clone, Debug)]
pub(crate) struct Face {
    /// Collection of 27 stickers arranged in a 3x3x3 cube
    pub(crate) stickers: Vec<Sticker>,
}

impl Face {
    /// Creates a new 3D face with all stickers of the same color.
    ///
    /// Generates a 3x3x3 grid of stickers positioned using authentic tesseract geometry.
    /// Uses the coordinate pattern {-2/3, 0, +2/3} for the free dimensions.
    ///
    /// # Arguments
    /// * `center_color` - Color for all stickers on this side
    /// * `face_center` - 4D center position for this face of the tesseract
    /// * `fixed_dim` - Which dimension (0=X, 1=Y, 2=Z, 3=W) is fixed for this face
    ///
    /// # Returns
    /// A new side with 27 stickers arranged in a 3D grid
    pub(crate) fn new(center_color: Color, face_center: Vector4<f32>, fixed_dim: usize) -> Self {
        let mut stickers = Vec::with_capacity(27);

        // Generate 3x3x3 grid with authentic tesseract spacing
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    // Convert grid indices to sticker coordinates: -2/3, 0, +2/3
                    let grid_coords = [
                        (i as f32 - 1.0) * 2.0 / 3.0,
                        (j as f32 - 1.0) * 2.0 / 3.0,
                        (k as f32 - 1.0) * 2.0 / 3.0,
                    ];

                    // Apply offsets to the free dimensions only
                    let mut position = face_center;
                    let mut coord_idx = 0;
                    for dim in 0..4 {
                        if dim != fixed_dim {
                            position[dim] += grid_coords[coord_idx];
                            coord_idx += 1;
                        }
                    }

                    stickers.push(Sticker {
                        color: center_color,
                        position,
                    });
                }
            }
        }
        Self { stickers }
    }
}

/// The complete 4D hypercube (tesseract) structure.
///
/// Consists of 8 cubic faces arranged in 4D space, representing a 4D Rubik's cube.
/// Each face is a 3x3x3 arrangement of colored stickers.
#[derive(Debug, Clone)]
pub(crate) struct Hypercube {
    /// The 8 cubic faces that make up the tesseract
    pub(crate) faces: Vec<Face>,
}

impl Hypercube {
    /// Creates a new hypercube in solved state.
    ///
    /// Initializes 8 sides with distinct colors, positioned at the vertices
    /// of a tesseract. Each face is a 3x3x3 grid positioned at the correct
    /// 4D location.
    ///
    /// # Returns
    /// A solved 4D hypercube ready for visualization and manipulation
    pub(crate) fn new() -> Self {
        let colors = [
            Color::White,
            Color::Yellow,
            Color::Blue,
            Color::Green,
            Color::Red,
            Color::Orange,
            Color::Purple,
            Color::Brown,
        ];

        let faces = colors
            .iter()
            .zip(FACE_CENTERS.iter())
            .zip(FIXED_DIMS.iter())
            .map(|((&color, &face_center), &fixed_dim)| Face::new(color, face_center, fixed_dim))
            .collect();

        Self { faces }
    }
}

impl From<Color> for Vector4<f32> {
    /// Converts a color enum to RGBA color values.
    ///
    /// Maps each hypercube color to its corresponding RGBA representation
    /// for rendering purposes.
    ///
    /// # Arguments
    /// * `color` - The color enum value to convert
    fn from(color: Color) -> Self {
        match color {
            Color::White => Vector4::new(1.0, 1.0, 1.0, 1.0),
            Color::Yellow => Vector4::new(1.0, 1.0, 0.0, 1.0),
            Color::Blue => Vector4::new(0.1, 0.1, 1.0, 1.0),
            Color::Green => Vector4::new(0.0, 1.0, 0.0, 1.0),
            Color::Red => Vector4::new(1.0, 0.0, 0.0, 1.0),
            Color::Orange => Vector4::new(1.0, 0.5, 0.0, 1.0),
            Color::Purple => Vector4::new(0.0, 0.5, 1.0, 1.0),
            Color::Brown => Vector4::new(0.5, 0.25, 0.0, 1.0),
        }
    }
}

/// 36 vertices for a cube (6 faces × 6 vertices per face using 2 triangles each).
///
/// Each face is defined by 2 triangles (6 vertices total).
/// Vertices are arranged by face: front, right, back, left, top, bottom.
/// Scaled to 1/3 size to match the original sticker scale.
#[rustfmt::skip]
pub(crate) const CUBE_VERTICES: [[f32; 3]; 36] = [
    // Front face (2 triangles: 0,1,2 and 2,3,0)
    [-1.0, -1.0, -1.0], // 0
    [ 1.0, -1.0, -1.0], // 1
    [ 1.0,  1.0, -1.0], // 2
    [ 1.0,  1.0, -1.0], // 2
    [-1.0,  1.0, -1.0], // 3
    [-1.0, -1.0, -1.0], // 0
    // Right face (2 triangles: 1,5,6 and 6,2,1)
    [ 1.0, -1.0, -1.0], // 1
    [ 1.0, -1.0,  1.0], // 5
    [ 1.0,  1.0,  1.0], // 6
    [ 1.0,  1.0,  1.0], // 6
    [ 1.0,  1.0, -1.0], // 2
    [ 1.0, -1.0, -1.0], // 1
    // Back face (2 triangles: 5,4,7 and 7,6,5)
    [ 1.0, -1.0,  1.0], // 5
    [-1.0, -1.0,  1.0], // 4
    [-1.0,  1.0,  1.0], // 7
    [-1.0,  1.0,  1.0], // 7
    [ 1.0,  1.0,  1.0], // 6
    [ 1.0, -1.0,  1.0], // 5
    // Left face (2 triangles: 4,0,3 and 3,7,4)
    [-1.0, -1.0,  1.0], // 4
    [-1.0, -1.0, -1.0], // 0
    [-1.0,  1.0, -1.0], // 3
    [-1.0,  1.0, -1.0], // 3
    [-1.0,  1.0,  1.0], // 7
    [-1.0, -1.0,  1.0], // 4
    // Top face (2 triangles: 3,2,6 and 6,7,3)
    [-1.0,  1.0, -1.0], // 3
    [ 1.0,  1.0, -1.0], // 2
    [ 1.0,  1.0,  1.0], // 6
    [ 1.0,  1.0,  1.0], // 6
    [-1.0,  1.0,  1.0], // 7
    [-1.0,  1.0, -1.0], // 3
    // Bottom face (2 triangles: 4,5,1 and 1,0,4)
    [-1.0, -1.0,  1.0], // 4
    [ 1.0, -1.0,  1.0], // 5
    [ 1.0, -1.0, -1.0], // 1
    [ 1.0, -1.0, -1.0], // 1
    [-1.0, -1.0, -1.0], // 0
    [-1.0, -1.0,  1.0], // 4
];

/// Used to manage winding issues that occur when rotating in 4D. Copied for each 4d face, and
/// each face can swap indices if there's a winding issue.
#[rustfmt::skip]
pub(crate) const BASE_INDICES: [u16; 36] = [
    0, 1, 2, 3, 4, 5,       // face 0
    6, 7, 8, 9, 10, 11,     // face 1
    12, 13, 14, 15, 16, 17, // face 2
    18, 19, 20, 21, 22, 23, // face 3
    24, 25, 26, 27, 28, 29, // face 4
    30, 31, 32, 33, 34, 35, // face 5
];
