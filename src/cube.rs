//! 4D hypercube data structures and geometry.
//! 
//! This module defines the core data structures for representing a 4D Rubik's cube,
//! including colors, individual stickers, 3D sides, and the complete hypercube.

use nalgebra::Vector4;

/// Colors for the 8 sides of the 4D hypercube.
/// 
/// Uses standard Rubik's cube colors for the first 6 sides, with two additional
/// colors (Purple and Brown) for the extra dimensions in 4D space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
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
pub struct Sticker {
    /// Color of this sticker
    pub color: Color,
    /// Position within the 4D hypercube coordinate system
    pub position: Vector4<f32>,
}

/// One 3D side of the 4D hypercube.
/// 
/// Each side is a 3x3x3 arrangement of stickers, representing one of the 8 cubic
/// cells that make up the tesseract (4D hypercube).
#[derive(Clone, Debug)]
pub struct Side {
    /// Collection of 27 stickers arranged in a 3x3x3 cube
    pub stickers: Vec<Sticker>,
}

impl Side {
    /// Creates a new 3D side with all stickers of the same color.
    /// 
    /// Generates a 3x3x3 grid of stickers positioned around the given offset.
    /// Initially all stickers have the same color (solved state).
    /// 
    /// # Arguments
    /// * `center_color` - Color for all stickers on this side
    /// * `offset` - 4D position offset for this side within the hypercube
    /// 
    /// # Returns
    /// A new side with 27 stickers arranged in a 3D grid
    pub fn new(center_color: Color, offset: Vector4<f32>) -> Self {
        let mut stickers = Vec::with_capacity(27);
        for i in -1..=1 {
            for j in -1..=1 {
                for k in -1..=1 {
                    stickers.push(Sticker {
                        color: center_color, // Initially, all stickers on a side are the same color
                        position: Vector4::new(i as f32, j as f32, k as f32, 0.0) + offset,
                    });
                }
            }
        }
        Self { stickers }
    }
}

/// The complete 4D hypercube (tesseract) structure.
/// 
/// Consists of 8 cubic sides arranged in 4D space, representing a 4D Rubik's cube.
/// Each side is a 3x3x3 arrangement of colored stickers.
#[derive(Debug)]
pub struct Hypercube {
    /// The 8 cubic sides that make up the tesseract
    pub sides: Vec<Side>,
}

impl Hypercube {
    /// Creates a new hypercube in solved state.
    /// 
    /// Initializes 8 sides with distinct colors, positioned at the vertices
    /// of a tesseract. Each side starts with all stickers of the same color.
    /// 
    /// # Returns
    /// A solved 4D hypercube ready for visualization and manipulation
    pub fn new() -> Self {
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

        // Position the 8 3D cubes: 7 in first W plane arranged around center, 1 in second W plane
        let positions = [
            Vector4::new( 0.0,  0.0,  0.0, -2.0), // White - center of first W plane
            Vector4::new( 4.0,  0.0,  0.0, -2.0), // Yellow - right
            Vector4::new(-4.0,  0.0,  0.0, -2.0), // Blue - left
            Vector4::new( 0.0,  4.0,  0.0, -2.0), // Green - up
            Vector4::new( 0.0, -4.0,  0.0, -2.0), // Red - down
            Vector4::new( 0.0,  0.0,  4.0, -2.0), // Orange - forward
            Vector4::new( 0.0,  0.0, -4.0, -2.0), // Purple - back
            Vector4::new( 0.0,  0.0,  0.0,  2.0), // Brown - center of second W plane
        ];

        let sides = colors.iter().zip(positions.iter()).map(|(&color, &position)| {
            Side::new(color, position)
        }).collect();

        Self { sides }
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
            Color::Blue => Vector4::new(0.0, 0.0, 1.0, 1.0),
            Color::Green => Vector4::new(0.0, 1.0, 0.0, 1.0),
            Color::Red => Vector4::new(1.0, 0.0, 0.0, 1.0),
            Color::Orange => Vector4::new(1.0, 0.5, 0.0, 1.0),
            Color::Purple => Vector4::new(0.0, 0.5, 1.0, 1.0),
            Color::Brown => Vector4::new(0.5, 0.25, 0.0, 1.0),
        }
    }
}

/// Vertex positions for a unit cube.
/// 
/// Defines the 8 vertices of a cube centered at origin with side length 1.
/// Used as the base geometry for all hypercube stickers.
pub const VERTICES: &[[f32; 3]] = &[
    [-0.5, -0.5, -0.5], // 0
    [0.5, -0.5, -0.5],  // 1
    [0.5, 0.5, -0.5],   // 2
    [-0.5, 0.5, -0.5],  // 3
    [-0.5, -0.5, 0.5],  // 4
    [0.5, -0.5, 0.5],   // 5
    [0.5, 0.5, 0.5],    // 6
    [-0.5, 0.5, 0.5],   // 7
];

/// Triangle indices for cube faces.
/// 
/// Defines how the vertices are connected to form the 6 faces of a cube.
/// Each face is made of 2 triangles (6 indices per face).
pub const INDICES: &[u16] = &[
    0, 1, 2, 2, 3, 0, // front
    1, 5, 6, 6, 2, 1, // right
    5, 4, 7, 7, 6, 5, // back
    4, 0, 3, 3, 7, 4, // left
    3, 2, 6, 6, 7, 3, // top
    4, 5, 1, 1, 0, 4, // bottom
];
