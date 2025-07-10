use nalgebra::Vector4;

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

#[derive(Clone, Copy, Debug)]
pub struct Sticker {
    pub color: Color,
    // Position within the 3x3x3x3 hypercube grid
    pub position: Vector4<f32>,
}

#[derive(Clone, Debug)]
pub struct Side {
    pub stickers: Vec<Sticker>,
}

impl Side {
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

#[derive(Debug)]
pub struct Hypercube {
    // 8 sides, corresponding to the 8 cells of a tesseract
    pub sides: Vec<Side>,
}

impl Hypercube {
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

// Vertex data for a cube.
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

// Index data for a cube.
pub const INDICES: &[u16] = &[
    0, 1, 2, 2, 3, 0, // front
    1, 5, 6, 6, 2, 1, // right
    5, 4, 7, 7, 6, 5, // back
    4, 0, 3, 3, 7, 4, // left
    3, 2, 6, 6, 7, 3, // top
    4, 5, 1, 1, 0, 4, // bottom
];
