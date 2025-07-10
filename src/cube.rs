use nalgebra::{Vector3, Vector4, Matrix5};

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
    pub fn new(center_color: Color, w_position: f32) -> Self {
        let mut stickers = Vec::with_capacity(27);
        for i in -1..=1 {
            for j in -1..=1 {
                for k in -1..=1 {
                    stickers.push(Sticker {
                        color: center_color, // Initially, all stickers on a side are the same color
                        position: Vector4::new(i as f32, j as f32, k as f32, w_position),
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

        let sides = colors.iter().enumerate().map(|(i, &color)| {
            // Position sides along the W axis at -1 and +1
            let w_position = if i < 4 { -1.0 } else { 1.0 };
            Side::new(color, w_position)
        }).collect();

        Self { sides }
    }
}

// 4D projection utilities
pub fn project_4d_to_3d(point_4d: Vector4<f32>, viewer_distance: f32) -> Vector3<f32> {
    // Perspective projection from 4D to 3D
    // Similar to 3D->2D projection but with an extra dimension
    let w_distance = viewer_distance - point_4d.w;
    let scale = viewer_distance / w_distance;
    
    Vector3::new(
        point_4d.x * scale,
        point_4d.y * scale,
        point_4d.z * scale,
    )
}

pub fn rotate_4d(point: Vector4<f32>, rotation_matrix: Matrix5<f32>) -> Vector4<f32> {
    let homogeneous = rotation_matrix * Vector4::new(point.x, point.y, point.z, point.w).insert_row(4, 1.0);
    Vector4::new(homogeneous.x, homogeneous.y, homogeneous.z, homogeneous.w)
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
