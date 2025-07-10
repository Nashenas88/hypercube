use nalgebra::{Vector3, Vector4};
use crate::cube::Hypercube;
use crate::renderer::Instance;

const STICKER_SPACING: f32 = 1.2;
const VIEWER_DISTANCE_4D: f32 = 3.0;
const MOUSE_SENSITIVITY: f32 = 0.5;

pub fn project_4d_to_3d(point_4d: Vector4<f32>, viewer_distance: f32) -> Vector3<f32> {
    let w_distance = viewer_distance - point_4d.w;
    let scale = viewer_distance / w_distance;
    
    Vector3::new(
        point_4d.x * scale,
        point_4d.y * scale,
        point_4d.z * scale,
    )
}

pub fn create_4d_rotation_xw(angle: f32) -> nalgebra::Matrix4<f32> {
    let cos_x = angle.cos();
    let sin_x = angle.sin();
    nalgebra::Matrix4::new(
        cos_x, 0.0, 0.0, -sin_x,
        0.0,   1.0, 0.0,  0.0,
        0.0,   0.0, 1.0,  0.0,
        sin_x, 0.0, 0.0,  cos_x,
    )
}

pub fn create_4d_rotation_yw(angle: f32) -> nalgebra::Matrix4<f32> {
    let cos_y = angle.cos();
    let sin_y = angle.sin();
    nalgebra::Matrix4::new(
        1.0,  0.0,   0.0,  0.0,
        0.0,  cos_y, 0.0, -sin_y,
        0.0,  0.0,   1.0,  0.0,
        0.0,  sin_y, 0.0,  cos_y,
    )
}

pub fn process_4d_rotation(current_rotation: &nalgebra::Matrix4<f32>, delta_x: f32, delta_y: f32) -> nalgebra::Matrix4<f32> {
    let angle_x = delta_x * MOUSE_SENSITIVITY * 0.01;
    let angle_y = delta_y * MOUSE_SENSITIVITY * 0.01;
    
    let rotation_xw = create_4d_rotation_xw(angle_x);
    let rotation_yw = create_4d_rotation_yw(angle_y);
    
    rotation_yw * rotation_xw * current_rotation
}

pub fn generate_instances(hypercube: &Hypercube, rotation_4d: &nalgebra::Matrix4<f32>) -> Vec<Instance> {
    let mut instances = Vec::new();
    
    for side in &hypercube.sides {
        for sticker in &side.stickers {
            let rotated_4d = rotation_4d * sticker.position.scale(STICKER_SPACING);
            let projected_3d = project_4d_to_3d(rotated_4d, VIEWER_DISTANCE_4D);
            
            instances.push(Instance {
                position: projected_3d,
                color: nalgebra::Vector4::from(sticker.color),
            });
        }
    }
    
    instances
}