//! Ray casting for mouse-based sticker selection.
//!
//! This module provides CPU-based ray casting to detect which sticker
//! the mouse cursor is hovering over. Projects stickers from 4D to 3D space
//! and performs intersection testing in 3D.

use iced::{Point, Rectangle};
use nalgebra::{Matrix4, Point3, Vector3, Vector4};

use crate::app::AABBMode;
use crate::camera::{Camera, Projection};
use crate::geometry::NORMAL_TO_BASE_INDICES;
use crate::math::{
    BASE_STICKER_SIZE, GRID_EXTENT, depth_preserving_push, face_push_offset_3d, is_face_visible,
    project_cube_point, transform_sticker_vertices_to_3d,
};
use crate::piece::FACET_TABLE;
use crate::renderer::DebugInstanceWithDistance;
use crate::shader_widget::HypercubeShaderState;

/// 3D ray for intersection testing
#[derive(Debug, Clone)]
pub(crate) struct Ray {
    /// Ray origin point in 3D space
    pub(crate) origin: Point3<f32>,
    /// Ray direction vector (normalized)
    pub(crate) direction: Vector3<f32>,
    /// Ray inverse direction vector (normalized)
    pub(crate) inverse_direction: Vector3<f32>,
}

/// Axis-aligned bounding box in 3D space
#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) struct AABB {
    /// Minimum corner of the 3D bounding box
    pub(crate) min: Point3<f32>,
    /// Maximum corner of the 3D bounding box
    pub(crate) max: Point3<f32>,
}

/// Calculate mouse ray from screen coordinates through the 3D scene
pub(crate) fn calculate_mouse_ray(
    mouse_pos: Point,
    bounds: Rectangle,
    camera: &Camera,
    projection: &Projection,
) -> Ray {
    let ndc_x = (2.0 * mouse_pos.x / bounds.width) - 1.0;
    let ndc_y = 1.0 - (2.0 * mouse_pos.y / bounds.height);

    let view_matrix = camera.build_view_matrix();
    let proj_matrix = projection.build_projection_matrix();
    let view_proj_matrix = proj_matrix * view_matrix;

    let inv_view_proj = view_proj_matrix
        .try_inverse()
        .expect("View-projection matrix should be invertible");

    let ray_start_ndc = Vector4::new(ndc_x, ndc_y, -1.0, 1.0);
    let ray_end_ndc = Vector4::new(ndc_x, ndc_y, 1.0, 1.0);

    let ray_start_world = inv_view_proj * ray_start_ndc;
    let ray_end_world = inv_view_proj * ray_end_ndc;

    let ray_start = Point3::new(
        ray_start_world.x / ray_start_world.w,
        ray_start_world.y / ray_start_world.w,
        ray_start_world.z / ray_start_world.w,
    );
    let ray_end = Point3::new(
        ray_end_world.x / ray_end_world.w,
        ray_end_world.y / ray_end_world.w,
        ray_end_world.z / ray_end_world.w,
    );

    let direction = (ray_end - ray_start).normalize();

    Ray {
        origin: ray_start,
        direction,
        inverse_direction: direction.map(|i| 1.0 / i),
    }
}

/// Test ray intersection with 3D axis-aligned bounding box using the slab method
///
/// Returns Some(distance) if the ray intersects the box, None otherwise.
/// Uses the standard 3D slab method for ray-AABB intersection.
pub(crate) fn ray_intersects_aabb(ray: &Ray, aabb: &AABB) -> bool {
    let t1 = (aabb.min.x - ray.origin.x) * ray.inverse_direction.x;
    let t2 = (aabb.max.x - ray.origin.x) * ray.inverse_direction.x;

    let t3 = (aabb.min.y - ray.origin.y) * ray.inverse_direction.y;
    let t4 = (aabb.max.y - ray.origin.y) * ray.inverse_direction.y;

    let t5 = (aabb.min.z - ray.origin.z) * ray.inverse_direction.z;
    let t6 = (aabb.max.z - ray.origin.z) * ray.inverse_direction.z;

    // tmin = where the ray enters the box (latest of the three near
    // intersections); tmax = where it exits (earliest of the three far
    // intersections). The ray misses the box unless it enters before it
    // exits, and the box isn't entirely behind the ray's origin.
    let tmin = f32::max(
        f32::max(f32::min(t1, t2), f32::min(t3, t4)),
        f32::min(t5, t6),
    );
    let tmax = f32::min(
        f32::min(f32::max(t1, t2), f32::max(t3, t4)),
        f32::max(t5, t6),
    );

    !(tmax < 0.0 || tmin > tmax)
}

/// Test ray intersection with actual sticker geometry using transformed vertices
/// Returns Some(distance) if ray intersects any triangle of the sticker
pub(crate) fn ray_sticker_intersection(ray: &Ray, world_vertices: &[Point3<f32>]) -> Option<f32> {
    let mut closest_distance = f32::INFINITY;
    let mut hit = false;

    for triangle_vertices in NORMAL_TO_BASE_INDICES.chunks(3) {
        let v0 = world_vertices[triangle_vertices[0]];
        let v0 = Point3::new(v0[0], v0[1], v0[2]);
        let v1 = world_vertices[triangle_vertices[1]];
        let v1 = Point3::new(v1[0], v1[1], v1[2]);
        let v2 = world_vertices[triangle_vertices[2]];
        let v2 = Point3::new(v2[0], v2[1], v2[2]);

        if let Some(distance) = ray_triangle_intersection(ray, v0, v1, v2)
            && distance < closest_distance
        {
            closest_distance = distance;
            hit = true;
        }
    }

    if hit { Some(closest_distance) } else { None }
}

/// Test ray intersection with a triangle using Möller-Trumbore algorithm
/// Returns Some(distance) if ray intersects the triangle
fn ray_triangle_intersection(
    ray: &Ray,
    v0: Point3<f32>,
    v1: Point3<f32>,
    v2: Point3<f32>,
) -> Option<f32> {
    const EPSILON: f32 = 1e-8;

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;

    let h = ray.direction.cross(&edge2);
    let a = edge1.dot(&h);
    if a > -EPSILON && a < EPSILON {
        return None;
    }

    let f = 1.0 / a;
    let s = ray.origin - v0;

    let u = f * s.dot(&h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = s.cross(&edge1);
    let v = f * ray.direction.dot(&q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * edge2.dot(&q);
    if t > EPSILON { Some(t) } else { None }
}

/// Calculate sticker-level AABB using actual transformed vertices
pub(crate) fn calculate_sticker_aabb(world_vertices: &[Point3<f32>]) -> AABB {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut max_z = f32::NEG_INFINITY;

    for vertex in world_vertices {
        min_x = min_x.min(vertex[0]);
        min_y = min_y.min(vertex[1]);
        min_z = min_z.min(vertex[2]);
        max_x = max_x.max(vertex[0]);
        max_y = max_y.max(vertex[1]);
        max_z = max_z.max(vertex[2]);
    }

    AABB {
        min: Point3::new(min_x, min_y, min_z),
        max: Point3::new(max_x, max_y, max_z),
    }
}

/// Calculate face-level AABB that encompasses all stickers on a face
fn calculate_face_aabb(
    face_id: usize,
    rotation_4d: &Matrix4<f32>,
    sticker_scale: f32,
    gap_distance: f32,
    gap_distance_4d: f32,
    viewer_distance: f32,
) -> AABB {
    use crate::geometry::{BASE_CUBE_VERTICES, FACE_CENTERS, FIXED_DIMS};

    let face_center_4d = FACE_CENTERS[face_id];
    let fixed_dim = FIXED_DIMS[face_id];
    let push = face_push_offset_3d(face_center_4d, rotation_4d, viewer_distance) * gap_distance;
    let push_4d = depth_preserving_push(face_center_4d, rotation_4d, gap_distance_4d - 1.0);

    // Bounds must encompass every sticker on the face, not just its own
    // extent: the sticker grid spans positions -2/3, 0, +2/3 (a 4/3 range),
    // and BASE_CUBE_VERTICES is scaled by BASE_STICKER_SIZE in renderer.rs
    // then by sticker_scale in the shaders, so add GRID_EXTENT on top of the
    // scaled sticker size to cover the whole face.
    let mut transformed_corners_3d = Vec::with_capacity(8);
    let actual_sticker_size = BASE_STICKER_SIZE * sticker_scale;
    let face_bound = actual_sticker_size + GRID_EXTENT;

    for &base_vertex in &BASE_CUBE_VERTICES {
        let local_vertex =
            Vector3::new(base_vertex[0], base_vertex[1], base_vertex[2]) * face_bound;
        let corner_3d = project_cube_point(
            local_vertex,
            face_center_4d,
            fixed_dim,
            rotation_4d,
            viewer_distance,
            push_4d,
        ) + push;
        transformed_corners_3d.push(corner_3d);
    }

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut max_z = f32::NEG_INFINITY;

    for corner in &transformed_corners_3d {
        min_x = min_x.min(corner.x);
        min_y = min_y.min(corner.y);
        min_z = min_z.min(corner.z);
        max_x = max_x.max(corner.x);
        max_y = max_y.max(corner.y);
        max_z = max_z.max(corner.z);
    }

    AABB {
        min: Point3::new(min_x, min_y, min_z),
        max: Point3::new(max_x, max_y, max_z),
    }
}

/// Get debug color for each face (8 distinct colors for visualization)
fn get_face_debug_color(face_id: usize) -> [f32; 4] {
    match face_id {
        0 => [1.0, 0.0, 0.0, 0.3], // Red with 30% alpha
        1 => [0.0, 1.0, 0.0, 0.3], // Green
        2 => [0.0, 0.0, 1.0, 0.3], // Blue
        3 => [1.0, 1.0, 0.0, 0.3], // Yellow
        4 => [1.0, 0.0, 1.0, 0.3], // Magenta
        5 => [0.0, 1.0, 1.0, 0.3], // Cyan
        6 => [0.8, 0.4, 0.0, 0.3], // Orange
        7 => [0.5, 0.0, 0.8, 0.3], // Purple
        _ => [0.5, 0.5, 0.5, 0.3], // Gray fallback
    }
}

/// Find the sticker that the 3D mouse ray intersects
/// Returns the sticker index and debug AABBs for intersected faces/stickers
pub(crate) fn find_intersected_sticker(
    ray: &Ray,
    state: &HypercubeShaderState,
    sticker_scale: f32,
    gap_distance: f32,
    gap_distance_4d: f32,
    viewer_distance: f32,
    aabb_mode: AABBMode,
) -> (Option<usize>, Vec<DebugInstanceWithDistance>) {
    let camera_pos = [state.camera.eye.x, state.camera.eye.y, state.camera.eye.z];

    // First, determine which faces are visible and ray-intersectable
    let mut intersectable_faces = Vec::new();
    let mut debug_instances = Vec::new();

    for face_id in 0..8 {
        if is_face_visible(face_id, &state.rotation_4d, viewer_distance) {
            // Check if ray intersects face-level AABB
            let face_aabb = calculate_face_aabb(
                face_id,
                &state.rotation_4d,
                sticker_scale,
                gap_distance,
                gap_distance_4d,
                viewer_distance,
            );
            if ray_intersects_aabb(ray, &face_aabb) {
                intersectable_faces.push(face_id);

                // Create debug instance for face AABB only if enabled
                if let AABBMode::Face = aabb_mode {
                    let color = get_face_debug_color(face_id);
                    let min: [f32; 3] = face_aabb.min.coords.as_slice().try_into().unwrap();
                    let max: [f32; 3] = face_aabb.max.coords.as_slice().try_into().unwrap();
                    let debug_instance =
                        DebugInstanceWithDistance::new(min, max, color, camera_pos, 3.0);
                    debug_instances.push(debug_instance);
                }
            }
        }
    }

    let mut closest_distance = f32::INFINITY;
    let mut closest_sticker = None;

    // Only check stickers on faces that the ray could potentially hit
    for (sticker_index, sticker) in FACET_TABLE.iter().enumerate() {
        // Skip stickers on faces that ray doesn't intersect
        if !intersectable_faces.contains(&sticker.face_id) {
            continue;
        }

        let world_vertices = transform_sticker_vertices_to_3d(
            nalgebra::Vector4::from(sticker.position_4d),
            sticker.face_id,
            &state.rotation_4d,
            sticker_scale,
            gap_distance,
            gap_distance_4d,
            viewer_distance,
        );

        let sticker_aabb = calculate_sticker_aabb(&world_vertices);
        if ray_intersects_aabb(ray, &sticker_aabb) {
            // If showing sticker AABBs, create debug instance for this intersected sticker
            if let AABBMode::Sticker = aabb_mode {
                let color = [1.0, 1.0, 0.0, 0.4]; // Yellow with transparency for highlighted sticker
                let min: [f32; 3] = sticker_aabb.min.coords.as_slice().try_into().unwrap();
                let max: [f32; 3] = sticker_aabb.max.coords.as_slice().try_into().unwrap();
                let debug_instance =
                    DebugInstanceWithDistance::new(min, max, color, camera_pos, 3.0);
                debug_instances.push(debug_instance);
            }

            if let Some(distance) = ray_sticker_intersection(ray, &world_vertices)
                && distance < closest_distance
            {
                closest_distance = distance;
                closest_sticker = if sticker.is_actionable {
                    Some(sticker_index)
                } else {
                    // Don't highlight the center piece. No actions can be performed on it.
                    None
                };
            }
        }
    }

    // Sort debug instances back-to-front for proper transparency rendering
    debug_instances.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap());

    (closest_sticker, debug_instances)
}
