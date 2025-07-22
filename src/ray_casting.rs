//! Ray casting for mouse-based sticker selection.
//!
//! This module provides CPU-based ray casting to detect which sticker
//! the mouse cursor is hovering over. Projects stickers from 4D to 3D space
//! and performs intersection testing in 3D.

use iced::{Point, Rectangle};
use nalgebra::{Matrix4, Point3, Vector3, Vector4};

use crate::camera::{Camera, Projection};
use crate::cube::FACE_CENTERS;
use crate::math::{
    is_face_visible, project_4d_to_3d, transform_sticker_to_3d, transform_sticker_vertices_to_3d,
};

/// 3D ray for intersection testing
#[derive(Debug, Clone)]
pub(crate) struct Ray {
    /// Ray origin point in 3D space
    pub(crate) origin: Point3<f32>,
    /// Ray direction vector (normalized)
    pub(crate) direction: Vector3<f32>,
}

/// Axis-aligned bounding box in 3D space
#[derive(Debug, Clone)]
pub(crate) struct AABB {
    /// Minimum corner of the 3D bounding box
    pub(crate) min: Point3<f32>,
    /// Maximum corner of the 3D bounding box
    pub(crate) max: Point3<f32>,
}

impl AABB {
    /// Create a 3D AABB centered at a point with given size
    pub(crate) fn from_center_size(center: Point3<f32>, size: f32) -> Self {
        let half_size = size * 0.5;
        Self {
            min: Point3::new(
                center.x - half_size,
                center.y - half_size,
                center.z - half_size,
            ),
            max: Point3::new(
                center.x + half_size,
                center.y + half_size,
                center.z + half_size,
            ),
        }
    }
}

/// Calculate mouse ray from screen coordinates through the 3D scene
pub(crate) fn calculate_mouse_ray(
    mouse_pos: Point,
    bounds: Rectangle,
    camera: &Camera,
    projection: &Projection,
) -> Ray {
    // Convert mouse position to normalized device coordinates (-1 to 1)
    let ndc_x = (2.0 * mouse_pos.x / bounds.width) - 1.0;
    let ndc_y = 1.0 - (2.0 * mouse_pos.y / bounds.height);

    // Build camera matrices
    let view_matrix = camera.build_view_matrix();
    let proj_matrix = projection.build_projection_matrix();
    let view_proj_matrix = proj_matrix * view_matrix;

    // Inverse transform to get ray in world space
    let inv_view_proj = view_proj_matrix
        .try_inverse()
        .expect("View-projection matrix should be invertible");

    // Calculate ray points in world space
    let ray_start_ndc = Vector4::new(ndc_x, ndc_y, -1.0, 1.0);
    let ray_end_ndc = Vector4::new(ndc_x, ndc_y, 1.0, 1.0);

    let ray_start_world = inv_view_proj * ray_start_ndc;
    let ray_end_world = inv_view_proj * ray_end_ndc;

    // Convert from homogeneous coordinates
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

    // Calculate ray direction
    let direction = (ray_end - ray_start).normalize();

    Ray {
        origin: ray_start,
        direction,
    }
}

/// Test ray intersection with 3D axis-aligned bounding box using the slab method
///
/// Returns Some(distance) if the ray intersects the box, None otherwise.
/// Uses the standard 3D slab method for ray-AABB intersection.
pub(crate) fn ray_aabb_intersection(ray: &Ray, aabb: &AABB) -> Option<f32> {
    // Pre-compute inverse ray direction to avoid division in the loop
    let inv_dir = Vector3::new(
        1.0 / ray.direction.x,
        1.0 / ray.direction.y,
        1.0 / ray.direction.z,
    );

    // Calculate intersection distances with each pair of parallel planes
    // X-axis slab: two planes at aabb.min.x and aabb.max.x
    let t1 = (aabb.min.x - ray.origin.x) * inv_dir.x; // Distance to min X plane
    let t2 = (aabb.max.x - ray.origin.x) * inv_dir.x; // Distance to max X plane

    // Y-axis slab: two planes at aabb.min.y and aabb.max.y
    let t3 = (aabb.min.y - ray.origin.y) * inv_dir.y; // Distance to min Y plane
    let t4 = (aabb.max.y - ray.origin.y) * inv_dir.y; // Distance to max Y plane

    // Z-axis slab: two planes at aabb.min.z and aabb.max.z
    let t5 = (aabb.min.z - ray.origin.z) * inv_dir.z; // Distance to min Z plane
    let t6 = (aabb.max.z - ray.origin.z) * inv_dir.z; // Distance to max Z plane

    // Find the farthest near intersection and nearest far intersection
    // tmin = where the ray ENTERS the 3D box (latest of all near intersections)
    // tmax = where the ray EXITS the 3D box (earliest of all far intersections)
    let tmin = t1
        .min(t2) // Entry distance for X slab
        .max(t3.min(t4)) // Take latest entry (X or Y)
        .max(t5.min(t6)); // Take latest entry (X, Y, or Z)

    let tmax = t1
        .max(t2) // Exit distance for X slab
        .min(t3.max(t4)) // Take earliest exit (X or Y)
        .min(t5.max(t6)); // Take earliest exit (X, Y, or Z)

    // Check for intersection conditions:
    // 1. tmax < 0: The box is entirely behind the ray (no intersection)
    // 2. tmin > tmax: The ray misses the box (exits before entering)
    if tmax < 0.0 || tmin > tmax {
        None
    } else {
        // Ray intersects the 3D box!
        // Use the closest valid intersection point:
        // - If tmin >= 0: Ray starts outside box, use entry point (tmin)
        // - If tmin < 0: Ray starts inside box, use exit point (tmax)
        let distance = if tmin >= 0.0 { tmin } else { tmax };
        Some(distance)
    }
}

/// Test ray intersection with actual sticker geometry using transformed vertices
/// Returns Some(distance) if ray intersects any triangle of the sticker
fn ray_sticker_intersection(
    ray: &Ray,
    sticker_position_4d: Vector4<f32>,
    face_id: usize,
    rotation_4d: &Matrix4<f32>,
    sticker_scale: f32,
    face_spacing: f32,
    viewer_distance: f32,
) -> Option<f32> {
    let mut closest_distance = f32::INFINITY;
    let mut hit = false;

    // Use shared transformation logic from math.rs
    let world_vertices = transform_sticker_vertices_to_3d(
        sticker_position_4d,
        face_id,
        rotation_4d,
        sticker_scale,
        face_spacing,
        viewer_distance,
    );

    // Test ray against each triangle (36 vertices = 12 triangles)
    for triangle_vertices in world_vertices.chunks(3) {
        let v0 = triangle_vertices[0];
        let v1 = triangle_vertices[1];
        let v2 = triangle_vertices[2];

        if let Some(distance) = ray_triangle_intersection(ray, v0, v1, v2) {
            if distance < closest_distance {
                closest_distance = distance;
                hit = true;
            }
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

    // Calculate triangle edges from v0
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;

    // Calculate determinant using cross product of ray direction and edge2
    let h = ray.direction.cross(&edge2);
    let a = edge1.dot(&h);

    // If determinant is near zero, ray is parallel to triangle
    if a > -EPSILON && a < EPSILON {
        return None;
    }

    let f = 1.0 / a;

    // Calculate vector from v0 to ray origin
    let s = ray.origin - v0;

    // Calculate u parameter and test bounds
    let u = f * s.dot(&h);
    if u < 0.0 || u > 1.0 {
        return None; // Intersection point is outside triangle
    }

    // Calculate v parameter using cross product
    let q = s.cross(&edge1);
    let v = f * ray.direction.dot(&q);

    // Test remaining triangle bounds
    if v < 0.0 || u + v > 1.0 {
        return None; // Intersection point is outside triangle
    }

    // Calculate distance along ray to intersection point
    let t = f * edge2.dot(&q);

    // Return distance if intersection is in front of ray origin
    if t > EPSILON {
        Some(t)
    } else {
        None // Intersection is behind ray origin
    }
}

/// Calculate face-level AABB that encompasses all stickers on a face
fn calculate_face_aabb(
    face_id: usize,
    rotation_4d: &Matrix4<f32>,
    face_spacing: f32,
    viewer_distance: f32,
) -> AABB {
    // Get face center in 4D
    let face_center_4d = FACE_CENTERS[face_id];
    let scaled_face_center = face_center_4d * face_spacing;

    // Use shared transformation logic from math.rs
    let face_center_3d = project_4d_to_3d(scaled_face_center, rotation_4d, viewer_distance);

    let face_size = 3.0;

    AABB::from_center_size(face_center_3d, face_size)
}

/// Find the sticker that the 3D mouse ray intersects
/// Returns the sticker index if found, None otherwise
pub(crate) fn find_intersected_sticker(
    ray: &Ray,
    sticker_positions: &[Vector4<f32>],
    face_ids: &[usize],
    rotation_4d: &Matrix4<f32>,
    sticker_scale: f32,
    face_spacing: f32,
    viewer_distance: f32,
) -> Option<usize> {
    // First, determine which faces are visible and ray-intersectable
    let mut intersectable_faces = Vec::new();
    for face_id in 0..8 {
        if is_face_visible(face_id, rotation_4d, viewer_distance) {
            // Check if ray intersects face-level AABB
            let face_aabb =
                calculate_face_aabb(face_id, rotation_4d, face_spacing, viewer_distance);
            if ray_aabb_intersection(ray, &face_aabb).is_some() {
                intersectable_faces.push(face_id);
            }
        }
    }

    let mut closest_distance = f32::INFINITY;
    let mut closest_sticker = None;

    // Only check stickers on faces that the ray could potentially hit
    for (sticker_index, (&sticker_position_4d, &face_id)) in
        sticker_positions.iter().zip(face_ids.iter()).enumerate()
    {
        // Skip stickers on faces that ray doesn't intersect
        if !intersectable_faces.contains(&face_id) {
            continue;
        }

        // Transform sticker to 3D world space for AABB check
        let sticker_center_3d = transform_sticker_to_3d(
            sticker_position_4d,
            face_id,
            rotation_4d,
            face_spacing,
            viewer_distance,
        );

        // First check: AABB intersection (fast rejection)
        let sticker_aabb = AABB::from_center_size(sticker_center_3d, sticker_scale);
        if let Some(_aabb_distance) = ray_aabb_intersection(ray, &sticker_aabb) {
            // Second check: Actual sticker geometry intersection (accurate)
            if let Some(distance) = ray_sticker_intersection(
                ray,
                sticker_position_4d,
                face_id,
                rotation_4d,
                sticker_scale,
                face_spacing,
                viewer_distance,
            ) {
                if distance < closest_distance {
                    closest_distance = distance;
                    closest_sticker = Some(sticker_index);
                }
            }
        }
    }

    closest_sticker
}
