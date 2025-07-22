//! Ray casting for mouse-based sticker selection.
//!
//! This module provides CPU-based ray casting to detect which sticker
//! the mouse cursor is hovering over. Projects stickers from 4D to 3D space
//! and performs intersection testing in 3D.

use iced::{Point, Rectangle};
use nalgebra::{Matrix4, Point3, Vector3, Vector4};

use crate::AABBMode;
use crate::camera::{Camera, Projection};
use crate::cube::NORMAL_TO_BASE_INDICES;
use crate::math::{
    calc_sticker_center, is_face_visible, project_cube_point, transform_sticker_vertices_to_3d,
};
use crate::renderer::DebugInstanceWithDistance;

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
        inverse_direction: direction.map(|i| 1.0 / i),
    }
}

/// Test ray intersection with 3D axis-aligned bounding box using the slab method
///
/// Returns Some(distance) if the ray intersects the box, None otherwise.
/// Uses the standard 3D slab method for ray-AABB intersection.
pub(crate) fn ray_intersects_aabb(ray: &Ray, aabb: &AABB) -> bool {
    // Calculate intersection distances with each pair of parallel planes
    // X-axis slab: two planes at aabb.min.x and aabb.max.x
    let t1 = (aabb.min.x - ray.origin.x) * ray.inverse_direction.x; // Distance to min X plane
    let t2 = (aabb.max.x - ray.origin.x) * ray.inverse_direction.x; // Distance to max X plane

    // Y-axis slab: two planes at aabb.min.y and aabb.max.y
    let t3 = (aabb.min.y - ray.origin.y) * ray.inverse_direction.y; // Distance to min Y plane
    let t4 = (aabb.max.y - ray.origin.y) * ray.inverse_direction.y; // Distance to max Y plane

    // Z-axis slab: two planes at aabb.min.z and aabb.max.z
    let t5 = (aabb.min.z - ray.origin.z) * ray.inverse_direction.z; // Distance to min Z plane
    let t6 = (aabb.max.z - ray.origin.z) * ray.inverse_direction.z; // Distance to max Z plane

    // Find the farthest near intersection and nearest far intersection
    // tmin = where the ray ENTERS the 3D box (latest of all near intersections)
    // tmax = where the ray EXITS the 3D box (earliest of all far intersections)
    let tmin = f32::max(
        f32::max(f32::min(t1, t2), f32::min(t3, t4)),
        f32::min(t5, t6),
    );
    let tmax = f32::min(
        f32::min(f32::max(t1, t2), f32::max(t3, t4)),
        f32::max(t5, t6),
    );

    // Check for intersection conditions:
    // 1. tmax < 0: The box is entirely behind the ray (no intersection)
    // 2. tmin > tmax: The ray misses the box (exits before entering)
    !(tmax < 0.0 || tmin > tmax)
}

/// Test ray intersection with actual sticker geometry using transformed vertices
/// Returns Some(distance) if ray intersects any triangle of the sticker
fn ray_sticker_intersection(ray: &Ray, world_vertices: &[Point3<f32>]) -> Option<f32> {
    let mut closest_distance = f32::INFINITY;
    let mut hit = false;

    // Test ray against each triangle (36 vertices = 12 triangles)
    for triangle_vertices in NORMAL_TO_BASE_INDICES.chunks(3) {
        let v0 = world_vertices[triangle_vertices[0]];
        let v0 = Point3::new(v0[0], v0[1], v0[2]);
        let v1 = world_vertices[triangle_vertices[1]];
        let v1 = Point3::new(v1[0], v1[1], v1[2]);
        let v2 = world_vertices[triangle_vertices[2]];
        let v2 = Point3::new(v2[0], v2[1], v2[2]);

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
    if !(0.0..=1.0).contains(&u) {
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

/// Calculate sticker-level AABB using actual transformed vertices
fn calculate_sticker_aabb(world_vertices: &[Point3<f32>]) -> AABB {
    // Find min and max bounds from all transformed vertices
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
    face_spacing: f32,
    viewer_distance: f32,
) -> AABB {
    use crate::cube::{BASE_CUBE_VERTICES, FACE_CENTERS, FIXED_DIMS};

    // Get face center and orientation info
    let face_center_4d = FACE_CENTERS[face_id];
    let scaled_face_center = face_center_4d * face_spacing;
    let fixed_dim = FIXED_DIMS[face_id];

    // Transform the 8 corner vertices of BASE_CUBE_VERTICES to match this face
    // We need to find the bounds that encompass all possible stickers on this face
    let mut transformed_corners_3d = Vec::with_capacity(8);

    // The face extends across the full 3x3x3 sticker grid plus sticker size
    // Sticker grid positions: -2/3, 0, +2/3 (range of 4/3)
    // BASE_CUBE_VERTICES are scaled by 1/3 in renderer.rs:518, then by sticker_scale in shaders
    // Plus add the grid extent to cover all stickers on the face
    let base_cube_size = 1.0 / 3.0; // Match renderer.rs scaling
    let actual_sticker_size = base_cube_size * sticker_scale; // Apply UI sticker scale
    let grid_extent = 2.0 / 3.0; // Half-width of 3x3x3 sticker grid  
    let face_bound = actual_sticker_size + grid_extent; // Total face extent

    for &base_vertex in &BASE_CUBE_VERTICES {
        // Use project_cube_point exactly like shader_widget does, but with face bounds
        let local_vertex =
            Vector3::new(base_vertex[0], base_vertex[1], base_vertex[2]) * face_bound;
        let corner_3d = project_cube_point(
            local_vertex,
            scaled_face_center,
            fixed_dim,
            rotation_4d,
            viewer_distance,
        );
        transformed_corners_3d.push(corner_3d);
    }

    // Find min and max bounds from all transformed corners
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
    sticker_positions: &[Vector4<f32>],
    face_ids: &[usize],
    rotation_4d: &Matrix4<f32>,
    sticker_scale: f32,
    face_spacing: f32,
    viewer_distance: f32,
    camera: &Camera,
    aabb_mode: AABBMode,
) -> (Option<usize>, Vec<DebugInstanceWithDistance>) {
    let camera_pos = [camera.eye.x, camera.eye.y, camera.eye.z];

    // First, determine which faces are visible and ray-intersectable
    let mut intersectable_faces = Vec::new();
    let mut debug_instances = Vec::new();

    for face_id in 0..8 {
        if is_face_visible(face_id, rotation_4d, viewer_distance) {
            // Check if ray intersects face-level AABB
            let face_aabb = calculate_face_aabb(
                face_id,
                rotation_4d,
                sticker_scale,
                face_spacing,
                viewer_distance,
            );
            if ray_intersects_aabb(ray, &face_aabb) {
                log::info!("Ray hit face {face_id}");
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
    for (sticker_index, (&sticker_position_4d, &face_id)) in
        sticker_positions.iter().zip(face_ids.iter()).enumerate()
    {
        // Skip stickers on faces that ray doesn't intersect
        if !intersectable_faces.contains(&face_id) {
            continue;
        }

        // Transform sticker to 4D world space for AABB calculation
        let sticker_center_4d = calc_sticker_center(sticker_position_4d, face_id, face_spacing);

        // Use shared transformation logic from math.rs
        let world_vertices = transform_sticker_vertices_to_3d(
            sticker_center_4d,
            face_id,
            rotation_4d,
            sticker_scale,
            viewer_distance,
        );

        // First check: AABB intersection using properly scaled vertices
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

            // Second check: Actual sticker geometry intersection (accurate)
            if let Some(distance) = ray_sticker_intersection(ray, &world_vertices) {
                if distance < closest_distance {
                    closest_distance = distance;
                    closest_sticker = Some(sticker_index);
                }
            }
        }
    }

    // Sort debug instances back-to-front for proper transparency rendering
    debug_instances.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap());

    (closest_sticker, debug_instances)
}
