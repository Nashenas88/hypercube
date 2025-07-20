//! Ray casting for mouse-based sticker selection.
//!
//! This module provides CPU-based ray casting to detect which sticker
//! the mouse cursor is hovering over. Projects stickers from 4D to 3D space
//! and performs intersection testing in 3D.

use nalgebra::{Matrix4, Point3, Vector3, Vector4};
use iced::{Point, Rectangle};

use crate::camera::{Camera, Projection};
use crate::cube::FACE_CENTERS;

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
    let t1 = (aabb.min.x - ray.origin.x) * inv_dir.x;  // Distance to min X plane
    let t2 = (aabb.max.x - ray.origin.x) * inv_dir.x;  // Distance to max X plane
    
    // Y-axis slab: two planes at aabb.min.y and aabb.max.y
    let t3 = (aabb.min.y - ray.origin.y) * inv_dir.y;  // Distance to min Y plane
    let t4 = (aabb.max.y - ray.origin.y) * inv_dir.y;  // Distance to max Y plane
    
    // Z-axis slab: two planes at aabb.min.z and aabb.max.z
    let t5 = (aabb.min.z - ray.origin.z) * inv_dir.z;  // Distance to min Z plane
    let t6 = (aabb.max.z - ray.origin.z) * inv_dir.z;  // Distance to max Z plane

    // Find the farthest near intersection and nearest far intersection
    // tmin = where the ray ENTERS the 3D box (latest of all near intersections)
    // tmax = where the ray EXITS the 3D box (earliest of all far intersections)
    let tmin = t1.min(t2)        // Entry distance for X slab
                .max(t3.min(t4))    // Take latest entry (X or Y)
                .max(t5.min(t6));   // Take latest entry (X, Y, or Z)
    
    let tmax = t1.max(t2)        // Exit distance for X slab
                .min(t3.max(t4))    // Take earliest exit (X or Y)
                .min(t5.max(t6));   // Take earliest exit (X, Y, or Z)

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

/// Transform a 4D sticker position to 3D world space
/// Replicates the shader's transformation logic on CPU
pub(crate) fn transform_sticker_to_3d(
    sticker_position_4d: Vector4<f32>,
    face_id: usize,
    rotation_4d: &Matrix4<f32>,
    face_spacing: f32,
    viewer_distance: f32,
) -> Point3<f32> {
    // Get face information
    let face_center_4d = FACE_CENTERS[face_id];
    
    // Calculate sticker center in 4D (matching shader logic)
    let sticker_offset_4d = sticker_position_4d - face_center_4d;
    let scaled_face_center = face_center_4d * face_spacing;
    let sticker_center_4d = scaled_face_center + sticker_offset_4d;

    // Apply 4D rotation
    let rotated_4d = rotation_4d * sticker_center_4d;

    // Project to 3D (matching shader logic)
    let w_distance = viewer_distance - rotated_4d.w;
    let scale = viewer_distance / w_distance;
    
    Point3::new(
        rotated_4d.x * scale,
        rotated_4d.y * scale,
        rotated_4d.z * scale,
    )
}

/// Check if a 4D face is visible from the viewer position
/// Replicates the shader's face culling logic
pub(crate) fn is_face_visible(
    face_id: usize,
    rotation_4d: &Matrix4<f32>,
    viewer_distance: f32,
) -> bool {
    let face_center_4d = FACE_CENTERS[face_id];
    let rotated_face_center = rotation_4d * face_center_4d;
    let viewer_position = Vector4::new(0.0, 0.0, 0.0, viewer_distance);
    let to_viewer = viewer_position - rotated_face_center;
    let dot_product = rotated_face_center.dot(&to_viewer);
    dot_product < 0.0
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
    use crate::cube::{CUBE_VERTICES, FACE_CENTERS, FIXED_DIMS};
    
    let mut closest_distance = f32::INFINITY;
    let mut hit = false;
    
    // Get face information
    let face_center_4d = FACE_CENTERS[face_id];
    let fixed_dim = FIXED_DIMS[face_id];
    
    // Calculate sticker center in 4D (matching shader logic exactly)
    let sticker_offset_4d = sticker_position_4d - face_center_4d;
    let scaled_face_center = face_center_4d * face_spacing;
    let sticker_center_4d = scaled_face_center + sticker_offset_4d;
    
    // Transform each cube vertex exactly like the shader does
    let mut world_vertices = Vec::with_capacity(36);
    for vertex in &CUBE_VERTICES {
        let local_vertex = Vector3::new(vertex[0], vertex[1], vertex[2]) * sticker_scale;
        
        // Generate vertex in 4D space around sticker center (matching shader logic)
        let mut vertex_4d = sticker_center_4d;
        let mut offset_idx = 0;
        
        for axis in 0..4 {
            if axis != fixed_dim {
                match offset_idx {
                    0 => vertex_4d[axis] += local_vertex.x,
                    1 => vertex_4d[axis] += local_vertex.y,
                    2 => vertex_4d[axis] += local_vertex.z,
                    _ => {}
                }
                offset_idx += 1;
            }
        }
        
        // Apply 4D rotation
        let rotated_vertex_4d = rotation_4d * vertex_4d;
        
        // Project to 3D (matching shader logic)
        let w_distance = viewer_distance - rotated_vertex_4d.w;
        let scale = viewer_distance / w_distance;
        let vertex_3d = Point3::new(
            rotated_vertex_4d.x * scale,
            rotated_vertex_4d.y * scale,
            rotated_vertex_4d.z * scale,
        );
        
        world_vertices.push(vertex_3d);
    }
    
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
    
    if hit {
        Some(closest_distance)
    } else {
        None
    }
}

/// Test ray intersection with a triangle using Möller-Trumbore algorithm
/// Returns Some(distance) if ray intersects the triangle
fn ray_triangle_intersection(
    ray: &Ray,
    v0: Point3<f32>,
    v1: Point3<f32>, 
    v2: Point3<f32>
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
    
    // Transform face center to 3D
    let rotated_4d = rotation_4d * scaled_face_center;
    let w_distance = viewer_distance - rotated_4d.w;
    let scale = viewer_distance / w_distance;
    let face_center_3d = Point3::new(
        rotated_4d.x * scale,
        rotated_4d.y * scale,
        rotated_4d.z * scale,
    );
    
    // Face size encompasses 3x3x3 grid with spacing of 2/3 units
    // Total face size is about 4.0 units (from -2/3*2 to +2/3*2 with some margin)
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
            let face_aabb = calculate_face_aabb(face_id, rotation_4d, face_spacing, viewer_distance);
            if ray_aabb_intersection(ray, &face_aabb).is_some() {
                intersectable_faces.push(face_id);
            }
        }
    }
    
    let mut closest_distance = f32::INFINITY;
    let mut closest_sticker = None;

    // Only check stickers on faces that the ray could potentially hit
    for (sticker_index, (&sticker_position_4d, &face_id)) in 
        sticker_positions.iter().zip(face_ids.iter()).enumerate() {
        
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