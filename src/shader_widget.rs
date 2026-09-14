//! Custom shader widget for 4D hypercube rendering.
//!
//! This module implements the shader widget that encapsulates all 3D rendering
//! logic, camera controls, and 4D transformations. It follows Option C architecture
//! where the shader widget manages its own state independently.

use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use iced::wgpu;
use iced::widget::{Action, shader};
use iced::{Event, Point, Rectangle, event, mouse};
use nalgebra::{Matrix4, Point3, UnitQuaternion, Vector3, Vector4};

use crate::animation::ease;
use crate::app::{AABBMode, Message, RenderMode, SolveOutcome};
use crate::camera::{Camera, CameraController, Projection};
use crate::geometry::{
    BASE_CUBE_VERTICES, FACE_CENTERS, FIXED_DIMS, NORMAL_TO_BASE_INDICES, VERTEX_NORMAL_INDICES,
};
use crate::math::{
    GRID_EXTENT, VIEWER_DISTANCE, compose_so4, create_4d_plane_rotation, decompose_so4,
    mouse_delta_to_plane_angles, orthogonal_complement_plane, process_4d_rotation,
    project_4d_to_3d, project_cube_point, project_face_point, quat_slerp_exact, shortest_arc_plane,
    transform_sticker_vertices_to_3d, visible_faces,
};
use crate::moves::{base_angle, clockwise_sign, rotate_local_position};
use crate::piece::{
    FACET_TABLE, Hypercube, Piece, StickerInstance, free_axes, generate_sticker_instances,
};
use crate::ray_casting::{
    AABB, Ray, calculate_mouse_ray, calculate_sticker_aabb, find_intersected_sticker,
    ray_intersects_aabb, ray_sticker_intersection,
};
use crate::renderer::{DebugInstanceWithDistance, GizmoVertex, Renderer};
use crate::settings::RotateButton;
use crate::snapshot::{self, ViewSnapshot};
use crate::solver::{self, SolveStep};
use crate::theme::{
    ELEMENTAL_DIRT_KIND, ELEMENTAL_FIRE_KIND, ELEMENTAL_ICE_KIND, ELEMENTAL_LIGHT_KIND, Theme,
};

/// An in-progress move's animation: piece state has already been committed
/// atomically by `apply_move`; this only drives the visual sweep from the
/// pre-move snapshot toward the (already-final) post-move positions.
struct AnimatingMove {
    side_axis: usize,
    side_sign: i8,
    local_coords: [i8; 3],
    /// Signed target angle (its sign already encodes direction).
    angle: f32,
    pre_move_pieces: Vec<Piece>,
    elapsed: Duration,
    duration: Duration,
}

/// Outcome of advancing the move animation by one tick.
enum AnimationTick {
    /// No animation was in progress this tick.
    Ignored,
    /// Animation advanced but is still running.
    Running,
    /// Animation just crossed its duration threshold this tick.
    Completed,
}

/// A pair of orthogonal 4D vectors spanning a rotation plane (or, for the
/// rotation-axis gizmo, an invariant plane - see `orthogonal_complement_plane`).
type GizmoPlane = (Vector4<f32>, Vector4<f32>);

/// An in-progress "center this face" animation, triggered by double-clicking
/// a sticker: sweeps `rotation_4d` from its value when the double-click
/// landed toward `start_rotation` rotated by `total_angle` in `plane`, which
/// by construction carries the double-clicked face's normal onto the
/// screen-centered pole (see `shortest_arc_plane` and its call site below).
struct AnimatingFocus {
    start_rotation: Matrix4<f32>,
    plane: GizmoPlane,
    total_angle: f32,
    elapsed: Duration,
    duration: Duration,
}

/// An in-progress "return to default orientation" animation, triggered by
/// Reset: slerps the isoclinic quaternion pair (see `math::decompose_so4`)
/// describing `rotation_4d` when Reset was pressed toward the identity
/// quaternion pair, recomposing `rotation_4d` each tick. Unlike
/// `AnimatingFocus`, which rotates in a single plane, this can undo an
/// arbitrary accumulated 4D orientation - generally a "double rotation"
/// with two independent invariant planes. The rotation-axis gizmo still
/// shows one ring for it (see `reset_plane_and_phase`), reusing the same
/// combination pattern as the Shift+drag gizmo; see that function's doc
/// comment for when this is exact versus approximate.
struct AnimatingReset {
    start_p: UnitQuaternion<f32>,
    start_q: UnitQuaternion<f32>,
    elapsed: Duration,
    duration: Duration,
}

/// Tracks a live Shift+drag 4D rotation gesture, for the rotation-axis
/// gizmo: the *signed* angle applied by each of `process_4d_rotation`'s two
/// independent component rotations (horizontal: camera-right/W plane;
/// vertical: camera-up/W plane), accumulated since the drag began. Created
/// lazily on the first Shift+drag mouse-move of a gesture and cleared
/// whenever the gesture ends (button release, Shift released, or a reset
/// animation taking over `rotation_4d`), so the gizmo only ever reflects an
/// actively-in-progress drag.
#[derive(Debug, Clone, Copy, Default)]
struct ActiveShiftDrag {
    horizontal_angle: f32,
    vertical_angle: f32,
}

/// An in-progress reveal/hide flourish: sweeps sticker scale, face gap, 4D
/// face gap, and camera yaw from their values when the toggle button was
/// pressed toward the reveal's (or hide's) target values, all driven by the
/// same `elapsed`/`duration`/`ease` progress.
struct AnimatingReveal {
    start_scale: f32,
    target_scale: f32,
    start_gap: f32,
    target_gap: f32,
    start_gap_4d: f32,
    target_gap_4d: f32,
    start_yaw: f32,
    target_yaw: f32,
    elapsed: Duration,
    duration: Duration,
}

/// What `HypercubeApp` last asked of solve playback, carried alongside
/// `solve_command_generation` since a bare generation bump carries no
/// payload (mirrors `random_move_count` alongside `random_moves_generation`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SolveCommand {
    /// Solve the live puzzle and start playing the solution back.
    Start,
    /// Abandon playback; the move in flight still finishes animating.
    Stop,
}

/// A solution being played back one move at a time, each starting as soon
/// as the previous one finishes animating.
struct SolvePlayback {
    /// The `solve_command_generation` that started it, echoed in every
    /// message so `HypercubeApp` can drop any from a cancelled run.
    generation: u64,
    queue: VecDeque<SolveStep>,
    total: usize,
}

/// Max cursor movement between a rotate-button press and release for it to
/// still count as a click rather than a drag.
const CLICK_DRAG_THRESHOLD_PX: f32 = 4.0;
/// Max gap between two qualifying clicks on the same face for them to count
/// as a double-click.
const DOUBLE_CLICK_WINDOW: Duration = Duration::from_millis(400);

/// Duration of the reveal/hide flourish (camera spin + scale/gap animation),
/// independent of `animation_duration_ms` which is tuned for quick move/focus
/// animations rather than a two-revolution camera spin.
pub(crate) const REVEAL_ANIMATION_DURATION: Duration = Duration::from_millis(2500);
/// Duration of each move while a solve plays back - a fixed fast pace,
/// independent of `animation_duration_ms`, since a solution runs to over a
/// thousand moves.
pub(crate) const SOLVE_MOVE_DURATION: Duration = Duration::from_millis(40);
/// Camera yaw delta applied by a single reveal or hide flourish. A multiple
/// of 360 degrees, so the camera always visually ends up where it started.
const REVEAL_YAW_SPIN_DEGREES: f32 = 720.0;
/// Sticker scale/face gap in the app's raw (slider) domain before a reveal.
/// Shared with `app.rs::HypercubeApp::new()` as the single source of truth.
pub(crate) const PRIMARY_STICKER_SCALE: f32 = 0.02;
pub(crate) const PRIMARY_FACE_GAP: f32 = 0.0;
/// Sticker scale/face gap in the app's raw (slider) domain a reveal animates
/// toward.
pub(crate) const SECONDARY_STICKER_SCALE: f32 = 0.4;
pub(crate) const SECONDARY_FACE_GAP: f32 = 0.45;
/// 4D face gap in the app's raw (slider) domain before a reveal.
pub(crate) const PRIMARY_FACE_GAP_4D: f32 = 1.0;
/// 4D face gap in the app's raw (slider) domain a reveal animates toward.
pub(crate) const SECONDARY_FACE_GAP_4D: f32 = 2.0;

/// Number of angular samples around the rotation-axis gizmo's main torus.
const GIZMO_RING_SEGMENTS: usize = 48;
/// Minor radius of the main ring's tube cross-section - giving the ring
/// real 3D thickness is what keeps it from ever collapsing to a zero-width
/// line when its invariant plane happens to include the camera's own
/// viewing axis (see this module's rotation-axis-gizmo doc comment).
/// Calibrated for the ring's default radius of `1.0` (see
/// `gizmo_ring_radius`); scales with it only by eye, not automatically.
const GIZMO_RING_TUBE_MINOR_RADIUS: f32 = 0.03;
/// Number of angular samples around the main ring's tube cross-section.
const GIZMO_RING_TUBE_SEGMENTS: usize = 8;
/// Number of alternating-hue color bands running around the main ring.
const GIZMO_BAND_COUNT: usize = 4;
/// Number of small "field loop" markers threaded around the ring like the
/// loops of a magnetic field around a current-carrying wire, fixed
/// equidistant positions along the ring - unlike the arrows drawn on each
/// one, which creep around that marker's *own* circumference as rotation
/// progresses (see `gizmo_torus_vertices`'s doc comment).
const GIZMO_MARKER_COUNT: usize = 4;
/// Radius of a field-loop marker's centerline circle.
const GIZMO_MARKER_RADIUS: f32 = 0.375;
/// Number of angular samples around each field-loop marker's centerline.
const GIZMO_MARKER_SEGMENTS: usize = 12;
/// Number of angular samples around a marker tube's cross-section.
const GIZMO_MARKER_TUBE_SEGMENTS: usize = 6;
/// Minor radius of a marker's tube cross-section.
const GIZMO_MARKER_TUBE_MINOR_RADIUS: f32 = 0.0375;
/// Number of small arrow marks evenly spaced around each field-loop
/// marker's own circumference.
const GIZMO_MARKER_ARROW_COUNT: usize = 3;
/// Length of a marker arrow's cone, from its base to its tip.
const GIZMO_ARROW_LENGTH: f32 = 0.1875;
/// How far behind the arrow's anchor point its base sits, giving the cone
/// visible depth rather than a flat fan.
const GIZMO_ARROW_BACK_OFFSET: f32 = 0.09375;
/// Radius of an arrow cone's circular base.
const GIZMO_ARROW_BASE_RADIUS: f32 = 0.078125;
/// Small angular offset used to numerically estimate the main ring's
/// tangent direction (in already-projected 3D space) at a marker's position.
const GIZMO_TANGENT_EPSILON: f32 = 0.01;
/// Angular offset applied to every major-angle sample of the main ring and
/// its markers, so neither lands on `0, π/2, π, 3π/2` - which
/// `orthogonal_complement_plane`'s standard-basis Gram-Schmidt tends to
/// align with other faces' own center stickers for an axis-aligned
/// rotation plane (any click-to-focus animation starting from the identity
/// orientation), hiding the ring/markers behind that opaque geometry.
const GIZMO_RING_PHASE_OFFSET: f32 = std::f32::consts::FRAC_PI_4;
/// Below this accumulated drag angle (radians), a Shift+drag's combined
/// ring is treated as not yet meaningfully rotating and is hidden. Also
/// reused by `reset_plane_and_phase` for the same purpose.
const GIZMO_MIN_DRAG_ANGLE: f32 = 1e-3;

/// Alternating-hue band palette for the click-to-focus gizmo ring. Fully
/// opaque so the main ring itself reads as solid rather than see-through.
const GIZMO_FOCUS_PALETTE: [[f32; 4]; GIZMO_BAND_COUNT] = [
    [0.3, 0.85, 1.0, 1.0],
    [0.1, 0.45, 0.9, 1.0],
    [0.3, 0.85, 1.0, 1.0],
    [0.1, 0.45, 0.9, 1.0],
];
/// Alternating-hue band palette for the Shift+drag gizmo ring. Fully opaque
/// so the main ring itself reads as solid rather than see-through.
const GIZMO_DRAG_PALETTE: [[f32; 4]; GIZMO_BAND_COUNT] = [
    [1.0, 0.55, 0.15, 1.0],
    [0.6, 0.4, 1.0, 1.0],
    [1.0, 0.55, 0.15, 1.0],
    [0.6, 0.4, 1.0, 1.0],
];
/// Flat color for the click-to-focus ring's field-loop markers, deliberately
/// distinct from both `GIZMO_FOCUS_PALETTE` hues (blue/cyan) so a marker's
/// own ring is easy to tell apart from the main ring threading through it.
const GIZMO_FOCUS_MARKER_COLOR: [f32; 4] = [1.0, 0.85, 0.15, 1.0];
/// Flat color for the Shift+drag ring's field-loop markers, deliberately
/// distinct from both `GIZMO_DRAG_PALETTE` hues (orange/purple) so a
/// marker's own ring is easy to tell apart from the main ring threading
/// through it.
const GIZMO_DRAG_MARKER_COLOR: [f32; 4] = [0.2, 1.0, 0.6, 1.0];
/// Alternating-hue band palette for the Reset gizmo ring. Fully opaque so
/// the main ring itself reads as solid rather than see-through. Distinct
/// from both `GIZMO_FOCUS_PALETTE` (blue/cyan) and `GIZMO_DRAG_PALETTE`
/// (orange/purple) so it's clear which interaction is driving the ring.
const GIZMO_RESET_PALETTE: [[f32; 4]; GIZMO_BAND_COUNT] = [
    [0.85, 0.25, 0.55, 1.0],
    [0.45, 0.85, 0.35, 1.0],
    [0.85, 0.25, 0.55, 1.0],
    [0.45, 0.85, 0.35, 1.0],
];
/// Flat color for the Reset ring's field-loop markers, deliberately
/// distinct from both `GIZMO_RESET_PALETTE` hues (magenta/green) so a
/// marker's own ring is easy to tell apart from the main ring threading
/// through it.
const GIZMO_RESET_MARKER_COLOR: [f32; 4] = [1.0, 1.0, 0.4, 1.0];
/// Shared bright accent color for marker arrows, distinct from either ring
/// palette or marker color so it reads clearly against the marker tube.
/// Fully opaque, like the rest of the gizmo, now that it writes real depth.
const GIZMO_ARROW_COLOR: [f32; 4] = [0.95, 0.95, 1.0, 1.0];

/// Component-wise linear interpolation between two RGBA colors, used to
/// make the main ring's band coloring a continuous function of rotation
/// progress (see `gizmo_torus_vertices`'s `band_color`) instead of a hard
/// step that pops between palette entries as `phase_angle` advances.
fn lerp_color(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}

/// Builds one `GizmoVertex`, converting a `Point3` to the plain `[f32; 3]`
/// the GPU buffer wants. `normal` starts zeroed and is filled in afterward
/// by `recompute_flat_normals`, once every triangle's three corners are
/// known - used only for the arrow cones' faceted geometry, where a flat
/// per-triangle normal is the desired look.
fn gizmo_vertex(p: Point3<f32>, color: [f32; 4]) -> GizmoVertex {
    GizmoVertex {
        position: [p.x, p.y, p.z],
        normal: [0.0; 3],
        color,
    }
}

/// Builds one `GizmoVertex` with an already-known, real per-vertex normal -
/// used for the main ring's and markers' tube surfaces, which are smooth
/// (not faceted) and so get a true smooth-shaded normal computed
/// analytically at generation time (see `gizmo_torus_vertices`'s
/// `ring_tube_normal`/marker equivalent) rather than the flat per-triangle
/// normal `recompute_flat_normals` derives for the arrow cones.
fn gizmo_vertex_with_normal(p: Point3<f32>, normal: Vector3<f32>, color: [f32; 4]) -> GizmoVertex {
    GizmoVertex {
        position: [p.x, p.y, p.z],
        normal: normal.into(),
        color,
    }
}

/// Fills in each triangle's flat face normal (the cross product of two of
/// its edges) across every consecutive triplet of `vertices`, which -
/// `gizmo_torus_vertices` only ever emitting `TriangleList` geometry with no
/// index buffer - are always exactly one triangle's three corners. Used only
/// for the arrow cones' faceted geometry: the ring and marker tubes instead
/// get a real smooth per-vertex normal computed analytically as they're
/// built (see `gizmo_torus_vertices`'s `ring_tube_normal` and its marker
/// equivalent), since they're true curved tube surfaces where flat shading
/// would read as faceted rather than round; a low-poly cone, by contrast,
/// is meant to look faceted.
fn recompute_flat_normals(vertices: &mut [GizmoVertex]) {
    for triangle in vertices.as_chunks_mut::<3>().0 {
        let p0 = Vector3::from(triangle[0].position);
        let p1 = Vector3::from(triangle[1].position);
        let p2 = Vector3::from(triangle[2].position);
        let raw_normal = (p1 - p0).cross(&(p2 - p0));
        let normal = if raw_normal.norm() > 1e-8 {
            raw_normal.normalize()
        } else {
            Vector3::z()
        };
        let normal: [f32; 3] = normal.into();
        for vertex in triangle {
            vertex.normal = normal;
        }
    }
}

/// Appends a two-triangle quad spanning corners `(p00, p01, p10, p11)`
/// (indexed by two independent parameters, e.g. a major/minor angle pair),
/// each corner carrying its own already-known smooth normal and color -
/// shared by the main ring's and markers' tube geometry, both true tube
/// surfaces where a real per-vertex normal (rather than a flat per-triangle
/// one) gives a smooth-shaded, non-faceted look.
#[allow(clippy::too_many_arguments)]
fn gizmo_push_quad(
    out: &mut Vec<GizmoVertex>,
    p00: Point3<f32>,
    n00: Vector3<f32>,
    c00: [f32; 4],
    p10: Point3<f32>,
    n10: Vector3<f32>,
    c10: [f32; 4],
    p01: Point3<f32>,
    n01: Vector3<f32>,
    c01: [f32; 4],
    p11: Point3<f32>,
    n11: Vector3<f32>,
    c11: [f32; 4],
) {
    out.push(gizmo_vertex_with_normal(p00, n00, c00));
    out.push(gizmo_vertex_with_normal(p10, n10, c10));
    out.push(gizmo_vertex_with_normal(p01, n01, c01));

    out.push(gizmo_vertex_with_normal(p01, n01, c01));
    out.push(gizmo_vertex_with_normal(p10, n10, c10));
    out.push(gizmo_vertex_with_normal(p11, n11, c11));
}

/// Appends the rotation-axis gizmo's main-ring-torus + field-loop-marker
/// geometry to `out`. The main ring is a genuine tube (not a flat ribbon)
/// swept around the unit circle in the invariant plane `(u, v)` - the 2D
/// subspace a 4D rotation in some *other* plane leaves fixed (see
/// `math::orthogonal_complement_plane`). Real 3D thickness is what keeps the
/// ring visible from *any* camera angle: a live Shift+drag's invariant plane
/// always contains the camera's own forward axis (a provable, unavoidable
/// fact of `process_4d_rotation`'s basis, not a bug), so a flat ribbon in
/// that plane collapses to a literal zero-width line - a tube instead
/// presents a visible cross-section from every angle, including squarely
/// edge-on.
///
/// `u`/`v` are always projected with an *identity* rotation, never the
/// puzzle's live `rotation_4d`: both callers derive them from vectors
/// already expressed in the same display-space frame `project_4d_to_3d`'s
/// perspective divide expects directly (`AnimatingFocus::plane`'s vectors
/// come from `state.rotation_4d`-multiplied face normals; a drag's come from
/// the camera's own basis) - see `math::project_4d_to_3d`'s "rotation_4d
/// maps *local* points into display space" contract. Multiplying by
/// `rotation_4d` again would double-apply whatever orientation the puzzle
/// had when the animation/drag began, so the ring would visibly drift/warp
/// as that *separate* rotation composes on top of the sweep this ring is
/// meant to hold still against - it would only look right by coincidence,
/// for whichever starting orientations happen to leave the invariant plane
/// unmoved by that extra multiplication.
///
/// The ring's own per-major-angle frame (`radial`/`binormal`, from which its
/// tube cross-section is built) is recomputed from scratch at every sampled
/// angle rather than propagated from the previous one, so the tube closes
/// with zero seam/twist automatically. `GIZMO_BAND_COUNT` alternating-hue
/// color bands run around the tube's *cross-section* (a function of the
/// minor angle and `phase_angle` alone, not the major angle), so the whole
/// tube reads as spinning in place about its own centerline in sync with
/// accumulated rotation progress, freezing the instant the rotation does -
/// never driven by wall-clock time, matching every other animation in this
/// feature.
///
/// `GIZMO_MARKER_COUNT` field-loop marker toruses - small tubes
/// perpendicular to the main ring's own tangent, like the loops of a
/// magnetic field around a current-carrying wire - sit at fixed, equidistant
/// positions along the ring (they do not travel around it) and carry a flat,
/// non-banded `marker_color` deliberately distinct from the main ring's own
/// band colors (see `GIZMO_FOCUS_MARKER_COLOR`/`GIZMO_DRAG_MARKER_COLOR`), so
/// it's clear at a glance which tube is which even where they cross. Each
/// carries `GIZMO_MARKER_ARROW_COUNT` small
/// solid 3D arrow cones that creep around *that marker's own* circumference
/// only as `phase_angle` advances, always pointing tangentially in the
/// actual direction of rotation (reversing if it reverses).
#[allow(clippy::too_many_arguments)]
fn gizmo_torus_vertices(
    center: Vector4<f32>,
    (u, v): GizmoPlane,
    phase_angle: f32,
    ring_radius: f32,
    palette: [[f32; 4]; GIZMO_BAND_COUNT],
    marker_color: [f32; 4],
    arrow_color: [f32; 4],
    viewer_distance: f32,
    out: &mut Vec<GizmoVertex>,
) {
    use std::f32::consts::TAU;

    let identity = Matrix4::identity();
    let project = |angle: f32, radius: f32| -> Point3<f32> {
        let point_4d = center + (u * angle.cos() + v * angle.sin()) * radius;
        project_4d_to_3d(point_4d, &identity, viewer_distance)
    };
    let ring_center_3d = project(0.0, 0.0);

    // The main ring's frame at major angle `a`: its projected centerline
    // point plus a unit `radial`/`binormal` pair spanning the plane
    // perpendicular to its own tangent there. Shared by the main ring's
    // tube and by each marker's outer (fixed) positioning, which is exactly
    // this same frame evaluated at that marker's own fixed angle.
    let ring_frame = |a: f32| -> Option<(Point3<f32>, Vector3<f32>, Vector3<f32>)> {
        let a = a + GIZMO_RING_PHASE_OFFSET;
        let center_3d = project(a, ring_radius);
        let ahead_3d = project(a + GIZMO_TANGENT_EPSILON, ring_radius);
        let tangent = ahead_3d - center_3d;
        let radial = center_3d - ring_center_3d;
        let binormal = tangent.cross(&radial);
        // Degenerate near the projection's own singularity (radial or
        // tangent collapsing to zero) - skip rather than divide by zero.
        if binormal.norm() < 1e-5 || radial.norm() < 1e-5 {
            return None;
        }
        Some((center_3d, radial.normalize(), binormal.normalize()))
    };
    // The unit radial direction from the tube's centerline to the surface
    // point at minor angle `beta` (`radial`/`binormal` are already an
    // orthonormal pair, so this needs no further normalizing) - used to
    // place the point itself.
    let ring_tube_offset = |(_, radial, binormal): (Point3<f32>, Vector3<f32>, Vector3<f32>),
                            beta: f32|
     -> Vector3<f32> { radial * beta.cos() + binormal * beta.sin() };
    let ring_tube_point =
        |frame: (Point3<f32>, Vector3<f32>, Vector3<f32>), beta: f32| -> Point3<f32> {
            frame.0 + ring_tube_offset(frame, beta) * GIZMO_RING_TUBE_MINOR_RADIUS
        };
    // The tube surface's true shading normal at that same point - the
    // *negation* of `ring_tube_offset`, not `ring_tube_offset` itself.
    // `gizmo_push_quad`'s two triangles are wound `(p00, p10, p01)`/`(p01,
    // p10, p11)`, i.e. mesh-winding order (major-angle edge first,
    // minor-angle edge second); working out
    // `cross(∂p/∂a, ∂p/∂beta)` for that winding against this tube's own
    // `(tangent, radial, binormal)` frame (`binormal = tangent × radial`)
    // gives exactly `-(radial*cos(beta) + binormal*sin(beta))` - the
    // opposite sign from the offset used to place the point. Passed to
    // `gizmo_push_quad` for a real smooth-shaded tube (rather than the flat
    // per-triangle normal `recompute_flat_normals` derives for the faceted
    // arrow cones), this is what keeps the normal's sign consistent with
    // the mesh winding the fragment shader's `front_facing` flip is
    // calibrated against - getting it backwards left the commonly-visible
    // side of the tube shaded as if facing away from the light (ambient
    // only, reading as much too dark).
    let ring_tube_normal = |frame: (Point3<f32>, Vector3<f32>, Vector3<f32>),
                            beta: f32|
     -> Vector3<f32> { -ring_tube_offset(frame, beta) };
    let band_color = |beta: f32| -> [f32; 4] {
        let band_width = TAU / GIZMO_BAND_COUNT as f32;
        let raw = (beta - phase_angle).rem_euclid(TAU) / band_width;
        let band_index = raw.floor() as usize % GIZMO_BAND_COUNT;
        let next_index = (band_index + 1) % GIZMO_BAND_COUNT;
        lerp_color(palette[band_index], palette[next_index], raw.fract())
    };

    for i in 0..GIZMO_RING_SEGMENTS {
        let a0 = (i as f32 / GIZMO_RING_SEGMENTS as f32) * TAU;
        let a1 = ((i + 1) as f32 / GIZMO_RING_SEGMENTS as f32) * TAU;
        let (Some(frame0), Some(frame1)) = (ring_frame(a0), ring_frame(a1)) else {
            continue;
        };

        for j in 0..GIZMO_RING_TUBE_SEGMENTS {
            let b0 = (j as f32 / GIZMO_RING_TUBE_SEGMENTS as f32) * TAU;
            let b1 = ((j + 1) as f32 / GIZMO_RING_TUBE_SEGMENTS as f32) * TAU;
            let c0 = band_color(b0);
            let c1 = band_color(b1);

            gizmo_push_quad(
                out,
                ring_tube_point(frame0, b0),
                ring_tube_normal(frame0, b0),
                c0,
                ring_tube_point(frame1, b0),
                ring_tube_normal(frame1, b0),
                c0,
                ring_tube_point(frame0, b1),
                ring_tube_normal(frame0, b1),
                c1,
                ring_tube_point(frame1, b1),
                ring_tube_normal(frame1, b1),
                c1,
            );
        }
    }

    // Field-loop marker toruses: small tubes threaded around the main ring,
    // perpendicular to its own tangent at that point. Built directly in
    // already-projected 3D space (rather than in the 4D invariant plane)
    // since the loop is a purely decorative screen-space visual, not a real
    // 4D subspace of its own.
    let marker_spacing = TAU / GIZMO_MARKER_COUNT as f32;
    for i in 0..GIZMO_MARKER_COUNT {
        let marker_angle = i as f32 * marker_spacing;
        let Some((center_3d, radial, binormal)) = ring_frame(marker_angle) else {
            continue;
        };

        // The marker's own secondary frame at angle `a2` around its small
        // circle: an exact closed form (no finite differencing needed,
        // since the marker circle is already fully known analytically from
        // `radial`/`binormal`).
        let marker_secondary_frame =
            |a2: f32| -> (Point3<f32>, Vector3<f32>, Vector3<f32>, Vector3<f32>) {
                let radial2 = radial * a2.cos() + binormal * a2.sin();
                let tangent2 = binormal * a2.cos() - radial * a2.sin();
                let q = center_3d + radial2 * GIZMO_MARKER_RADIUS;
                let binormal2 = tangent2.cross(&radial2).normalize();
                (q, tangent2, radial2, binormal2)
            };

        for j in 0..GIZMO_MARKER_SEGMENTS {
            let a2_0 = (j as f32 / GIZMO_MARKER_SEGMENTS as f32) * TAU;
            let a2_1 = ((j + 1) as f32 / GIZMO_MARKER_SEGMENTS as f32) * TAU;
            let (q0, _, radial2_0, binormal2_0) = marker_secondary_frame(a2_0);
            let (q1, _, radial2_1, binormal2_1) = marker_secondary_frame(a2_1);

            for k in 0..GIZMO_MARKER_TUBE_SEGMENTS {
                let g0 = (k as f32 / GIZMO_MARKER_TUBE_SEGMENTS as f32) * TAU;
                let g1 = ((k + 1) as f32 / GIZMO_MARKER_TUBE_SEGMENTS as f32) * TAU;

                // Same unit radial offset used to place the point, evaluated
                // in this marker's own secondary frame.
                let offset00 = radial2_0 * g0.cos() + binormal2_0 * g0.sin();
                let offset01 = radial2_0 * g1.cos() + binormal2_0 * g1.sin();
                let offset10 = radial2_1 * g0.cos() + binormal2_1 * g0.sin();
                let offset11 = radial2_1 * g1.cos() + binormal2_1 * g1.sin();
                let p00 = q0 + offset00 * GIZMO_MARKER_TUBE_MINOR_RADIUS;
                let p01 = q0 + offset01 * GIZMO_MARKER_TUBE_MINOR_RADIUS;
                let p10 = q1 + offset10 * GIZMO_MARKER_TUBE_MINOR_RADIUS;
                let p11 = q1 + offset11 * GIZMO_MARKER_TUBE_MINOR_RADIUS;

                // Same negated-offset shading normal as `ring_tube_normal`
                // (see its doc comment for the winding derivation) - the
                // offset used for position is the wrong sign for shading.
                gizmo_push_quad(
                    out,
                    p00,
                    -offset00,
                    marker_color,
                    p10,
                    -offset10,
                    marker_color,
                    p01,
                    -offset01,
                    marker_color,
                    p11,
                    -offset11,
                    marker_color,
                );
            }
        }

        // Arrows around this marker's own circumference: evenly spaced,
        // creeping by `phase_angle` modulo one arrow's own spacing - like
        // the flow of a field around its own field line - so their motion
        // is purely a function of accumulated rotation, never wall-clock
        // time. Each is a solid 3D cone anchored on the marker tube's outer
        // surface, always pointing tangentially in the direction of
        // rotation (the sign of `phase_angle`).
        let arrow_spacing = TAU / GIZMO_MARKER_ARROW_COUNT as f32;
        let creep = phase_angle.rem_euclid(arrow_spacing);
        let direction = if phase_angle < 0.0 { -1.0 } else { 1.0 };
        const ARROW_BASE_POINTS: usize = 4;
        let arrows_start = out.len();
        for k in 0..GIZMO_MARKER_ARROW_COUNT {
            let a2 = k as f32 * arrow_spacing + creep;
            let (q, tangent2, radial2, binormal2) = marker_secondary_frame(a2);

            let surface = q + radial2 * GIZMO_MARKER_TUBE_MINOR_RADIUS;
            let tip = surface + tangent2 * direction * GIZMO_ARROW_LENGTH;
            let base_center = surface - tangent2 * direction * GIZMO_ARROW_BACK_OFFSET;
            let bases: [Point3<f32>; ARROW_BASE_POINTS] = std::array::from_fn(|b| {
                let phi = (b as f32 / ARROW_BASE_POINTS as f32) * TAU;
                base_center
                    + (radial2 * phi.cos() + binormal2 * phi.sin()) * GIZMO_ARROW_BASE_RADIUS
            });

            for b in 0..ARROW_BASE_POINTS {
                let next = (b + 1) % ARROW_BASE_POINTS;
                out.push(gizmo_vertex(tip, arrow_color));
                out.push(gizmo_vertex(bases[b], arrow_color));
                out.push(gizmo_vertex(bases[next], arrow_color));
            }
            for b in 1..ARROW_BASE_POINTS - 1 {
                out.push(gizmo_vertex(bases[0], arrow_color));
                out.push(gizmo_vertex(bases[b], arrow_color));
                out.push(gizmo_vertex(bases[b + 1], arrow_color));
            }
        }
        // Only the arrow cones want flat, faceted normals - the ring and
        // marker tube quads above already carried real smooth normals from
        // `gizmo_push_quad` itself, computed analytically as they were
        // built.
        recompute_flat_normals(&mut out[arrows_start..]);
    }
}

/// Builds the GPU instance list for the current frame. Piece state is
/// already final (`apply_move` commits atomically) - while a move is
/// animating, the 27 affected facets are instead swept from their pre-move
/// position/kind toward that already-committed final position, using the
/// exact same rotation formula `apply_move` used, so the last animated
/// frame always lines up perfectly with the static post-move render it
/// hands off to.
pub fn sticker_instances_for_render(state: &HypercubeShaderState) -> Vec<StickerInstance> {
    let Some(animating) = &state.animating_move else {
        return generate_sticker_instances(&state.hypercube);
    };

    let t = if animating.duration.is_zero() {
        1.0
    } else {
        (animating.elapsed.as_secs_f32() / animating.duration.as_secs_f32()).clamp(0.0, 1.0)
    };
    let partial_angle = animating.angle * ease(t);
    let axes = free_axes(animating.side_axis);

    FACET_TABLE
        .iter()
        .map(|facet| {
            let pre_move_piece = &animating.pre_move_pieces[facet.piece_slot];
            let kind = pre_move_piece.kinds[facet.axis]
                .expect("FACET_TABLE entries are only built where kinds[axis] is Some");

            let (position_4d, basis, face_normal_4d) = if pre_move_piece.position
                [animating.side_axis]
                == animating.side_sign
            {
                // `facet_position_4d`'s static convention is `pos *
                // GRID_EXTENT + extension`, where `extension` is zero except
                // at the facet's own axis (the extra push from grid-scale
                // out to the tesseract boundary). `extension` is fixed to
                // the piece's own body, so it must rotate along with the
                // piece rather than stay pinned to the pre-move axis -
                // folding it into the local position before rotating
                // (instead of exempting an axis after rotating) achieves
                // that, since rotation is linear.
                let mut local_combined = [
                    pre_move_piece.position[axes[0]] as f32 * GRID_EXTENT,
                    pre_move_piece.position[axes[1]] as f32 * GRID_EXTENT,
                    pre_move_piece.position[axes[2]] as f32 * GRID_EXTENT,
                ];
                let facet_axis_is_free = axes.iter().position(|&axis| axis == facet.axis);
                if let Some(i) = facet_axis_is_free {
                    local_combined[i] +=
                        pre_move_piece.position[facet.axis] as f32 * (1.0 - GRID_EXTENT);
                }
                let rotated =
                    rotate_local_position(animating.local_coords, partial_angle, local_combined);

                let mut position_4d = [0.0f32; 4];
                position_4d[animating.side_axis] = pre_move_piece.position[animating.side_axis]
                    as f32
                    * if facet.axis == animating.side_axis {
                        1.0
                    } else {
                        GRID_EXTENT
                    };
                for i in 0..3 {
                    position_4d[axes[i]] = rotated[i];
                }

                // The facet's static mesh basis is the unit vectors along
                // `facet.free_axes`. Each one is either exactly `side_axis`
                // (untouched by the slab's rotation) or a one-hot vector
                // inside the rotating `axes` subspace - rotating that one-hot
                // vector the same way position is rotated above gives the
                // basis vector's new direction directly, by linearity.
                let mut basis = [[0.0f32; 4]; 3];
                for (i, &a) in facet.free_axes.iter().enumerate() {
                    basis[i] = if a == animating.side_axis {
                        let mut v = [0.0f32; 4];
                        v[a] = 1.0;
                        v
                    } else {
                        let j = axes.iter().position(|&x| x == a).expect(
                            "a facet free axis other than side_axis must be one of the move's free axes",
                        );
                        let mut one_hot = [0.0f32; 3];
                        one_hot[j] = 1.0;
                        let rotated =
                            rotate_local_position(animating.local_coords, partial_angle, one_hot);
                        let mut v = [0.0f32; 4];
                        for k in 0..3 {
                            v[axes[k]] = rotated[k];
                        }
                        v
                    };
                }

                // The facet's outward normal is the one-hot vector along its
                // own axis (signed by `side_sign`), used for 4D face
                // culling. When `facet.axis == side_axis`, this is the
                // slab's own outer face - it doesn't rotate, so it stays
                // static. Otherwise it's genuinely sweeping toward a
                // different tesseract cell along with the rest of the
                // rotating subspace, so it rotates the same way the
                // tangent basis vectors above do.
                let face_normal_4d = if let Some(j) = facet_axis_is_free {
                    let mut one_hot = [0.0f32; 3];
                    one_hot[j] = facet.side_sign as f32;
                    let rotated =
                        rotate_local_position(animating.local_coords, partial_angle, one_hot);
                    let mut v = [0.0f32; 4];
                    for k in 0..3 {
                        v[axes[k]] = rotated[k];
                    }
                    v
                } else {
                    FACE_CENTERS[facet.face_id].into()
                };

                (position_4d, basis, face_normal_4d)
            } else {
                (facet.position_4d, facet.basis, FACE_CENTERS[facet.face_id].into())
            };

            StickerInstance {
                position_4d,
                basis,
                face_normal_4d,
                kind: kind as u32,
                _padding: [0; 3],
            }
        })
        .collect()
}

/// Parameters controlled from the ui.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UiControls {
    pub(crate) sticker_scale: f32,
    pub(crate) face_gap: f32,
    pub(crate) face_gap_4d: f32,
    pub(crate) viewer_distance: f32,
    pub(crate) render_mode: RenderMode,
    pub(crate) theme: Theme,
    /// See `Renderer::set_ground_truth_debug_face`.
    pub(crate) ground_truth_debug_face: Option<u32>,
}

/// How much a ray's hit distance against another sticker's geometry may
/// differ from a tested corner's own distance from the camera before the
/// two count as actually separated, rather than the same point restated by
/// floating-point noise.
const DEPTH_TIE_EPSILON: f32 = 1e-4;

/// One sticker of the kind being ordered (Fire, Ice, ...) as `depth_draw_order`
/// sees it.
struct FireSticker {
    instance_index: u32,
    /// Depth along the camera's view axis of the sticker's rendered center.
    depth: f32,
    /// The sticker's 8 world-space corners, placed the same way the vertex
    /// shader places them (see `transform_sticker_vertices_to_3d`).
    world_vertices: Vec<Point3<f32>>,
    /// Bounding box of `world_vertices`, checked before the exact per-corner
    /// ray test.
    aabb: AABB,
}

impl FireSticker {
    /// Whether this sticker must be drawn before `other` - that is, whether
    /// it is behind it.
    ///
    /// For each of this sticker's 8 corners, casts a ray from the camera eye
    /// through that corner and tests it against `other`'s real geometry with
    /// `ray_sticker_intersection` - the same exact ray-vs-cube test hover and
    /// click picking already use. A nearer hit means `other` occludes this
    /// corner there; a farther hit means this corner is nearer than
    /// `other`'s surface there. The same is done with the two stickers
    /// swapped, since a size mismatch after the 4D perspective divide can
    /// leave every corner of a smaller sticker outside a larger one's
    /// silhouette even while they visually overlap.
    ///
    /// Two stickers on different tesseract cells can meet at a boundary
    /// that, after each cell's own projection, is no longer a single plane;
    /// viewed near-edge-on, that boundary can curve enough that no single
    /// front/back answer is correct for the whole overlap. When tested
    /// corners disagree, the verdict from whichever corner point is nearest
    /// the camera wins, since a cycle of disagreeing corners has no answer
    /// more correct than that.
    fn is_behind(&self, camera_eye: Point3<f32>, other: &Self) -> Option<Ordering> {
        let mut nearest: Option<(f32, Ordering)> = None;

        let mut test_corner = |corner: Point3<f32>,
                               target: &Self,
                               order_if_target_nearer: Ordering| {
            let offset = corner - camera_eye;
            let corner_distance = offset.norm();
            if corner_distance < f32::EPSILON {
                return;
            }
            let direction = offset / corner_distance;
            let ray = Ray {
                origin: camera_eye,
                direction,
                inverse_direction: direction.map(|component| 1.0 / component),
            };
            if !ray_intersects_aabb(&ray, &target.aabb) {
                return;
            }
            let Some(hit_distance) = ray_sticker_intersection(&ray, &target.world_vertices) else {
                return;
            };
            if (hit_distance - corner_distance).abs() < DEPTH_TIE_EPSILON {
                return;
            }
            let order = if hit_distance < corner_distance {
                order_if_target_nearer
            } else {
                order_if_target_nearer.reverse()
            };
            let is_nearest = match nearest {
                Some((nearest_distance, _)) => corner_distance < nearest_distance,
                None => true,
            };
            if is_nearest {
                nearest = Some((corner_distance, order));
            }
        };

        for &corner in &self.world_vertices {
            test_corner(corner, other, Ordering::Greater);
        }
        for &corner in &other.world_vertices {
            test_corner(corner, self, Ordering::Less);
        }

        nearest.map(|(_, order)| order)
    }
}

/// How long the Fire ground-truth debug inset spends on each face before
/// cycling to the next, in seconds.
const GROUND_TRUTH_DEBUG_FACE_CYCLE_SECONDS: f32 = 2.0;

/// Which face_id, if any, the Fire ground-truth debug inset should draw this
/// frame: `None` when the feature is off, the theme has no Fire pass, or no
/// instance is currently kind `ELEMENTAL_FIRE_KIND`; otherwise cycles
/// through every face_id currently holding a Fire sticker in ascending
/// order, `GROUND_TRUTH_DEBUG_FACE_CYCLE_SECONDS` per face, looping forever.
/// Driven by `elapsed_seconds` rather than its own timer, since Fire's
/// continuous animation already forces a redraw every tick under
/// `Theme::Elemental`.
fn ground_truth_debug_face(
    enabled: bool,
    theme: Theme,
    instances: &[StickerInstance],
    elapsed_seconds: f32,
) -> Option<u32> {
    if !enabled || theme != Theme::Elemental || instances.is_empty() {
        return None;
    }

    let facets_per_face = instances.len() / 8;
    let fire_faces: Vec<u32> = (0..8u32)
        .filter(|&face_id| {
            let start = face_id as usize * facets_per_face;
            instances[start..start + facets_per_face]
                .iter()
                .any(|instance| instance.kind == ELEMENTAL_FIRE_KIND)
        })
        .collect();

    let cycle_index = (elapsed_seconds / GROUND_TRUTH_DEBUG_FACE_CYCLE_SECONDS) as usize
        % fire_faces.len().max(1);
    fire_faces.get(cycle_index).copied()
}

/// One sticker within a batch `depth_draw_order` produces, tagging which
/// pass/pipeline `render()` needs to draw it with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DepthLayer {
    /// Blended, no depth write - see `fs_fire`.
    Fire(u32),
    /// Opaque, drawn against its batch's shared background snapshot - see
    /// `fs_ice`.
    Ice(u32),
    /// Blended, no depth write - see `fs_light`.
    Light(u32),
    /// Blended, no depth write - see `fs_dirt`.
    Dirt(u32),
}

/// The batches Fire's, Ice's, Light's and Dirt's stickers are drawn in
/// together under `Theme::Elemental`: every instance of
/// `ELEMENTAL_FIRE_KIND`, `ELEMENTAL_ICE_KIND`, `ELEMENTAL_LIGHT_KIND` or
/// `ELEMENTAL_DIRT_KIND` on a visible face, grouped
/// into passes that must run strictly in the returned order (farthest batch
/// first), but whose members within one batch can draw in any order - all
/// four need a back-to-front order because what they draw depends on draw
/// order, whether from blending (Fire, Light, Dirt) or from each layer's
/// background snapshot needing to already contain every farther layer's own
/// result (Ice), and any of the four can sit in front of or behind any
/// other one depending on the current 4D rotation.
///
/// Fire, Light and Dirt draw blended and write no depth, so where two of
/// their stickers overlap on screen the result depends on the order they
/// are drawn in. Ice instead writes an opaque result but reads back a
/// snapshot of the scene so far, so a nearer Ice sticker must draw after
/// every farther one - Fire, Light and Dirt included - for its refraction
/// to show them. Either way, `FireSticker::is_behind` decides each pair
/// exactly, by casting rays through their real projected corners and
/// testing against each other's actual geometry - the same ray-vs-cube test
/// hover and click picking already use - rather than an approximating
/// scalar key. That relation is combined into batches by topologically
/// sorting it in layers: every step takes every sticker nothing undrawn
/// still blocks, all at once, rather than just the single farthest one. A
/// pair with no edge between them in the relation is a pair `is_behind`
/// found no evidence occludes the other on screen, which is exactly what
/// makes sharing one pass safe for Fire, Light and Dirt, whose blending
/// only depends on literally-overlapping pixels. It's a deliberately looser
/// bar for Ice,
/// whose reflection can sample any point on screen regardless of 3D
/// proximity: two same-batch Ice stickers no longer see each other's own
/// result the way strictly sequential draws would, so mutual reflection
/// detail between them (e.g. two occlusion-unrelated neighbors on the same
/// face) is traded away for far fewer background-snapshot copies - accepted
/// because the alternative (never batching more than one Ice sticker
/// together) keeps the exact cost this batching exists to avoid. Should the
/// relation ever contain a cycle (which can happen for two stickers on
/// different tesseract cells whose shared boundary is viewed near-edge-on,
/// see `FireSticker::is_behind`), the stickers caught in it fall back to
/// their own singleton batch, farthest first, rather than being lost.
///
/// At most `instances.len() / 4` stickers take part, so this runs on the CPU
/// each frame.
///
/// # Arguments
/// * `instances` - this frame's full instance list, in face-major order
/// * `visible_faces` - per-`face_id` visibility (see `math::visible_faces`)
/// * `rotation_4d` - 4D rotation matrix
/// * `camera` - the 3D camera, for its eye point and view axis
/// * `lattice_is_aligned` - false while a move animation is sweeping a slab;
///   depth alone is used for every pair in that case, since the moving
///   face's cells briefly aren't the axis-aligned lattice the rest of the
///   frame assumes
/// * `sticker_scale`/`face_gap`/`face_gap_4d`/`viewer_distance` - the same
///   placement parameters the vertex shader is given, so the centers and
///   corners land where the stickers actually render
#[allow(clippy::too_many_arguments)]
pub(crate) fn depth_draw_order(
    instances: &[StickerInstance],
    visible_faces: &[bool; 8],
    rotation_4d: &Matrix4<f32>,
    camera: &Camera,
    lattice_is_aligned: bool,
    sticker_scale: f32,
    face_gap: f32,
    face_gap_4d: f32,
    viewer_distance: f32,
) -> Vec<Vec<DepthLayer>> {
    if instances.is_empty() {
        return Vec::new();
    }

    let forward = (camera.target - camera.eye).normalize();
    let facets_per_face = instances.len() / 8;

    let stickers: Vec<FireSticker> = instances
        .iter()
        .enumerate()
        .filter(|(index, instance)| {
            (instance.kind == ELEMENTAL_FIRE_KIND
                || instance.kind == ELEMENTAL_ICE_KIND
                || instance.kind == ELEMENTAL_LIGHT_KIND
                || instance.kind == ELEMENTAL_DIRT_KIND)
                && visible_faces[index / facets_per_face]
        })
        .map(|(index, instance)| {
            let face_id = index / facets_per_face;
            let position_4d = Vector4::from(instance.position_4d);
            let center = project_face_point(
                position_4d,
                Vector4::from(instance.face_normal_4d),
                rotation_4d,
                face_gap,
                face_gap_4d,
                viewer_distance,
            );
            let world_vertices = transform_sticker_vertices_to_3d(
                position_4d,
                face_id,
                rotation_4d,
                sticker_scale,
                face_gap,
                face_gap_4d,
                viewer_distance,
            );
            let aabb = calculate_sticker_aabb(&world_vertices);

            FireSticker {
                instance_index: index as u32,
                depth: (center - camera.eye).dot(&forward),
                world_vertices,
                aabb,
            }
        })
        .collect();

    topological_draw_order(&stickers, camera.eye, lattice_is_aligned)
        .into_iter()
        .map(|batch| {
            batch
                .into_iter()
                .map(|instance_index| {
                    match instances[instance_index as usize].kind {
                        ELEMENTAL_FIRE_KIND => DepthLayer::Fire(instance_index),
                        ELEMENTAL_ICE_KIND => DepthLayer::Ice(instance_index),
                        ELEMENTAL_LIGHT_KIND => DepthLayer::Light(instance_index),
                        ELEMENTAL_DIRT_KIND => DepthLayer::Dirt(instance_index),
                        // The filter above only lets these four kinds
                        // through.
                        _ => unreachable!(),
                    }
                })
                .collect()
        })
        .collect()
}

/// Layers the "is behind" relation over `stickers` into draw batches,
/// farthest batch first: each batch holds every sticker nothing
/// still-undrawn blocks, taken all at once rather than one at a time, so a
/// batch's members are exactly those with no established order between any
/// pair of them - safe to draw in any order, and so safe to share one
/// render pass. Ties and unrelated-when-batching pairs are broken by depth,
/// so the result is deterministic; a cycle (which the relation should not
/// contain except for the near-edge-on case documented on
/// `FireSticker::is_behind`) degrades to a singleton batch, farthest
/// remaining first, for the stickers caught in it, rather than losing them
/// or wrongly batching stickers an unresolved cycle proves DO interact.
fn topological_draw_order(
    stickers: &[FireSticker],
    camera_eye: Point3<f32>,
    lattice_is_aligned: bool,
) -> Vec<Vec<u32>> {
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); stickers.len()];
    let mut blocked_by = vec![0usize; stickers.len()];

    if lattice_is_aligned {
        for (i, sticker) in stickers.iter().enumerate() {
            for (j, other) in stickers.iter().enumerate().skip(i + 1) {
                let (before, after) = match sticker.is_behind(camera_eye, other) {
                    Some(Ordering::Greater) => (i, j),
                    Some(Ordering::Less) => (j, i),
                    _ => continue,
                };
                successors[before].push(after);
                blocked_by[after] += 1;
            }
        }
    }

    let farthest = |candidates: &mut dyn Iterator<Item = usize>| {
        candidates.max_by(|&a, &b| stickers[a].depth.total_cmp(&stickers[b].depth))
    };

    let mut batches: Vec<Vec<u32>> = Vec::new();
    let mut drawn = vec![false; stickers.len()];
    let mut remaining = stickers.len();

    while remaining > 0 {
        let unblocked: Vec<usize> = (0..stickers.len())
            .filter(|&i| !drawn[i] && blocked_by[i] == 0)
            .collect();

        let batch: Vec<usize> = if unblocked.is_empty() {
            // A cycle among what's left: these stickers DO have edges among
            // themselves, just none currently satisfiable, so only the
            // single farthest remaining one goes next to break it, rather
            // than batching a set that isn't actually proven
            // non-interacting.
            match farthest(&mut (0..stickers.len()).filter(|&i| !drawn[i])) {
                Some(next) => vec![next],
                None => break,
            }
        } else if lattice_is_aligned {
            let mut batch = unblocked;
            batch.sort_by(|&a, &b| stickers[b].depth.total_cmp(&stickers[a].depth));
            batch
        } else {
            // No relation was built at all this frame (a move animation is
            // sweeping a slab), so every sticker looks unblocked with zero
            // overlap evidence behind that either way - batching on that
            // would claim a guarantee that doesn't hold. Fall back to
            // today's one-at-a-time depth order instead.
            let next = farthest(&mut unblocked.into_iter()).expect("unblocked is non-empty here");
            vec![next]
        };

        for &index in &batch {
            drawn[index] = true;
            remaining -= 1;
            for &successor in &successors[index] {
                blocked_by[successor] = blocked_by[successor].saturating_sub(1);
            }
        }
        batches.push(
            batch
                .iter()
                .map(|&index| stickers[index].instance_index)
                .collect(),
        );
    }

    batches
}

fn scale_bounds(bounds: &Rectangle, scale: f32) -> Rectangle {
    Rectangle {
        x: bounds.x * scale,
        y: bounds.y * scale,
        width: bounds.width * scale,
        height: bounds.height * scale,
    }
}

/// Custom primitive for rendering our 4D hypercube
#[derive(Debug, Clone)]
pub(crate) struct HypercubePrimitive {
    pub(crate) camera: Camera,
    pub(crate) projection: Projection,
    pub(crate) rotation_4d: Matrix4<f32>,
    pub(crate) ui_controls: UiControls,
    pub(crate) cached_indices: Arc<[u16]>,
    pub(crate) indices_generation: u64,
    pub(crate) hovered_sticker: Option<usize>,
    pub(crate) debug_instances: Vec<DebugInstanceWithDistance>,
    /// This frame's rotation-axis gizmo geometry (see `gizmo_torus_vertices`);
    /// empty unless a focus animation or Shift+drag is currently in
    /// progress.
    pub(crate) gizmo_vertices: Vec<GizmoVertex>,
    pub(crate) sticker_instances: Arc<[StickerInstance]>,
    pub(crate) sticker_generation: u64,
    pub(crate) visible_faces: [bool; 8],
    /// Back-to-front draw batches for Fire's blended pass and Ice's
    /// per-batch raymarch passes, interleaved so either can sit in front of
    /// the other; batches must render in order, but a batch's own members
    /// share one pass since nothing established an order between them
    /// (see `depth_draw_order`). Empty under any theme that doesn't draw
    /// either.
    pub(crate) depth_order: Vec<Vec<DepthLayer>>,
    /// Wall-clock seconds since the app started, wrapped modulo 3600.
    pub(crate) elapsed_seconds: f32,
    /// Set by a `save_snapshot_generation` mismatch; `prepare()` captures
    /// this frame's pixels and writes both to disk alongside it when present.
    pub(crate) snapshot_request: Option<ViewSnapshot>,
}

impl shader::Primitive for HypercubePrimitive {
    type Pipeline = Renderer;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        viewport: &shader::Viewport,
    ) {
        let scale = viewport.scale_factor();
        let physical_bounds = scale_bounds(bounds, scale);
        pipeline.resize(device, physical_bounds, viewport.physical_size());
        pipeline.update_instances(
            queue,
            &self.rotation_4d,
            self.ui_controls.sticker_scale,
            self.ui_controls.face_gap,
            self.ui_controls.face_gap_4d,
            self.ui_controls.viewer_distance,
            self.elapsed_seconds,
        );
        pipeline.update_camera(queue, &self.camera, &self.projection);
        pipeline.update_light(queue, &self.camera);
        pipeline.update_indices(queue, &self.cached_indices, self.indices_generation);
        pipeline.update_highlighting(queue, self.hovered_sticker);
        pipeline.update_debug_instances(queue, &self.debug_instances);
        pipeline.update_gizmo(queue, &self.gizmo_vertices);
        pipeline.update_sticker_instances(queue, &self.sticker_instances, self.sticker_generation);
        pipeline.update_depth_batches(queue, &self.depth_order);
        pipeline.set_render_mode(self.ui_controls.render_mode);
        pipeline.set_theme(self.ui_controls.theme);
        pipeline.set_ground_truth_debug_face(self.ui_controls.ground_truth_debug_face);

        if let Some(request) = &self.snapshot_request {
            let (rgba, width, height) = pipeline.capture_frame(device, queue, &self.visible_faces);
            snapshot::save(request, &rgba, width, height);
        }
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        _clip_bounds: &Rectangle<u32>,
    ) {
        // The rotation-axis gizmo, if any, is drawn inside `pipeline.render`
        // itself now (right after the opaque hypercube pass, before the
        // translucent Fire/Ice/Light/Dirt batches), so it gets real shading
        // and depth instead of compositing on top as a flat overlay.
        pipeline.render(encoder, &self.visible_faces);
        pipeline.composite(encoder, target);

        // Render transparent debug AABBs
        pipeline.render_debug_aabb(encoder, target, self.debug_instances.len() as u32);
    }
}

/// Internal state managed by the shader widget
pub struct HypercubeShaderState {
    pub(crate) camera: Camera,
    camera_controller: CameraController,
    projection: Projection,
    pub(crate) rotation_4d: nalgebra::Matrix4<f32>,
    mouse_pressed: bool,
    last_mouse_pos: Option<Point>,
    shift_pressed: bool,
    cached_indices: Arc<[u16]>,
    /// Bumped every time `cached_indices` is replaced; carried on
    /// `HypercubePrimitive` so `Renderer` can skip re-uploading the index
    /// buffer to the GPU when it hasn't actually changed since last frame.
    indices_generation: u64,
    cached_sticker_instances: Arc<[StickerInstance]>,
    /// Bumped every time `cached_sticker_instances` is replaced; same
    /// upload-skipping purpose as `indices_generation`.
    sticker_generation: u64,
    hovered_sticker: Option<usize>,
    debug_instances: Vec<DebugInstanceWithDistance>,
    hypercube: Hypercube,
    animating_move: Option<AnimatingMove>,
    animating_focus: Option<AnimatingFocus>,
    animating_reset: Option<AnimatingReset>,
    animating_reveal: Option<AnimatingReveal>,
    /// Set while a Shift+drag 4D rotation gesture is in progress; drives the
    /// rotation-axis gizmo the same way `animating_focus` does for the
    /// click-to-focus animation. See `ActiveShiftDrag`.
    active_shift_drag: Option<ActiveShiftDrag>,
    /// Live sticker scale/face gap while a reveal/hide flourish is playing,
    /// consulted by `draw()`/`update_hover` in preference to
    /// `HypercubeShaderProgram`'s own fields. Self-corrects back to `None`
    /// once `HypercubeApp` has caught up to the animation's final value (see
    /// `Program::update`), so it never masks a later manual slider drag.
    reveal_scale_override: Option<f32>,
    reveal_gap_override: Option<f32>,
    reveal_gap_4d_override: Option<f32>,
    /// Position and hovered sticker (if any) recorded when the rotate button
    /// was last pressed, used at release time to tell a click from a drag.
    rotate_press: Option<(Point, Option<usize>)>,
    /// Time and face of an unmatched first click on the rotate button,
    /// waiting to see if a second click lands within `DOUBLE_CLICK_WINDOW`.
    pending_face_click: Option<(Instant, usize)>,
    last_redraw_instant: Option<Instant>,
    /// Timestamp of the previous `RedrawRequested` tick, used only to
    /// accumulate `elapsed_seconds`; unlike `last_redraw_instant`, this is
    /// never reset to `None` while animations are idle.
    last_tick_instant: Option<Instant>,
    /// Wall-clock seconds since the app started, wrapped modulo 3600.
    elapsed_seconds: f32,
    reset_generation: u64,
    random_moves_generation: u64,
    /// Seeded once at construction, reused across every random-move press so
    /// results are reproducible from a fixed seed in tests but still vary
    /// run-to-run in the live app (`Rng::new()` seeds from OS entropy).
    rng: fastrand::Rng,
    reveal_generation: u64,
    save_generation: u64,
    load_generation: u64,
    save_snapshot_generation: u64,
    /// A snapshot request built by `Program::update` on a
    /// `save_snapshot_generation` mismatch, for `draw()` to attach to the
    /// next `HypercubePrimitive` - `Cell` because `draw()` only gets
    /// `&State`, mirroring `HypercubeShaderProgram::pending_load`.
    pending_snapshot: Cell<Option<ViewSnapshot>>,
    solve_command_generation: u64,
    solve_playback: Option<SolvePlayback>,
    /// A solve progress/end message waiting to be published: `update` can
    /// publish only one message per call and a completed reveal's goes
    /// first, so this holds the solve's until the next call.
    solve_outbox: Option<Message>,
    /// How far the last move animation ran past its duration, carried into
    /// the next solve move so playback keeps pace with wall-clock time
    /// instead of losing part of a frame on every move.
    move_overshoot: Duration,
}

impl HypercubeShaderState {
    /// Replaces `cached_indices` and bumps `indices_generation`, so
    /// `Renderer::update_indices` can tell this frame's data apart from
    /// what's already on the GPU.
    fn set_cached_indices(&mut self, indices: Vec<u16>) {
        self.cached_indices = indices.into();
        self.indices_generation += 1;
    }

    /// Replaces `cached_sticker_instances` and bumps `sticker_generation`,
    /// mirroring `set_cached_indices`.
    fn set_cached_sticker_instances(&mut self, instances: Vec<StickerInstance>) {
        self.cached_sticker_instances = instances.into();
        self.sticker_generation += 1;
    }
}

/// The shader program that handles 4D hypercube rendering
pub struct HypercubeShaderProgram {
    sticker_scale: f32,
    face_gap: f32,
    face_gap_4d: f32,
    viewer_distance: f32,
    render_mode: RenderMode,
    theme: Theme,
    aabb_mode: AABBMode,
    fire_ground_truth_debug: bool,
    show_gizmo_ring: bool,
    rotate_button: RotateButton,
    animation_duration_ms: u32,
    reset_generation: u64,
    random_moves_generation: u64,
    random_move_count: u32,
    reveal_generation: u64,
    revealed_target: bool,
    save_generation: u64,
    load_generation: u64,
    /// Puzzle state loaded by a `LoadPuzzle` press, if any.
    pending_load: Cell<Option<Hypercube>>,
    save_snapshot_generation: u64,
    solve_command_generation: u64,
    solve_command: SolveCommand,
}

impl HypercubeShaderProgram {
    /// Create a new shader program with the given parameters
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        sticker_scale: f32,
        face_gap: f32,
        face_gap_4d: f32,
        viewer_distance: f32,
        render_mode: RenderMode,
        theme: Theme,
        aabb_mode: AABBMode,
        fire_ground_truth_debug: bool,
        show_gizmo_ring: bool,
        rotate_button: RotateButton,
        animation_duration_ms: u32,
        reset_generation: u64,
        random_moves_generation: u64,
        random_move_count: u32,
        reveal_generation: u64,
        revealed_target: bool,
        save_generation: u64,
        load_generation: u64,
        pending_load: Option<Hypercube>,
        save_snapshot_generation: u64,
        solve_command_generation: u64,
        solve_command: SolveCommand,
    ) -> Self {
        Self {
            sticker_scale,
            face_gap,
            face_gap_4d,
            viewer_distance,
            render_mode,
            theme,
            aabb_mode,
            fire_ground_truth_debug,
            show_gizmo_ring,
            rotate_button,
            animation_duration_ms,
            reset_generation,
            random_moves_generation,
            random_move_count,
            reveal_generation,
            revealed_target,
            save_generation,
            load_generation,
            pending_load: Cell::new(pending_load),
            save_snapshot_generation,
            solve_command_generation,
            solve_command,
        }
    }
}

impl shader::Program<Message> for HypercubeShaderProgram {
    type State = HypercubeShaderState;
    type Primitive = HypercubePrimitive;

    fn update(
        &self,
        state: &mut Self::State,
        event: &Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<Action<Message>> {
        if self.reset_generation != state.reset_generation {
            state.animating_move = None;
            state.animating_focus = None;
            state.active_shift_drag = None;
            state.rotate_press = None;
            state.pending_face_click = None;
            state.hovered_sticker = None;
            state.debug_instances.clear();
            state.solve_playback = None;
            state.solve_outbox = None;
            state.reset_generation = self.reset_generation;

            let (start_p, start_q) = decompose_so4(&state.rotation_4d);
            state.animating_reset = Some(AnimatingReset {
                start_p,
                start_q,
                elapsed: Duration::ZERO,
                duration: Duration::from_millis(self.animation_duration_ms as u64),
            });
            state.last_redraw_instant = None;

            let instances = sticker_instances_for_render(state);
            state.set_cached_sticker_instances(instances);
            return Some(Action::request_redraw());
        }

        if self.random_moves_generation != state.random_moves_generation {
            state
                .hypercube
                .apply_random_moves(self.random_move_count, &mut state.rng);
            state.animating_move = None;
            state.animating_focus = None;
            state.rotate_press = None;
            state.pending_face_click = None;
            state.last_redraw_instant = None;
            state.hovered_sticker = None;
            state.debug_instances.clear();
            state.solve_playback = None;
            state.solve_outbox = None;
            state.random_moves_generation = self.random_moves_generation;
            let instances = sticker_instances_for_render(state);
            state.set_cached_sticker_instances(instances);
            return Some(Action::request_redraw());
        }

        if self.save_generation != state.save_generation {
            state.save_generation = self.save_generation;
            return Some(Action::publish(Message::PuzzleReadyToSave(
                state.hypercube.clone(),
            )));
        }

        if self.load_generation != state.load_generation {
            state.load_generation = self.load_generation;

            if let Some(hypercube) = self.pending_load.take() {
                state.hypercube = hypercube;
                state.animating_move = None;
                state.animating_focus = None;
                state.rotate_press = None;
                state.pending_face_click = None;
                state.hovered_sticker = None;
                state.debug_instances.clear();
                state.last_redraw_instant = None;
                state.solve_playback = None;
                state.solve_outbox = None;

                let instances = sticker_instances_for_render(state);
                state.set_cached_sticker_instances(instances);
                return Some(Action::request_redraw());
            }
        }

        if self.solve_command_generation != state.solve_command_generation {
            state.solve_command_generation = self.solve_command_generation;
            state.solve_playback = None;
            state.solve_outbox = None;
            if self.solve_command == SolveCommand::Start {
                let message = Self::start_solve(state, self.solve_command_generation);
                return Some(Action::publish(message));
            }
        }

        if self.save_snapshot_generation != state.save_snapshot_generation {
            state.save_snapshot_generation = self.save_snapshot_generation;
            state.pending_snapshot.set(Some(ViewSnapshot::capture(
                state.hypercube.clone(),
                state.rotation_4d,
                state.camera.clone(),
                state.camera_controller,
                state.projection,
                self.sticker_scale,
                self.face_gap,
                self.face_gap_4d,
                self.viewer_distance,
                self.render_mode,
                self.theme,
                self.aabb_mode,
            )));
            return None;
        }

        // Once `HypercubeApp` has caught up to a completed reveal/hide
        // flourish's final value (via `Message::RevealAnimationComplete`),
        // the fresh `self` built from it matches the override's stored
        // target - clear it so future manual slider drags aren't masked.
        if let Some(target) = state.reveal_scale_override
            && self.sticker_scale == target
        {
            state.reveal_scale_override = None;
        }
        if let Some(target) = state.reveal_gap_override
            && self.face_gap == target
        {
            state.reveal_gap_override = None;
        }
        if let Some(target) = state.reveal_gap_4d_override
            && self.face_gap_4d == target
        {
            state.reveal_gap_4d_override = None;
        }

        if self.reveal_generation != state.reveal_generation {
            state.animating_move = None;
            state.animating_focus = None;

            let (target_scale_raw, target_gap, target_gap_4d) = if self.revealed_target {
                (
                    SECONDARY_STICKER_SCALE,
                    SECONDARY_FACE_GAP,
                    SECONDARY_FACE_GAP_4D,
                )
            } else {
                (PRIMARY_STICKER_SCALE, PRIMARY_FACE_GAP, PRIMARY_FACE_GAP_4D)
            };
            let start_yaw = state.camera_controller.yaw;

            state.animating_reveal = Some(AnimatingReveal {
                start_scale: self.sticker_scale,
                target_scale: 1.0 - target_scale_raw,
                start_gap: self.face_gap,
                target_gap,
                start_gap_4d: self.face_gap_4d,
                target_gap_4d,
                start_yaw,
                target_yaw: start_yaw + REVEAL_YAW_SPIN_DEGREES,
                elapsed: Duration::ZERO,
                duration: REVEAL_ANIMATION_DURATION,
            });
            state.reveal_scale_override = Some(self.sticker_scale);
            state.reveal_gap_override = Some(self.face_gap);
            state.reveal_gap_4d_override = Some(self.face_gap_4d);
            state.reveal_generation = self.reveal_generation;
            state.last_redraw_instant = None;
            let instances = sticker_instances_for_render(state);
            state.set_cached_sticker_instances(instances);
            return Some(Action::request_redraw());
        }

        // Update camera each frame
        state.camera_controller.update_camera(&mut state.camera);

        // Update viewport size if bounds changed
        if bounds.width > 0.0 && bounds.height > 0.0 {
            state.projection.aspect = bounds.width / bounds.height;
        }

        // Check if 4D rotation changed and recalculate indices
        let mut rotation_changed = false;
        // Whether `sticker_instances_for_render` needs to run again this
        // tick: true whenever a move animation started, ended, or is still
        // in progress (its swept position changes every tick), false when
        // `animating_move` was and still is absent.
        let mut regenerate_stickers = false;
        let mut reveal_completed_message: Option<Message> = None;

        let status = match event {
            Event::Mouse(mouse_event) => {
                let old_rotation = state.rotation_4d;
                let was_animating = state.animating_move.is_some();
                let result = self.handle_mouse_event(state, mouse_event, bounds, cursor);
                if state.rotation_4d != old_rotation {
                    rotation_changed = true;
                }
                if was_animating || state.animating_move.is_some() {
                    regenerate_stickers = true;
                }
                result
            }
            Event::Keyboard(keyboard_event) => self.handle_keyboard_event(state, keyboard_event),
            Event::Window(iced::window::Event::RedrawRequested(now)) => {
                let delta = state
                    .last_redraw_instant
                    .map(|last| now.duration_since(last))
                    .unwrap_or_default();

                let tick_delta = state
                    .last_tick_instant
                    .map(|last| now.duration_since(last))
                    .unwrap_or_default();
                state.elapsed_seconds = (state.elapsed_seconds + tick_delta.as_secs_f32()) % 3600.0;
                state.last_tick_instant = Some(*now);

                let was_animating = state.animating_move.is_some();
                let move_tick = Self::advance_animation(state, delta);
                // Chain straight into the next solve move within the same
                // tick, so playback never idles a frame between moves.
                if state.animating_move.is_none()
                    && let Some(message) = Self::advance_solve_playback(state)
                {
                    state.solve_outbox = Some(message);
                }
                let focus_tick = Self::advance_focus_animation(state, delta);
                let reset_tick = Self::advance_reset_animation(state, delta);
                let reveal_tick = Self::advance_reveal_animation(state, delta);

                if was_animating || state.animating_move.is_some() {
                    regenerate_stickers = true;
                }

                if state.animating_move.is_none()
                    && state.animating_focus.is_none()
                    && state.animating_reset.is_none()
                    && state.animating_reveal.is_none()
                {
                    state.last_redraw_instant = None;
                } else {
                    state.last_redraw_instant = Some(*now);
                }

                if matches!(
                    focus_tick,
                    AnimationTick::Running | AnimationTick::Completed
                ) || matches!(
                    reset_tick,
                    AnimationTick::Running | AnimationTick::Completed
                ) {
                    rotation_changed = true;
                }

                if matches!(
                    (&move_tick, &focus_tick, &reset_tick, &reveal_tick),
                    (AnimationTick::Completed, _, _, _)
                        | (_, AnimationTick::Completed, _, _)
                        | (_, _, AnimationTick::Completed, _)
                        | (_, _, _, AnimationTick::Completed)
                ) && !state.mouse_pressed
                    && state.solve_playback.is_none()
                    && let Some(position) = cursor.position_in(bounds)
                {
                    self.update_hover(state, position, bounds);
                }

                if matches!(reveal_tick, AnimationTick::Completed) {
                    let (final_scale, final_gap, final_gap_4d) = if self.revealed_target {
                        (
                            SECONDARY_STICKER_SCALE,
                            SECONDARY_FACE_GAP,
                            SECONDARY_FACE_GAP_4D,
                        )
                    } else {
                        (PRIMARY_STICKER_SCALE, PRIMARY_FACE_GAP, PRIMARY_FACE_GAP_4D)
                    };
                    reveal_completed_message = Some(Message::RevealAnimationComplete {
                        final_scale,
                        final_gap,
                        final_gap_4d,
                    });
                }

                // Elemental's sticker materials animate continuously from
                // elapsed time alone, so its redraw loop must stay alive
                // even while every other animation is idle.
                if self.theme == Theme::Elemental
                    || !matches!(move_tick, AnimationTick::Ignored)
                    || !matches!(focus_tick, AnimationTick::Ignored)
                    || !matches!(reset_tick, AnimationTick::Ignored)
                    || !matches!(reveal_tick, AnimationTick::Ignored)
                {
                    event::Status::Captured
                } else {
                    event::Status::Ignored
                }
            }
            _ => event::Status::Ignored,
        };

        // Recalculate indices if rotation changed
        if rotation_changed {
            state.set_cached_indices(Self::calculate_indices(&state.rotation_4d));
        }
        if regenerate_stickers {
            let instances = sticker_instances_for_render(state);
            state.set_cached_sticker_instances(instances);
        }

        // A completed reveal's message goes first; a pending solve message
        // waits in the outbox for the next call (publishing makes iced
        // request another redraw, so there always is one).
        if let Some(message) = reveal_completed_message.or_else(|| state.solve_outbox.take()) {
            return Some(Action::publish(message));
        }

        match status {
            event::Status::Captured => Some(Action::request_redraw()),
            event::Status::Ignored => None,
        }
    }

    fn draw(
        &self,
        state: &Self::State,
        _cursor: mouse::Cursor,
        _bounds: Rectangle,
    ) -> Self::Primitive {
        // A move animation can rotate a facet's `face_normal_4d` away from
        // its static `face_id`'s `FACE_CENTERS` direction (see
        // `sticker_instances_for_render`'s `facet_axis_is_free` branch), so
        // the per-`face_id` visibility this is based on can't be trusted
        // while one is in progress - fall back to drawing every face and let
        // the vertex shader's own `is_face_visible` cull per-instance
        // instead.
        let face_visibility = if state.animating_move.is_some() {
            [true; 8]
        } else {
            visible_faces(&state.rotation_4d, self.viewer_distance)
        };

        let face_gap = state.reveal_gap_override.unwrap_or(self.face_gap);
        let face_gap_4d = state.reveal_gap_4d_override.unwrap_or(self.face_gap_4d);

        HypercubePrimitive {
            camera: state.camera.clone(),
            projection: state.projection,
            rotation_4d: state.rotation_4d,
            ui_controls: UiControls {
                sticker_scale: state.reveal_scale_override.unwrap_or(self.sticker_scale),
                face_gap,
                face_gap_4d,
                viewer_distance: self.viewer_distance,
                render_mode: self.render_mode,
                theme: self.theme,
                ground_truth_debug_face: ground_truth_debug_face(
                    self.fire_ground_truth_debug,
                    self.theme,
                    &state.cached_sticker_instances,
                    state.elapsed_seconds,
                ),
            },
            cached_indices: state.cached_indices.clone(),
            indices_generation: state.indices_generation,
            hovered_sticker: state.hovered_sticker,
            debug_instances: state.debug_instances.clone(),
            sticker_instances: state.cached_sticker_instances.clone(),
            sticker_generation: state.sticker_generation,
            visible_faces: face_visibility,
            depth_order: if self.theme == Theme::Elemental {
                depth_draw_order(
                    &state.cached_sticker_instances,
                    &face_visibility,
                    &state.rotation_4d,
                    &state.camera,
                    state.animating_move.is_none(),
                    state.reveal_scale_override.unwrap_or(self.sticker_scale),
                    face_gap,
                    face_gap_4d,
                    self.viewer_distance,
                )
            } else {
                Vec::new()
            },
            gizmo_vertices: self.build_gizmo_vertices(state, face_gap, face_gap_4d),
            elapsed_seconds: state.elapsed_seconds,
            snapshot_request: state.pending_snapshot.take(),
        }
    }
}

impl HypercubeShaderProgram {
    /// Builds this frame's rotation-axis gizmo geometry (see
    /// `gizmo_torus_vertices`), or an empty `Vec` when `show_gizmo_ring` is
    /// off, or when no focus animation, Shift+drag, or Reset animation is
    /// currently in progress (see
    /// `AnimatingFocus`/`ActiveShiftDrag`/`AnimatingReset`'s docs). Focus and
    /// drag each have a single well-defined rotation plane; Reset's
    /// accumulated orientation is generally a 4D "double rotation" with two
    /// independent invariant planes, so `reset_plane_and_phase` reuses the
    /// same combination pattern as `combined_drag_plane_and_phase` to
    /// collapse it to one ring - exact in the common case, an approximation
    /// for a fully generic double rotation (see that function's doc comment
    /// for the precise exactness boundary). Puzzle moves (`AnimatingMove`)
    /// are still excluded: a slab turn has no meaningful single 4D rotation
    /// plane at all to derive a ring from.
    /// `face_gap`/`face_gap_4d` are the same (reveal-override-resolved)
    /// values `draw()` already computed for this frame's stickers, used to
    /// size the ring via `gizmo_ring_radius`.
    fn build_gizmo_vertices(
        &self,
        state: &HypercubeShaderState,
        face_gap: f32,
        face_gap_4d: f32,
    ) -> Vec<GizmoVertex> {
        let mut vertices = Vec::new();
        if !self.show_gizmo_ring {
            return vertices;
        }
        let ring_radius = gizmo_ring_radius(face_gap, face_gap_4d);

        if let Some(animating) = &state.animating_focus {
            let t = if animating.duration.is_zero() {
                1.0
            } else {
                (animating.elapsed.as_secs_f32() / animating.duration.as_secs_f32()).clamp(0.0, 1.0)
            };
            let (invariant_plane, phase_angle) =
                focus_plane_and_phase(animating.plane, animating.total_angle, t);
            gizmo_torus_vertices(
                Vector4::zeros(),
                invariant_plane,
                phase_angle,
                ring_radius,
                GIZMO_FOCUS_PALETTE,
                GIZMO_FOCUS_MARKER_COLOR,
                GIZMO_ARROW_COLOR,
                self.viewer_distance,
                &mut vertices,
            );
        } else if let Some(drag) = state.active_shift_drag {
            let (right, up) = state.camera.right_and_up();
            let right_4d = Vector4::new(right.x, right.y, right.z, 0.0);
            let up_4d = Vector4::new(up.x, up.y, up.z, 0.0);

            if let Some((invariant_plane, phase_angle)) =
                combined_drag_plane_and_phase(drag, right_4d, up_4d)
            {
                gizmo_torus_vertices(
                    Vector4::zeros(),
                    invariant_plane,
                    phase_angle,
                    ring_radius,
                    GIZMO_DRAG_PALETTE,
                    GIZMO_DRAG_MARKER_COLOR,
                    GIZMO_ARROW_COLOR,
                    self.viewer_distance,
                    &mut vertices,
                );
            }
        } else if let Some(reset) = &state.animating_reset {
            let t = if reset.duration.is_zero() {
                1.0
            } else {
                (reset.elapsed.as_secs_f32() / reset.duration.as_secs_f32()).clamp(0.0, 1.0)
            };

            // Threshold is checked once against the animation's starting
            // angle (inside reset_plane_and_phase, called with the fixed
            // start_p/start_q), not re-checked against the live shrinking
            // angle each frame - so the ring doesn't pop away mid-animation
            // as it settles toward zero motion.
            if let Some((invariant_plane, base_phase_angle)) =
                reset_plane_and_phase(reset.start_p, reset.start_q)
            {
                let phase_angle = reset_phase_at(base_phase_angle, t);
                gizmo_torus_vertices(
                    Vector4::zeros(),
                    invariant_plane,
                    phase_angle,
                    ring_radius,
                    GIZMO_RESET_PALETTE,
                    GIZMO_RESET_MARKER_COLOR,
                    GIZMO_ARROW_COLOR,
                    self.viewer_distance,
                    &mut vertices,
                );
            }
        }

        vertices
    }
}

/// The gizmo ring's radius: half the on-screen distance between two
/// opposite face centers (e.g. green/blue) at the *current* gap settings.
/// Working through `math::project_face_point`'s exact formula for one
/// axis-aligned side face under an identity rotation (the same convention
/// `gizmo_torus_vertices` projects its own ring with): the 4D
/// depth-preserving push scales that face's own coordinate to exactly
/// `-face_gap_4d` (from `-1` at `face_gap_4d == 1.0`, i.e. no push), the
/// perspective scale stays exactly `1` throughout (`w` never leaves `0`),
/// and the 3D push then adds `-face_gap` along that same unit-length
/// direction - so the face's anchor sits at `-(face_gap + face_gap_4d)` and
/// its opposite (the exact negation) at `+(face_gap + face_gap_4d)`, twice
/// this radius apart.
fn gizmo_ring_radius(face_gap: f32, face_gap_4d: f32) -> f32 {
    face_gap + face_gap_4d
}

/// Recovers the one bit of rotation handedness `orthogonal_complement_plane`
/// discards (it's proven sign-invariant in its first argument), so
/// `gizmo_torus_vertices`'s `phase_angle` - and therefore its spinning
/// bands and marker arrows, both defined purely in terms of the returned
/// invariant plane `(p, q)` - tracks the actual direction of the `(a, b)`
/// rotation (a positive `angle_magnitude` rotates `a` toward `b` via
/// `create_4d_plane_rotation(a, b, angle_magnitude)`) instead of always
/// reading positive. `(a, b, p, q)` is an orthonormal 4-frame, so its
/// determinant is always exactly +-1; flipping `a`'s sign flips the
/// determinant but leaves `(p, q)` unchanged, which is exactly the missing
/// signal.
fn oriented_phase_angle(
    a: Vector4<f32>,
    b: Vector4<f32>,
    angle_magnitude: f32,
    (p, q): GizmoPlane,
) -> f32 {
    let sign = if Matrix4::from_columns(&[a, b, p, q]).determinant() < 0.0 {
        -1.0
    } else {
        1.0
    };
    sign * angle_magnitude
}

/// Computes the click-to-focus gizmo's invariant plane and signed phase
/// angle from an in-progress `AnimatingFocus`'s own rotation plane and
/// current eased progress. Mirrors `combined_drag_plane_and_phase` for the
/// focus case: `shortest_arc_plane`'s `total_angle` (an `acos`, always
/// `>= 0`) can't by itself carry which way the puzzle is actually turning,
/// so `oriented_phase_angle` recovers it from the full `(plane.0, plane.1,
/// invariant_plane)` orthonormal 4-frame's orientation.
fn focus_plane_and_phase(plane: GizmoPlane, total_angle: f32, t: f32) -> (GizmoPlane, f32) {
    let invariant_plane = orthogonal_complement_plane(plane.0, plane.1);
    let phase_angle =
        oriented_phase_angle(plane.0, plane.1, total_angle * ease(t), invariant_plane);
    (invariant_plane, phase_angle)
}

/// Combines a Shift+drag's two independent component rotations
/// (`process_4d_rotation`'s horizontal camera-right/W and vertical
/// camera-up/W planes) into the single ring "whose angle is determined by
/// the relative motion of the two original rings": to first order, applying
/// both simultaneously is exactly equivalent to one rotation about
/// `span(normalize(h*right + v*up), w)` by angle `sqrt(h^2 + v^2)` (their
/// generators add linearly since `right_4d ⊥ up_4d`). Returns `None` once
/// that combined angle is too small to show meaningfully (`GIZMO_MIN_DRAG_ANGLE`).
///
/// A drag along a single axis (the other component exactly zero)
/// degenerates to exactly that axis's own plane and angle, since the zero
/// component contributes nothing to `combined_4d`.
fn combined_drag_plane_and_phase(
    drag: ActiveShiftDrag,
    right_4d: Vector4<f32>,
    up_4d: Vector4<f32>,
) -> Option<(GizmoPlane, f32)> {
    let w_axis = Vector4::new(0.0, 0.0, 0.0, 1.0);

    let angle_magnitude = (drag.horizontal_angle.powi(2) + drag.vertical_angle.powi(2)).sqrt();
    if angle_magnitude <= GIZMO_MIN_DRAG_ANGLE {
        return None;
    }

    let combined_4d = (drag.horizontal_angle * right_4d + drag.vertical_angle * up_4d).normalize();
    let invariant_plane = orthogonal_complement_plane(combined_4d, w_axis);
    let phase_angle = oriented_phase_angle(combined_4d, w_axis, angle_magnitude, invariant_plane);
    Some((invariant_plane, phase_angle))
}

/// Combines a Reset animation's isoclinic pair (`AnimatingReset::start_p`/
/// `start_q`, from `math::decompose_so4`) into a single ring, via the same
/// combination pattern as `combined_drag_plane_and_phase`: each
/// quaternion's `scaled_axis()` (axis * angle, zero at identity) stands in
/// for one of that function's two independent signed angles, combined
/// linearly and paired with the fixed `w_axis` the same way. Returns `None`
/// once the combined angle is too small to show meaningfully (reuses
/// `GIZMO_MIN_DRAG_ANGLE`).
///
/// This is **exact** when the accumulated rotation is a simple rotation
/// whose active plane pairs some 3D axis with `w_axis` - the same
/// structural form every Shift+drag component (and this module's combined
/// drag ring) already assumes - and remains exact even for a genuine
/// double rotation *as long as its other invariant plane doesn't involve
/// `w` at all* (e.g. an accumulated `xw`-plus-`yz`-style rotation): the
/// `w`-free component cancels out of `q_gen - p_gen` identically, leaving
/// exactly the `w`-paired component's own angle. For a fully generic
/// double rotation whose invariant planes both mix `w` with other axes,
/// this is only an approximation, the same way `combined_drag_plane_and_phase`
/// is only a first-order approximation for a drag with two large
/// simultaneous components - there is no single plane that's exactly
/// invariant for a generic double rotation, so one ring can only ever be
/// approximately representative in that case.
fn reset_plane_and_phase(
    start_p: UnitQuaternion<f32>,
    start_q: UnitQuaternion<f32>,
) -> Option<(GizmoPlane, f32)> {
    let w_axis = Vector4::new(0.0, 0.0, 0.0, 1.0);

    let p_gen = start_p.scaled_axis();
    let q_gen = start_q.scaled_axis();
    let combined_gen = q_gen - p_gen;
    let angle_magnitude = combined_gen.norm() / 2.0;
    if angle_magnitude <= GIZMO_MIN_DRAG_ANGLE {
        return None;
    }

    let axis = combined_gen.normalize();
    let combined_4d = Vector4::new(axis.x, axis.y, axis.z, 0.0);
    let invariant_plane = orthogonal_complement_plane(combined_4d, w_axis);
    let phase_angle = oriented_phase_angle(combined_4d, w_axis, angle_magnitude, invariant_plane);
    Some((invariant_plane, phase_angle))
}

/// Scales a Reset gizmo's base plane/phase (from `reset_plane_and_phase`,
/// computed once from the animation's start quaternions) down to the
/// current eased progress `t`. Exact, not an approximation: slerping a
/// quaternion toward identity (`quat_slerp_exact`) moves along the
/// geodesic at constant angular velocity, so the interpolated quaternion's
/// axis never moves and its angle decays exactly linearly in `t` - so the
/// invariant plane derived from the fixed start quaternions stays valid
/// for the whole animation, and only the phase angle needs to shrink here.
fn reset_phase_at(base_phase_angle: f32, t: f32) -> f32 {
    base_phase_angle * (1.0 - ease(t))
}

impl HypercubeShaderProgram {
    /// Calculate the winding-corrected index buffer for all cube faces after
    /// 4D transformation and 3D projection. Shading normals are computed
    /// directly in the vertex shader from each instance's own basis instead
    /// (see `compute_world_normal` in shader.wgsl/normal_shader.wgsl).
    pub fn calculate_indices(rotation_4d: &nalgebra::Matrix4<f32>) -> Vec<u16> {
        let mut indices = Vec::with_capacity(288); // 36 indices * 8 4d faces

        for (face_idx, (face_center_4d, fixed_dim)) in
            FACE_CENTERS.iter().zip(FIXED_DIMS.iter()).enumerate()
        {
            // Transform 8 cube vertices to 3D
            let mut transformed_vertices = Vec::with_capacity(8);

            for (vertex_idx, vertex) in BASE_CUBE_VERTICES.iter().enumerate() {
                let local_vertex = Vector3::new(vertex[0], vertex[1], vertex[2]);
                let vertex_3d = project_cube_point(
                    local_vertex,
                    *face_center_4d,
                    *fixed_dim,
                    rotation_4d,
                    VIEWER_DISTANCE,
                    Vector4::zeros(),
                )
                .coords;

                log::debug!(
                    "{face_idx} * 8 + {vertex_idx} = {}",
                    face_idx * 8 + vertex_idx
                );
                transformed_vertices.push(vertex_3d);
            }

            let cube_center = transformed_vertices.iter().sum::<Vector3<f32>>() / 8.0;

            // Calculate one normal per cube face (6 faces), each spanning two
            // triangles (6 index slots); the two triangles of a face always
            // share a winding decision since they lie on the same plane.
            for (local_face_idx, mut face_indices) in VERTEX_NORMAL_INDICES
                .as_chunks::<6>()
                .0
                .iter()
                .copied()
                .enumerate()
            {
                let corner = |slot: usize| {
                    transformed_vertices[NORMAL_TO_BASE_INDICES[face_indices[slot] as usize]]
                };

                let v0 = corner(0);
                let v1 = corner(1);
                let v2 = corner(2);

                // Calculate triangle normal using cross product
                let edge1 = v1 - v0;
                let edge2 = v2 - v0;
                let mut normal = edge1.cross(&edge2);

                // Normalize and check for degenerate triangles
                let length = normal.norm();
                if length > 1e-6 {
                    normal /= length;
                } else {
                    // Degenerate triangle, use a default normal
                    log::warn!(
                        "Degenerate triangle detected for 4D face {face_idx} cube face {local_face_idx}: vertices {v0:?}, {v1:?}, {v2:?}"
                    );
                    normal = Vector3::new(0.0, 0.0, 1.0);
                }

                // Check winding order: the normal should point away from the
                // cube's own center toward this face's own center, not
                // toward/away from the world origin (the cube's center is at
                // an arbitrary offset from the origin, so that comparison is
                // unrelated to winding).
                //
                // A face's 4 unique corners sit at slots 0, 1, 2, 4 of its
                // 6-slot index chunk (slots 3 and 5 repeat slots 2 and 0 to
                // close the second triangle) — see VERTEX_NORMAL_INDICES /
                // NORMAL_TO_BASE_INDICES.
                let face_center = (corner(0) + corner(1) + corner(2) + corner(4)) / 4.0;
                if normal.dot(&(face_center - cube_center)) < 0.0 {
                    log::debug!(
                        "Bad winding order detected for 4D face {face_idx} cube face {local_face_idx}: normal {normal:?} points inward, flipping"
                    );
                    face_indices.swap(1, 2);
                    face_indices.swap(4, 5);
                }

                indices.extend(face_indices);
            }
        }

        indices
    }

    /// Handle mouse events for 3D navigation and 4D rotation
    fn handle_mouse_event(
        &self,
        state: &mut HypercubeShaderState,
        mouse_event: &mouse::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> event::Status {
        match mouse_event {
            mouse::Event::CursorMoved { .. } => {
                let Some(position) = cursor.position_in(bounds) else {
                    state.hovered_sticker = None;
                    return event::Status::Ignored;
                };

                // Calculate mouse delta for camera movement
                if let Some(last_pos) = state.last_mouse_pos {
                    let delta_x = position.x - last_pos.x;
                    let delta_y = position.y - last_pos.y;

                    // Apply mouse movement to camera or 4D rotation
                    if state.mouse_pressed {
                        if state.shift_pressed {
                            // 4D rotation, camera-relative. Skipped while
                            // `animating_reset` is already driving
                            // `rotation_4d`, rather than fighting it
                            // frame-by-frame.
                            if state.animating_reset.is_none() {
                                // A live drag takes over `rotation_4d` from
                                // any in-progress "center this face"
                                // animation the instant it starts writing to
                                // it, rather than fighting it frame-by-frame
                                // (a plain, non-Shift camera-orbit drag never
                                // reaches this branch, so it never touches
                                // `animating_focus` and the animation keeps
                                // playing on its own).
                                state.animating_focus = None;
                                let (right, up) = state.camera.right_and_up();
                                state.rotation_4d = process_4d_rotation(
                                    &state.rotation_4d,
                                    delta_x,
                                    delta_y,
                                    right,
                                    up,
                                );

                                let (angle_x, angle_y) =
                                    mouse_delta_to_plane_angles(delta_x, delta_y);
                                let drag =
                                    state.active_shift_drag.get_or_insert_with(Default::default);
                                drag.horizontal_angle += angle_x;
                                drag.vertical_angle += angle_y;
                            }
                        } else {
                            // 3D camera rotation
                            state
                                .camera_controller
                                .process_mouse_motion(delta_x, delta_y);
                        }
                    }
                }

                // Perform ray casting for sticker hover detection (only when not
                // dragging or mid-animation, since state has already moved past
                // what's currently rendering)
                if !state.mouse_pressed
                    && state.animating_move.is_none()
                    && state.solve_playback.is_none()
                {
                    self.update_hover(state, position, bounds);
                }

                state.last_mouse_pos = Some(position);
                return event::Status::Captured;
            }
            mouse::Event::ButtonPressed(button) => {
                if let Some(position) = cursor.position_in(bounds)
                    && *button == self.rotate_button.to_mouse_button()
                    && state.animating_reveal.is_none()
                {
                    // Reset the drag accumulator for this fresh gesture;
                    // whether it turns into a Shift-drag that takes over
                    // `rotation_4d` isn't known until `CursorMoved`.
                    state.active_shift_drag = None;
                    state.rotate_press = Some((position, state.hovered_sticker));
                    state.mouse_pressed = true;
                    return event::Status::Captured;
                }
                if cursor.position_in(bounds).is_some()
                    && *button == self.rotate_button.click_button()
                    && state.animating_move.is_none()
                    && state.animating_focus.is_none()
                    && state.animating_reset.is_none()
                    && state.animating_reveal.is_none()
                    && state.solve_playback.is_none()
                    && let Some(sticker_index) = state.hovered_sticker
                {
                    self.handle_facet_click(state, sticker_index);
                    return event::Status::Captured;
                }
            }
            mouse::Event::ButtonReleased(_) => {
                let was_dragging = state.mouse_pressed;
                if was_dragging {
                    state.mouse_pressed = false;
                    state.active_shift_drag = None;
                }

                if let Some((press_pos, sticker_at_press)) = state.rotate_press.take() {
                    self.handle_rotate_click(
                        state,
                        press_pos,
                        cursor.position_in(bounds),
                        sticker_at_press,
                    );
                    return event::Status::Captured;
                }

                if was_dragging {
                    return event::Status::Captured;
                }
            }
            mouse::Event::WheelScrolled { delta } => {
                if cursor.position_in(bounds).is_some() {
                    let scroll_delta = match delta {
                        mouse::ScrollDelta::Lines { y, .. } => *y,
                        mouse::ScrollDelta::Pixels { y, .. } => y * 0.01,
                    };
                    state.camera_controller.process_scroll(scroll_delta);
                    return event::Status::Captured;
                }
            }
            mouse::Event::CursorEntered => {
                // Handle cursor enter if needed
            }
            mouse::Event::CursorLeft => {
                // Clear hover state when cursor leaves the viewport
                state.hovered_sticker = None;
            }
        }

        event::Status::Ignored
    }

    /// Casts a ray from the given position and updates `hovered_sticker` and
    /// `debug_instances` on `state` with the result.
    fn update_hover(&self, state: &mut HypercubeShaderState, position: Point, bounds: Rectangle) {
        let mouse_ray = calculate_mouse_ray(position, bounds, &state.camera, &state.projection);
        let sticker_scale = state.reveal_scale_override.unwrap_or(self.sticker_scale);
        let face_gap = state.reveal_gap_override.unwrap_or(self.face_gap);
        let face_gap_4d = state.reveal_gap_4d_override.unwrap_or(self.face_gap_4d);

        let (hovered_sticker, debug_instances) = find_intersected_sticker(
            &mouse_ray,
            state,
            sticker_scale,
            face_gap,
            face_gap_4d,
            self.viewer_distance,
            self.aabb_mode,
        );
        state.hovered_sticker = hovered_sticker;
        state.debug_instances = debug_instances;
    }

    /// Resolves a rotate-button release into a click-vs-drag decision and,
    /// for a qualifying click, double-click detection. `press_pos`/
    /// `sticker_at_press` were captured when the button went down;
    /// `release_pos` is the cursor position now (`None` if it left the
    /// widget bounds). A release that isn't a same-spot click on a sticker
    /// (a real drag, or landing off any sticker) always clears any pending
    /// click, matching standard double-click behavior.
    fn handle_rotate_click(
        &self,
        state: &mut HypercubeShaderState,
        press_pos: Point,
        release_pos: Option<Point>,
        sticker_at_press: Option<usize>,
    ) {
        let is_click = release_pos.is_some_and(|release_pos| {
            let dx = release_pos.x - press_pos.x;
            let dy = release_pos.y - press_pos.y;
            (dx * dx + dy * dy).sqrt() < CLICK_DRAG_THRESHOLD_PX
        });

        let Some(sticker_index) = sticker_at_press.filter(|_| is_click) else {
            state.pending_face_click = None;
            return;
        };

        let face_id = FACET_TABLE[sticker_index].face_id;
        let now = Instant::now();
        let is_double_click = state
            .pending_face_click
            .is_some_and(|(last_time, last_face)| {
                last_face == face_id && now.duration_since(last_time) <= DOUBLE_CLICK_WINDOW
            });

        if is_double_click && state.animating_move.is_none() && state.animating_reset.is_none() {
            self.start_focus_animation(state, face_id);
            state.pending_face_click = None;
        } else {
            state.pending_face_click = Some((now, face_id));
        }
    }

    /// Starts an animation that reorients the puzzle in 4D so `face_id`'s
    /// normal ends up centered and facing the viewer. The target is
    /// `FACE_CENTERS[0]` (W=-1), not `FACE_CENTERS[7]` (W=+1): the latter is
    /// exactly the pole `is_face_visible` culls (see `math.rs`'s doc
    /// comment), so aiming there would make the double-clicked face vanish
    /// instead of centering it.
    fn start_focus_animation(&self, state: &mut HypercubeShaderState, face_id: usize) {
        let target = FACE_CENTERS[0];
        let current_normal = (state.rotation_4d * FACE_CENTERS[face_id]).normalize();
        let (u, v, total_angle) = shortest_arc_plane(current_normal, target);

        state.animating_focus = Some(AnimatingFocus {
            start_rotation: state.rotation_4d,
            plane: (u, v),
            total_angle,
            elapsed: Duration::ZERO,
            duration: Duration::from_millis(self.animation_duration_ms as u64),
        });
        state.last_redraw_instant = None;
    }

    /// Applies the move triggered by clicking the given facet, if any -
    /// non-actionable facets (cell-centers, the invisible center) are a
    /// no-op. A plain click always turns clockwise as viewed from beyond the
    /// clicked facet, looking back in along its own rotation axis
    /// (`moves::clockwise_sign`) - independent of the puzzle's current
    /// orientation or camera position. Shift reverses it to
    /// counterclockwise.
    fn handle_facet_click(&self, state: &mut HypercubeShaderState, sticker_index: usize) {
        let facet = &FACET_TABLE[sticker_index];
        if !facet.is_actionable {
            return;
        }

        let local_nonzero_count = facet.local_coords.iter().filter(|c| **c != 0).count();
        let magnitude = base_angle(local_nonzero_count);

        let sign = clockwise_sign(facet);
        let angle = if state.shift_pressed {
            -sign * magnitude
        } else {
            sign * magnitude
        };

        let pre_move_pieces = state.hypercube.pieces.clone();
        state
            .hypercube
            .apply_move(facet.axis, facet.side_sign, facet.local_coords, angle);

        state.animating_move = Some(AnimatingMove {
            side_axis: facet.axis,
            side_sign: facet.side_sign,
            local_coords: facet.local_coords,
            angle,
            pre_move_pieces,
            elapsed: Duration::ZERO,
            duration: Duration::from_millis(self.animation_duration_ms as u64),
        });
        state.last_redraw_instant = None;
        state.hovered_sticker = None;
    }

    /// Advances the in-progress move animation (if any) by the time elapsed
    /// since the last redraw, and requests another redraw if it isn't done
    /// yet - self-sustaining until the animation completes, at which point
    /// no further redraw is requested and the loop naturally stops.
    fn advance_animation(state: &mut HypercubeShaderState, delta: Duration) -> AnimationTick {
        let Some(animating) = state.animating_move.as_mut() else {
            return AnimationTick::Ignored;
        };

        animating.elapsed += delta;

        if animating.elapsed >= animating.duration {
            // Capped, so one long stall can't make every following solve
            // move complete the instant it starts.
            let overshoot = (animating.elapsed - animating.duration).min(SOLVE_MOVE_DURATION);
            state.move_overshoot = overshoot;
            state.animating_move = None;
            return AnimationTick::Completed;
        }

        AnimationTick::Running
    }

    /// Handles `SolveCommand::Start`: solves the live puzzle (a few
    /// milliseconds, so synchronously) and sets up playback, starting the
    /// first move now unless one is already animating - then it starts as
    /// soon as that one finishes. Returns the message to publish.
    fn start_solve(state: &mut HypercubeShaderState, generation: u64) -> Message {
        let solution = match solver::solve(&state.hypercube) {
            Ok(solution) => solution,
            Err(error) => {
                log::warn!("can't solve the puzzle: {error}");
                return Message::SolveEnded {
                    generation,
                    outcome: SolveOutcome::Failed(error),
                };
            }
        };
        let Some(first_stage) = solution.steps.first().map(|step| step.stage) else {
            return Message::SolveEnded {
                generation,
                outcome: SolveOutcome::AlreadySolved,
            };
        };
        log::info!(
            "solved in {} moves, merged from {} quarter turns",
            solution.steps.len(),
            solution.raw_twists
        );
        let total = solution.steps.len();
        state.solve_playback = Some(SolvePlayback {
            generation,
            queue: solution.steps.into(),
            total,
        });
        state.rotate_press = None;
        state.pending_face_click = None;
        state.hovered_sticker = None;
        state.debug_instances.clear();

        let waiting = Message::SolveProgress {
            generation,
            stage: first_stage,
            done: 0,
            total,
        };
        if state.animating_move.is_some() {
            return waiting;
        }
        state.move_overshoot = Duration::ZERO;
        state.last_redraw_instant = None;
        let message = Self::advance_solve_playback(state).unwrap_or(waiting);
        let instances = sticker_instances_for_render(state);
        state.set_cached_sticker_instances(instances);
        message
    }

    /// Starts the next move of the solve being played back (the caller has
    /// checked nothing is animating), or ends playback once its queue is
    /// empty. Returns the progress/end message to publish, or `None` if no
    /// solve is playing.
    fn advance_solve_playback(state: &mut HypercubeShaderState) -> Option<Message> {
        let playback = state.solve_playback.as_mut()?;
        let (generation, total) = (playback.generation, playback.total);
        let Some(step) = playback.queue.pop_front() else {
            state.solve_playback = None;
            return Some(Message::SolveEnded {
                generation,
                outcome: SolveOutcome::Completed { total },
            });
        };
        let done = total - playback.queue.len();

        let pre_move_pieces = state.hypercube.pieces.clone();
        state.hypercube.apply(&step.mv);
        state.animating_move = Some(AnimatingMove {
            side_axis: step.mv.side_axis,
            side_sign: step.mv.side_sign,
            local_coords: step.mv.local_coords,
            angle: step.mv.angle,
            pre_move_pieces,
            elapsed: std::mem::take(&mut state.move_overshoot),
            duration: SOLVE_MOVE_DURATION,
        });
        Some(Message::SolveProgress {
            generation,
            stage: step.stage,
            done,
            total,
        })
    }

    /// Advances an in-progress "center this face" animation (see
    /// `AnimatingFocus`) by `delta`, mirroring `advance_animation`'s shape.
    /// Unlike a move animation, this one has an externally visible effect
    /// every tick - it directly drives `rotation_4d` - rather than only
    /// affecting a separate per-frame sweep computation.
    fn advance_focus_animation(state: &mut HypercubeShaderState, delta: Duration) -> AnimationTick {
        let Some(animating) = state.animating_focus.as_mut() else {
            return AnimationTick::Ignored;
        };

        animating.elapsed += delta;

        let t = if animating.duration.is_zero() {
            1.0
        } else {
            (animating.elapsed.as_secs_f32() / animating.duration.as_secs_f32()).clamp(0.0, 1.0)
        };
        let (u, v) = animating.plane;
        let angle = animating.total_angle * ease(t);
        state.rotation_4d = create_4d_plane_rotation(u, v, angle) * animating.start_rotation;

        if animating.elapsed >= animating.duration {
            state.animating_focus = None;
            return AnimationTick::Completed;
        }

        AnimationTick::Running
    }

    /// Advances an in-progress reset animation (see `AnimatingReset`) by
    /// `delta`, slerping both quaternions toward identity and recomposing
    /// `rotation_4d`, mirroring `advance_focus_animation`'s shape.
    fn advance_reset_animation(state: &mut HypercubeShaderState, delta: Duration) -> AnimationTick {
        let Some(animating) = state.animating_reset.as_mut() else {
            return AnimationTick::Ignored;
        };

        animating.elapsed += delta;

        let t = if animating.duration.is_zero() {
            1.0
        } else {
            (animating.elapsed.as_secs_f32() / animating.duration.as_secs_f32()).clamp(0.0, 1.0)
        };
        let eased = ease(t);
        let identity = UnitQuaternion::identity();
        let p = quat_slerp_exact(animating.start_p, identity, eased);
        let q = quat_slerp_exact(animating.start_q, identity, eased);
        state.rotation_4d = compose_so4(p, q);

        if animating.elapsed >= animating.duration {
            state.rotation_4d = Matrix4::identity();
            state.animating_reset = None;
            return AnimationTick::Completed;
        }

        AnimationTick::Running
    }

    /// Advances an in-progress reveal/hide flourish (see `AnimatingReveal`)
    /// by `delta`. Drives `state.camera_controller.yaw` directly - picked up
    /// automatically by the unconditional `update_camera` call each tick -
    /// and updates the scale/gap overrides `draw()`/`update_hover` consult.
    /// On completion the overrides are left set to the exact target rather
    /// than cleared, so `Program::update`'s reconciliation check can drop
    /// them once `HypercubeApp` has caught up via the published completion
    /// message, avoiding a one-frame flash back to the pre-animation value.
    fn advance_reveal_animation(
        state: &mut HypercubeShaderState,
        delta: Duration,
    ) -> AnimationTick {
        let Some(animating) = state.animating_reveal.as_mut() else {
            return AnimationTick::Ignored;
        };

        animating.elapsed += delta;

        let t = if animating.duration.is_zero() {
            1.0
        } else {
            (animating.elapsed.as_secs_f32() / animating.duration.as_secs_f32()).clamp(0.0, 1.0)
        };
        let eased = ease(t);

        state.reveal_scale_override =
            Some(animating.start_scale + (animating.target_scale - animating.start_scale) * eased);
        state.reveal_gap_override =
            Some(animating.start_gap + (animating.target_gap - animating.start_gap) * eased);
        state.reveal_gap_4d_override = Some(
            animating.start_gap_4d + (animating.target_gap_4d - animating.start_gap_4d) * eased,
        );
        state.camera_controller.yaw =
            animating.start_yaw + (animating.target_yaw - animating.start_yaw) * eased;

        if animating.elapsed >= animating.duration {
            state.reveal_scale_override = Some(animating.target_scale);
            state.reveal_gap_override = Some(animating.target_gap);
            state.reveal_gap_4d_override = Some(animating.target_gap_4d);
            state.camera_controller.yaw = animating.target_yaw;
            state.animating_reveal = None;
            return AnimationTick::Completed;
        }

        AnimationTick::Running
    }

    /// Handle keyboard events for additional controls
    fn handle_keyboard_event(
        &self,
        state: &mut HypercubeShaderState,
        keyboard_event: &iced::keyboard::Event,
    ) -> event::Status {
        use iced::keyboard::Event;
        use iced::keyboard::{Key, key};
        match keyboard_event {
            Event::KeyPressed {
                key: Key::Named(key::Named::Shift),
                ..
            } => {
                state.shift_pressed = true;
                return event::Status::Captured;
            }
            Event::KeyReleased {
                key: Key::Named(key::Named::Shift),
                ..
            } => {
                state.shift_pressed = false;
                state.active_shift_drag = None;
                return event::Status::Captured;
            }
            _ => {}
        }

        event::Status::Ignored
    }
}

impl Default for HypercubeShaderState {
    fn default() -> Self {
        let mut camera = Camera {
            eye: nalgebra::Point3::new(0.0, 0.0, 15.0),
            target: nalgebra::Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
        };

        let camera_controller = CameraController::new(15.0);
        camera_controller.update_camera(&mut camera);

        let projection = Projection {
            aspect: 800.0 / 600.0,
            fovy: std::f32::consts::FRAC_PI_4,
            znear: 0.1,
            zfar: 100.0,
        };

        let rotation_4d = nalgebra::Matrix4::identity();
        let cached_indices = HypercubeShaderProgram::calculate_indices(&rotation_4d).into();
        let hypercube = Hypercube::solved();
        let cached_sticker_instances = generate_sticker_instances(&hypercube).into();

        Self {
            camera,
            camera_controller,
            projection,
            rotation_4d,
            mouse_pressed: false,
            last_mouse_pos: None,
            shift_pressed: false,
            cached_indices,
            indices_generation: 0,
            cached_sticker_instances,
            sticker_generation: 0,
            hovered_sticker: None,
            debug_instances: Vec::new(),
            hypercube,
            animating_move: None,
            animating_focus: None,
            animating_reset: None,
            animating_reveal: None,
            active_shift_drag: None,
            reveal_scale_override: None,
            reveal_gap_override: None,
            reveal_gap_4d_override: None,
            rotate_press: None,
            pending_face_click: None,
            last_redraw_instant: None,
            last_tick_instant: None,
            elapsed_seconds: 0.0,
            reset_generation: 0,
            random_moves_generation: 0,
            rng: fastrand::Rng::new(),
            reveal_generation: 0,
            save_generation: 0,
            load_generation: 0,
            save_snapshot_generation: 0,
            pending_snapshot: Cell::new(None),
            solve_command_generation: 0,
            solve_playback: None,
            solve_outbox: None,
            move_overshoot: Duration::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::FACE_CENTERS;
    use iced::widget::shader::Program;

    fn round_key(v: [f32; 4]) -> [i32; 4] {
        v.map(|x| (x * 1000.0).round() as i32)
    }

    /// At the end of a move, a rotated basis vector is `±` some world unit
    /// vector, matching `discrete_rotation`'s signed-permutation snap - but
    /// unlike position, the *sign* isn't independently meaningful here: the
    /// sticker mesh spans `[-s, s]` symmetrically along each of its 3 basis
    /// vectors, so any signed permutation of a facet's basis sweeps out the
    /// exact same rendered point set (e.g. a piece that spins in place on
    /// the turning face's own layer can land on a basis that's a nontrivial
    /// signed permutation of the static identity basis and still be
    /// pixel-identical, since the mesh has no per-face markings to reveal
    /// that it rotated). What *is* meaningful, and load-bearing for the
    /// sticker to render on the correct face at all, is which 3 of the 4
    /// world axes end up spanned - reduce to that set, dropping sign and
    /// order, for comparisons against the static post-move basis.
    fn basis_axis_set(basis: [[f32; 4]; 3]) -> Vec<usize> {
        let mut axes: Vec<usize> = basis
            .iter()
            .map(|v| {
                let (axis, value) = v
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
                    .expect("basis vector has 4 components");
                assert!(
                    value.abs() > 0.5,
                    "basis vector isn't close to a signed unit axis: {v:?}"
                );
                axis
            })
            .collect();
        axes.sort_unstable();
        axes
    }

    /// At `partial_angle = 0` (start of a move), the rotated basis must
    /// exactly reproduce the static pre-move basis - `rotate_local_position`
    /// at angle 0 is the identity, so this should hold bit-for-bit.
    #[test]
    fn animated_basis_matches_static_basis_at_start_of_move() {
        for side_axis in 0..4usize {
            for side_sign in [-1i8, 1] {
                for local_coords in [[1i8, 0, 0], [1, 1, 0], [1, 1, 1]] {
                    let nonzero = local_coords.iter().filter(|c| **c != 0).count();
                    let angle = base_angle(nonzero);

                    let pre_move = Hypercube::solved();
                    let state = HypercubeShaderState {
                        hypercube: pre_move.clone(),
                        animating_move: Some(AnimatingMove {
                            side_axis,
                            side_sign,
                            local_coords,
                            angle,
                            pre_move_pieces: pre_move.pieces.clone(),
                            elapsed: Duration::ZERO,
                            duration: Duration::from_millis(250),
                        }),
                        ..Default::default()
                    };

                    let instances = sticker_instances_for_render(&state);
                    for (facet, instance) in FACET_TABLE.iter().zip(instances.iter()) {
                        assert_eq!(
                            instance.basis, facet.basis,
                            "mismatch for side_axis={side_axis} side_sign={side_sign} \
                             local_coords={local_coords:?} piece_slot={} axis={}",
                            facet.piece_slot, facet.axis
                        );
                    }
                }
            }
        }
    }

    /// At `partial_angle = 0` (start of a move), every instance's
    /// `face_normal_4d` - the vector the shader culls against - must exactly
    /// reproduce the static pre-move `FACE_CENTERS[face_id]`, the same way
    /// `basis` does above.
    #[test]
    fn animated_face_normal_matches_static_face_center_at_start_of_move() {
        for side_axis in 0..4usize {
            for side_sign in [-1i8, 1] {
                for local_coords in [[1i8, 0, 0], [1, 1, 0], [1, 1, 1]] {
                    let nonzero = local_coords.iter().filter(|c| **c != 0).count();
                    let angle = base_angle(nonzero);

                    let pre_move = Hypercube::solved();
                    let state = HypercubeShaderState {
                        hypercube: pre_move.clone(),
                        animating_move: Some(AnimatingMove {
                            side_axis,
                            side_sign,
                            local_coords,
                            angle,
                            pre_move_pieces: pre_move.pieces.clone(),
                            elapsed: Duration::ZERO,
                            duration: Duration::from_millis(250),
                        }),
                        ..Default::default()
                    };

                    let instances = sticker_instances_for_render(&state);
                    for (facet, instance) in FACET_TABLE.iter().zip(instances.iter()) {
                        let expected: [f32; 4] = FACE_CENTERS[facet.face_id].into();
                        assert_eq!(
                            instance.face_normal_4d, expected,
                            "mismatch for side_axis={side_axis} side_sign={side_sign} \
                             local_coords={local_coords:?} piece_slot={} axis={}",
                            facet.piece_slot, facet.axis
                        );
                    }
                }
            }
        }
    }

    /// At `partial_angle` fully swept (end of a move, right before the snap
    /// to `generate_sticker_instances`), a facet whose own axis is one of
    /// the rotating slab's `free_axes` (`facet_axis_is_free`) must have its
    /// `face_normal_4d` fully onto its new face's `FACE_CENTERS`, not still
    /// mid-sweep or left on the old one - the counterpart to the start-of-move
    /// check above, checked directly per facet (not just as part of the
    /// set-based end-state comparison, which can't distinguish a wrong
    /// normal on one row from a legitimate match on another).
    #[test]
    fn animated_face_normal_matches_post_move_face_center_at_end_of_move() {
        use crate::moves::discrete_rotation;
        use crate::piece::face_id_for;

        for side_axis in 0..4usize {
            for side_sign in [-1i8, 1] {
                for local_coords in [[1i8, 0, 0], [1, 1, 0], [1, 1, 1]] {
                    let nonzero = local_coords.iter().filter(|c| **c != 0).count();
                    let angle = base_angle(nonzero);
                    let axes = free_axes(side_axis);
                    let (perm, sign) = discrete_rotation(local_coords, angle);
                    let mut inv_perm = [0usize; 3];
                    for slot in 0..3 {
                        inv_perm[perm[slot]] = slot;
                    }

                    let pre_move = Hypercube::solved();
                    let state = HypercubeShaderState {
                        hypercube: pre_move.clone(),
                        animating_move: Some(AnimatingMove {
                            side_axis,
                            side_sign,
                            local_coords,
                            angle,
                            pre_move_pieces: pre_move.pieces.clone(),
                            elapsed: Duration::from_millis(250),
                            duration: Duration::from_millis(250),
                        }),
                        ..Default::default()
                    };

                    let instances = sticker_instances_for_render(&state);
                    for (facet, instance) in FACET_TABLE.iter().zip(instances.iter()) {
                        let expected_face_id = if facet.axis == side_axis
                            || pre_move.pieces[facet.piece_slot].position[side_axis] != side_sign
                        {
                            facet.face_id
                        } else {
                            let p = axes
                                .iter()
                                .position(|&x| x == facet.axis)
                                .expect("facet.axis != side_axis must be one of axes");
                            let slot = inv_perm[p];
                            face_id_for(axes[slot], sign[slot] * facet.side_sign)
                        };
                        let expected: [f32; 4] = FACE_CENTERS[expected_face_id].into();
                        assert_eq!(
                            round_key(instance.face_normal_4d),
                            round_key(expected),
                            "mismatch for side_axis={side_axis} side_sign={side_sign} \
                             local_coords={local_coords:?} piece_slot={} axis={}",
                            facet.piece_slot,
                            facet.axis
                        );
                    }
                }
            }
        }
    }

    /// Position, kind, spanned basis axes, and face normal for one rendered
    /// row, used to compare animated vs. static render output as a set.
    type RenderRow = ([i32; 4], u32, Vec<usize>, [i32; 4]);

    /// At the end of an animation, the full set of rendered (position,
    /// kind) pairs must exactly match what the static post-move render
    /// would show - checked as a set (not a row-by-row comparison), since
    /// each animated row keeps its pre-move identity while sweeping to
    /// wherever its content ends up, which is a different GPU row than the
    /// static render uses for the same visual result.
    #[test]
    fn animated_end_state_matches_post_move_static_render_for_all_move_types() {
        for side_axis in 0..4usize {
            for side_sign in [-1i8, 1] {
                for local_coords in [
                    [1i8, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1],
                ] {
                    for direction in [1i8, -1] {
                        let nonzero = local_coords.iter().filter(|c| **c != 0).count();
                        let angle = base_angle(nonzero) * direction as f32;

                        let pre_move = Hypercube::solved();
                        let mut post_move = pre_move.clone();
                        post_move.apply_move(side_axis, side_sign, local_coords, angle);

                        let state = HypercubeShaderState {
                            hypercube: pre_move.clone(),
                            animating_move: Some(AnimatingMove {
                                side_axis,
                                side_sign,
                                local_coords,
                                angle,
                                pre_move_pieces: pre_move.pieces.clone(),
                                elapsed: Duration::from_millis(250),
                                duration: Duration::from_millis(250),
                            }),
                            ..Default::default()
                        };

                        let mut animated_end: Vec<RenderRow> = sticker_instances_for_render(&state)
                            .iter()
                            .map(|inst| {
                                (
                                    round_key(inst.position_4d),
                                    inst.kind,
                                    basis_axis_set(inst.basis),
                                    round_key(inst.face_normal_4d),
                                )
                            })
                            .collect();
                        let mut static_post: Vec<RenderRow> =
                            generate_sticker_instances(&post_move)
                                .iter()
                                .map(|inst| {
                                    (
                                        round_key(inst.position_4d),
                                        inst.kind,
                                        basis_axis_set(inst.basis),
                                        round_key(inst.face_normal_4d),
                                    )
                                })
                                .collect();
                        animated_end.sort_unstable();
                        static_post.sort_unstable();

                        assert_eq!(
                            animated_end, static_post,
                            "mismatch for side_axis={side_axis} side_sign={side_sign} \
                             local_coords={local_coords:?} direction={direction}"
                        );
                    }
                }
            }
        }
    }

    /// The set-based checks above can't catch a wrong basis on one row being
    /// masked by another row that legitimately has the same spanned axis
    /// set (face-swapping facets sharing a move come in groups). This pins
    /// down the one thing that's actually load-bearing per facet: at the end
    /// of a move, a facet whose own axis isn't `side_axis` swaps onto
    /// whichever new axis `apply_move`'s permutation sends it to - checked
    /// directly against `discrete_rotation` (already covered by its own
    /// tests in `moves.rs`), independent of the position/color machinery.
    #[test]
    fn animated_basis_flat_direction_matches_new_facet_axis_at_end_of_move() {
        use crate::moves::discrete_rotation;

        for side_axis in 0..4usize {
            for side_sign in [-1i8, 1] {
                for local_coords in [
                    [1i8, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1],
                ] {
                    for direction in [1i8, -1] {
                        let nonzero = local_coords.iter().filter(|c| **c != 0).count();
                        let angle = base_angle(nonzero) * direction as f32;
                        let axes = free_axes(side_axis);
                        let (perm, _sign) = discrete_rotation(local_coords, angle);
                        let mut inv_perm = [0usize; 3];
                        for slot in 0..3 {
                            inv_perm[perm[slot]] = slot;
                        }

                        let pre_move = Hypercube::solved();
                        let state = HypercubeShaderState {
                            hypercube: pre_move.clone(),
                            animating_move: Some(AnimatingMove {
                                side_axis,
                                side_sign,
                                local_coords,
                                angle,
                                pre_move_pieces: pre_move.pieces.clone(),
                                elapsed: Duration::from_millis(250),
                                duration: Duration::from_millis(250),
                            }),
                            ..Default::default()
                        };

                        let instances = sticker_instances_for_render(&state);
                        for (facet, instance) in FACET_TABLE.iter().zip(instances.iter()) {
                            // Facets on the turning face's own layer
                            // (`facet.axis == side_axis`) genuinely spin in
                            // place but their basis stays entirely within
                            // `axes`, spanning the same set regardless of
                            // rotation - nothing to pin down there, covered
                            // by the symmetric-mesh reasoning above instead.
                            if facet.axis == side_axis {
                                continue;
                            }
                            if pre_move.pieces[facet.piece_slot].position[side_axis] != side_sign {
                                continue;
                            }
                            let p = axes
                                .iter()
                                .position(|&x| x == facet.axis)
                                .expect("facet.axis != side_axis must be one of axes");
                            let new_axis = axes[inv_perm[p]];
                            let spanned = basis_axis_set(instance.basis);
                            assert!(
                                !spanned.contains(&new_axis),
                                "facet piece_slot={} axis={} should have swapped flat \
                                 direction onto axis {new_axis}, but basis still spans it: \
                                 {spanned:?} (side_axis={side_axis} side_sign={side_sign} \
                                 local_coords={local_coords:?} direction={direction})",
                                facet.piece_slot,
                                facet.axis
                            );
                        }
                    }
                }
            }
        }
    }

    /// A bumped `reset_generation` must leave the puzzle's piece arrangement
    /// untouched, cancel any in-progress move animation, and request a
    /// redraw - the mechanism a "Reset" button relies on to reach state
    /// owned by the shader widget's `Program::State`. Reset only resets the
    /// 4D orientation, not the puzzle.
    #[test]
    fn reset_generation_mismatch_cancels_animation_without_resetting_hypercube() {
        let mut state = HypercubeShaderState::default();
        assert_eq!(state.reset_generation, 0);

        let x = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let w = Vector4::new(0.0, 0.0, 0.0, 1.0);
        let starting_rotation = create_4d_plane_rotation(x, w, 1.2);
        state.rotation_4d = starting_rotation;

        let facet = FACET_TABLE
            .iter()
            .find(|f| f.is_actionable)
            .expect("at least one actionable facet exists");
        let nonzero = facet.local_coords.iter().filter(|c| **c != 0).count();
        let angle = base_angle(nonzero);
        let pre_move_pieces = state.hypercube.pieces.clone();
        state
            .hypercube
            .apply_move(facet.axis, facet.side_sign, facet.local_coords, angle);
        assert!(!state.hypercube.is_solved());
        let expected_pieces = state.hypercube.pieces.clone();
        state.animating_move = Some(AnimatingMove {
            side_axis: facet.axis,
            side_sign: facet.side_sign,
            local_coords: facet.local_coords,
            angle,
            pre_move_pieces,
            elapsed: Duration::ZERO,
            duration: Duration::from_millis(250),
        });

        let program = HypercubeShaderProgram::new(
            0.5,
            2.0,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            true,
            RotateButton::default(),
            250,
            1,
            0,
            0,
            0,
            false,
            0,
            0,
            None,
            0,
            0,
            SolveCommand::Stop,
        );

        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));
        let sticker_generation_before = state.sticker_generation;
        let action = program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );

        assert!(action.is_some(), "reset must request a redraw");
        assert!(
            !state.hypercube.is_solved(),
            "reset must not touch the puzzle's piece arrangement, only its 4D orientation"
        );
        assert_eq!(state.hypercube.pieces, expected_pieces);
        assert!(state.animating_move.is_none());
        assert_eq!(state.reset_generation, 1);
        assert_eq!(
            state.sticker_generation,
            sticker_generation_before + 1,
            "reset must regenerate cached sticker instances, not leave the \
             pre-reset (mid-move) cache in place"
        );
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(state.cached_sticker_instances.as_ref()),
            bytemuck::cast_slice::<_, u8>(&generate_sticker_instances(&state.hypercube)),
        );

        // The 4D orientation must not snap instantly - it's handed off to
        // `AnimatingReset` to animate back to identity over subsequent
        // ticks.
        assert!(
            (state.rotation_4d - starting_rotation).norm() < 1e-4,
            "rotation_4d must be untouched at the instant reset is pressed"
        );
        let animating = state
            .animating_reset
            .as_ref()
            .expect("reset must start a 4D orientation animation");
        let (expected_p, expected_q) = decompose_so4(&starting_rotation);
        assert!((animating.start_p.coords - expected_p.coords).norm() < 1e-4);
        assert!((animating.start_q.coords - expected_q.coords).norm() < 1e-4);
        assert_eq!(animating.duration, Duration::from_millis(250));
    }

    #[test]
    fn random_moves_generation_mismatch_applies_moves_and_cancels_animation() {
        let mut state = HypercubeShaderState {
            rng: fastrand::Rng::with_seed(1),
            ..Default::default()
        };
        assert_eq!(state.random_moves_generation, 0);

        let mut expected = Hypercube::solved();
        expected.apply_random_moves(3, &mut fastrand::Rng::with_seed(1));

        let program = HypercubeShaderProgram::new(
            0.5,
            2.0,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            true,
            RotateButton::default(),
            250,
            0,
            1,
            3,
            0,
            false,
            0,
            0,
            None,
            0,
            0,
            SolveCommand::Stop,
        );

        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));
        let sticker_generation_before = state.sticker_generation;
        let action = program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );

        assert!(action.is_some(), "random moves must request a redraw");
        assert_eq!(state.hypercube, expected);
        assert!(state.animating_move.is_none());
        assert!(state.animating_focus.is_none());
        assert_eq!(state.random_moves_generation, 1);
        assert_eq!(
            state.sticker_generation,
            sticker_generation_before + 1,
            "random moves must regenerate cached sticker instances"
        );
    }

    #[test]
    fn random_moves_generation_mismatch_with_zero_count_is_a_no_op_move_wise() {
        let mut state = HypercubeShaderState::default();
        assert!(state.hypercube.is_solved());

        let program = HypercubeShaderProgram::new(
            0.5,
            2.0,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            true,
            RotateButton::default(),
            250,
            0,
            1,
            0,
            0,
            false,
            0,
            0,
            None,
            0,
            0,
            SolveCommand::Stop,
        );

        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));
        let action = program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );

        assert!(action.is_some());
        assert!(state.hypercube.is_solved());
        assert_eq!(state.random_moves_generation, 1);
    }

    /// A `RedrawRequested` tick with nothing animating and no input must not
    /// regenerate or re-upload cached indices/sticker instances - the "camera
    /// at rest" case #3's generation-counter dirty-flag mechanism exists to
    /// skip.
    #[test]
    fn idle_redraw_does_not_bump_generations() {
        let mut state = HypercubeShaderState::default();
        let program = HypercubeShaderProgram::new(
            0.9,
            0.0,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            true,
            RotateButton::default(),
            250,
            state.reset_generation,
            state.random_moves_generation,
            0,
            state.reveal_generation,
            false,
            0,
            0,
            None,
            0,
            0,
            SolveCommand::Stop,
        );
        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));

        program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );

        assert_eq!(state.indices_generation, 0);
        assert_eq!(state.sticker_generation, 0);
    }

    /// Clicking an actionable facet starts a move animation and must
    /// regenerate (and bump the generation of) cached sticker instances -
    /// otherwise the render would keep showing the pre-move snapshot.
    #[test]
    fn clicking_actionable_facet_bumps_sticker_generation() {
        let mut state = HypercubeShaderState::default();
        let sticker_index = FACET_TABLE
            .iter()
            .position(|f| f.is_actionable)
            .expect("at least one actionable facet exists");
        state.hovered_sticker = Some(sticker_index);
        let sticker_generation_before = state.sticker_generation;

        let rotate_button = RotateButton::default();
        let program = HypercubeShaderProgram::new(
            0.9,
            0.0,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            true,
            rotate_button,
            250,
            state.reset_generation,
            state.random_moves_generation,
            0,
            state.reveal_generation,
            false,
            0,
            0,
            None,
            0,
            0,
            SolveCommand::Stop,
        );
        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));
        let cursor = mouse::Cursor::Available(Point::new(10.0, 10.0));

        program.update(
            &mut state,
            &Event::Mouse(mouse::Event::ButtonPressed(rotate_button.click_button())),
            bounds,
            cursor,
        );

        assert!(state.animating_move.is_some(), "click must start a move");
        assert_eq!(state.sticker_generation, sticker_generation_before + 1);
    }

    /// A "center this face" animation tick rotates `rotation_4d` every frame
    /// but never touches `Hypercube` state or `animating_move` - it must bump
    /// `indices_generation` (the winding-corrected index buffer depends on
    /// rotation) but leave `sticker_generation` untouched.
    #[test]
    fn focus_animation_tick_bumps_indices_generation_but_not_sticker_generation() {
        let mut state = HypercubeShaderState {
            animating_focus: Some(AnimatingFocus {
                start_rotation: Matrix4::identity(),
                plane: (
                    Vector4::new(1.0, 0.0, 0.0, 0.0),
                    Vector4::new(0.0, 1.0, 0.0, 0.0),
                ),
                total_angle: 90.0,
                elapsed: Duration::ZERO,
                duration: Duration::from_millis(250),
            }),
            ..Default::default()
        };
        let indices_generation_before = state.indices_generation;
        let sticker_generation_before = state.sticker_generation;

        let program = HypercubeShaderProgram::new(
            0.9,
            0.0,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            true,
            RotateButton::default(),
            250,
            state.reset_generation,
            state.random_moves_generation,
            0,
            state.reveal_generation,
            false,
            0,
            0,
            None,
            0,
            0,
            SolveCommand::Stop,
        );
        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));

        program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );

        assert_eq!(state.indices_generation, indices_generation_before + 1);
        assert_eq!(state.sticker_generation, sticker_generation_before);
    }

    /// At `t=0` a reveal animation's overrides/yaw must exactly reproduce
    /// its start values; once `elapsed` reaches `duration`, they must snap
    /// exactly to the target values and the animation must report Completed
    /// and clear itself.
    #[test]
    fn advance_reveal_animation_interpolates_and_completes() {
        let mut state = HypercubeShaderState {
            animating_reveal: Some(AnimatingReveal {
                start_scale: 0.9,
                target_scale: 0.98,
                start_gap: 0.0,
                target_gap: 1.0,
                start_gap_4d: 1.0,
                target_gap_4d: 2.0,
                start_yaw: 10.0,
                target_yaw: 10.0 + REVEAL_YAW_SPIN_DEGREES,
                elapsed: Duration::ZERO,
                duration: Duration::from_millis(1000),
            }),
            ..Default::default()
        };

        let tick = HypercubeShaderProgram::advance_reveal_animation(&mut state, Duration::ZERO);
        assert!(matches!(tick, AnimationTick::Running));
        assert_eq!(state.reveal_scale_override, Some(0.9));
        assert_eq!(state.reveal_gap_override, Some(0.0));
        assert_eq!(state.reveal_gap_4d_override, Some(1.0));
        assert_eq!(state.camera_controller.yaw, 10.0);

        let tick = HypercubeShaderProgram::advance_reveal_animation(
            &mut state,
            Duration::from_millis(2000),
        );
        assert!(matches!(tick, AnimationTick::Completed));
        assert_eq!(state.reveal_scale_override, Some(0.98));
        assert_eq!(state.reveal_gap_override, Some(1.0));
        assert_eq!(state.reveal_gap_4d_override, Some(2.0));
        assert_eq!(state.camera_controller.yaw, 10.0 + REVEAL_YAW_SPIN_DEGREES);
        assert!(state.animating_reveal.is_none());
    }

    #[test]
    fn advance_reset_animation_interpolates_and_completes() {
        let x = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let w = Vector4::new(0.0, 0.0, 0.0, 1.0);
        let start_rotation = create_4d_plane_rotation(x, w, 1.2);
        let (start_p, start_q) = decompose_so4(&start_rotation);

        let mut state = HypercubeShaderState {
            rotation_4d: start_rotation,
            animating_reset: Some(AnimatingReset {
                start_p,
                start_q,
                elapsed: Duration::ZERO,
                duration: Duration::from_millis(1000),
            }),
            ..Default::default()
        };

        let tick = HypercubeShaderProgram::advance_reset_animation(&mut state, Duration::ZERO);
        assert!(matches!(tick, AnimationTick::Running));
        assert!((state.rotation_4d - start_rotation).norm() < 1e-4);

        let tick = HypercubeShaderProgram::advance_reset_animation(
            &mut state,
            Duration::from_millis(2000),
        );
        assert!(matches!(tick, AnimationTick::Completed));
        assert!((state.rotation_4d - Matrix4::identity()).norm() < 1e-4);
        assert!(state.animating_reset.is_none());
    }

    /// A bumped `reveal_generation` with `revealed_target: true` must start
    /// an `AnimatingReveal` toward the secondary defaults from the program's
    /// current values, cancel any in-progress move/focus animation, and sync
    /// `state.reveal_generation`.
    #[test]
    fn reveal_generation_mismatch_starts_reveal_animation_and_cancels_others() {
        let mut state = HypercubeShaderState::default();
        assert_eq!(state.reveal_generation, 0);

        let facet = FACET_TABLE
            .iter()
            .find(|f| f.is_actionable)
            .expect("at least one actionable facet exists");
        let nonzero = facet.local_coords.iter().filter(|c| **c != 0).count();
        let angle = base_angle(nonzero);
        let pre_move_pieces = state.hypercube.pieces.clone();
        state.animating_move = Some(AnimatingMove {
            side_axis: facet.axis,
            side_sign: facet.side_sign,
            local_coords: facet.local_coords,
            angle,
            pre_move_pieces,
            elapsed: Duration::ZERO,
            duration: Duration::from_millis(250),
        });
        state.animating_focus = Some(AnimatingFocus {
            start_rotation: Matrix4::identity(),
            plane: (
                Vector4::new(1.0, 0.0, 0.0, 0.0),
                Vector4::new(0.0, 1.0, 0.0, 0.0),
            ),
            total_angle: 1.0,
            elapsed: Duration::ZERO,
            duration: Duration::from_millis(250),
        });
        state.camera_controller.yaw = 42.0;

        let program = HypercubeShaderProgram::new(
            0.9,
            0.0,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            true,
            RotateButton::default(),
            250,
            0,
            0,
            0,
            1,
            true,
            0,
            0,
            None,
            0,
            0,
            SolveCommand::Stop,
        );

        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));
        let action = program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );

        assert!(action.is_some(), "starting a reveal must request a redraw");
        assert!(state.animating_move.is_none());
        assert!(state.animating_focus.is_none());
        let animating = state
            .animating_reveal
            .as_ref()
            .expect("reveal animation should have started");
        assert_eq!(animating.start_scale, 0.9);
        assert_eq!(animating.start_gap, 0.0);
        assert_eq!(animating.start_yaw, 42.0);
        assert_eq!(animating.target_yaw, 42.0 + REVEAL_YAW_SPIN_DEGREES);
        assert_eq!(animating.target_scale, 1.0 - SECONDARY_STICKER_SCALE);
        assert_eq!(animating.target_gap, SECONDARY_FACE_GAP);
        assert_eq!(state.reveal_generation, 1);
    }

    /// With `revealed_target: false` (hiding), the animation must target the
    /// primary defaults instead of the secondary ones.
    #[test]
    fn reveal_generation_mismatch_targets_primary_defaults_when_hiding() {
        let mut state = HypercubeShaderState::default();
        let program = HypercubeShaderProgram::new(
            1.0 - SECONDARY_STICKER_SCALE,
            SECONDARY_FACE_GAP,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            true,
            RotateButton::default(),
            250,
            0,
            0,
            0,
            1,
            false,
            0,
            0,
            None,
            0,
            0,
            SolveCommand::Stop,
        );
        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));
        program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );

        let animating = state
            .animating_reveal
            .as_ref()
            .expect("reveal animation should have started");
        assert_eq!(animating.target_scale, 1.0 - PRIMARY_STICKER_SCALE);
        assert_eq!(animating.target_gap, PRIMARY_FACE_GAP);
    }

    /// The reveal overrides must stay put until the program's own
    /// sticker_scale/face_gap (i.e. `HypercubeApp`, once it has processed
    /// `RevealAnimationComplete`) match the stored target, then clear so a
    /// later manual slider drag isn't masked.
    #[test]
    fn reveal_override_clears_once_program_value_catches_up() {
        let mut state = HypercubeShaderState {
            reveal_scale_override: Some(0.9),
            reveal_gap_override: Some(1.0),
            ..Default::default()
        };
        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));

        let stale_program = HypercubeShaderProgram::new(
            0.5,
            0.0,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            true,
            RotateButton::default(),
            250,
            0,
            0,
            0,
            0,
            false,
            0,
            0,
            None,
            0,
            0,
            SolveCommand::Stop,
        );
        stale_program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );
        assert_eq!(state.reveal_scale_override, Some(0.9));
        assert_eq!(state.reveal_gap_override, Some(1.0));

        let caught_up_program = HypercubeShaderProgram::new(
            0.9,
            1.0,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            true,
            RotateButton::default(),
            250,
            0,
            0,
            0,
            0,
            false,
            0,
            0,
            None,
            0,
            0,
            SolveCommand::Stop,
        );
        caught_up_program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );
        assert_eq!(state.reveal_scale_override, None);
        assert_eq!(state.reveal_gap_override, None);
    }

    /// Once a reveal animation completes, `Program::update` must publish
    /// `Message::RevealAnimationComplete` carrying the raw-domain constants
    /// directly (not back-derived from the render-domain target), since a
    /// float round-trip through `1.0 - x` isn't guaranteed to reproduce the
    /// exact constant the override-reconciliation `==` check needs.
    #[test]
    fn reveal_completion_publishes_message_with_raw_domain_constants() {
        let mut state = HypercubeShaderState {
            animating_reveal: Some(AnimatingReveal {
                start_scale: 1.0 - PRIMARY_STICKER_SCALE,
                target_scale: 1.0 - SECONDARY_STICKER_SCALE,
                start_gap: PRIMARY_FACE_GAP,
                target_gap: SECONDARY_FACE_GAP,
                start_gap_4d: PRIMARY_FACE_GAP_4D,
                target_gap_4d: SECONDARY_FACE_GAP_4D,
                start_yaw: 0.0,
                target_yaw: REVEAL_YAW_SPIN_DEGREES,
                elapsed: REVEAL_ANIMATION_DURATION + Duration::from_millis(100),
                duration: REVEAL_ANIMATION_DURATION,
            }),
            ..Default::default()
        };
        let program = HypercubeShaderProgram::new(
            1.0 - PRIMARY_STICKER_SCALE,
            PRIMARY_FACE_GAP,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            true,
            RotateButton::default(),
            250,
            0,
            0,
            0,
            0,
            true,
            0,
            0,
            None,
            0,
            0,
            SolveCommand::Stop,
        );
        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));
        let action = program.update(
            &mut state,
            &Event::Window(iced::window::Event::RedrawRequested(Instant::now())),
            bounds,
            mouse::Cursor::Unavailable,
        );

        let (message, ..) = action
            .expect("a completed reveal must produce an action")
            .into_inner();
        match message.expect("a completed reveal must publish a message") {
            Message::RevealAnimationComplete {
                final_scale,
                final_gap,
                final_gap_4d,
            } => {
                assert_eq!(final_scale, SECONDARY_STICKER_SCALE);
                assert_eq!(final_gap, SECONDARY_FACE_GAP);
                assert_eq!(final_gap_4d, SECONDARY_FACE_GAP_4D);
            }
            other => panic!("expected RevealAnimationComplete, got {other:?}"),
        }
    }

    /// A program at the given solve command, with everything else idle and
    /// in sync with `state`.
    fn solve_program(
        state: &HypercubeShaderState,
        generation: u64,
        command: SolveCommand,
    ) -> HypercubeShaderProgram {
        HypercubeShaderProgram::new(
            0.9,
            0.0,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            true,
            RotateButton::default(),
            250,
            state.reset_generation,
            state.random_moves_generation,
            0,
            state.reveal_generation,
            false,
            state.save_generation,
            state.load_generation,
            None,
            state.save_snapshot_generation,
            generation,
            command,
        )
    }

    fn redraw_at(now: Instant) -> Event {
        Event::Window(iced::window::Event::RedrawRequested(now))
    }

    fn published(action: Option<Action<Message>>) -> Option<Message> {
        action.and_then(|action| action.into_inner().0)
    }

    fn scrambled_state(seed: u64) -> HypercubeShaderState {
        let mut state = HypercubeShaderState::default();
        state
            .hypercube
            .apply_random_moves(3, &mut fastrand::Rng::with_seed(seed));
        state
    }

    fn test_bounds() -> Rectangle {
        Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0))
    }

    #[test]
    fn solve_start_begins_the_first_move_and_publishes_progress() {
        let mut state = scrambled_state(1);
        let program = solve_program(&state, 1, SolveCommand::Start);
        let message = published(program.update(
            &mut state,
            &redraw_at(Instant::now()),
            test_bounds(),
            mouse::Cursor::Unavailable,
        ));

        match message {
            Some(Message::SolveProgress {
                generation: 1,
                done: 1,
                total,
                ..
            }) => assert!(total > 0),
            other => panic!("expected progress for move 1, got {other:?}"),
        }
        let animating = state
            .animating_move
            .as_ref()
            .expect("the first solve move must start right away");
        assert_eq!(animating.duration, SOLVE_MOVE_DURATION);
        assert!(state.solve_playback.is_some());
    }

    #[test]
    fn solve_start_on_a_solved_cube_publishes_already_solved() {
        let mut state = HypercubeShaderState::default();
        let program = solve_program(&state, 1, SolveCommand::Start);
        let message = published(program.update(
            &mut state,
            &redraw_at(Instant::now()),
            test_bounds(),
            mouse::Cursor::Unavailable,
        ));

        assert!(matches!(
            message,
            Some(Message::SolveEnded {
                generation: 1,
                outcome: SolveOutcome::AlreadySolved
            })
        ));
        assert!(state.solve_playback.is_none());
        assert!(state.animating_move.is_none());
    }

    #[test]
    fn solve_playback_runs_to_solved_and_reports_completion() {
        let mut state = scrambled_state(2);
        let program = solve_program(&state, 1, SolveCommand::Start);
        let start = Instant::now();
        let first = published(program.update(
            &mut state,
            &redraw_at(start),
            test_bounds(),
            mouse::Cursor::Unavailable,
        ));
        let Some(Message::SolveProgress { total, .. }) = first else {
            panic!("expected progress, got {first:?}");
        };

        // 50ms frames against 40ms moves: one move completes (and the next
        // starts) every frame.
        let mut ended = None;
        for frame in 1..=(total as u32 * 2 + 10) {
            let now = start + Duration::from_millis(50) * frame;
            if let Some(Message::SolveEnded { outcome, .. }) = published(program.update(
                &mut state,
                &redraw_at(now),
                test_bounds(),
                mouse::Cursor::Unavailable,
            )) {
                ended = Some(outcome);
                break;
            }
        }

        assert_eq!(ended, Some(SolveOutcome::Completed { total }));
        assert!(state.hypercube.is_solved());
        assert!(state.solve_playback.is_none());
    }

    #[test]
    fn solve_stop_cancels_playback_but_lets_the_move_in_flight_finish() {
        let mut state = scrambled_state(3);
        let now = Instant::now();
        solve_program(&state, 1, SolveCommand::Start).update(
            &mut state,
            &redraw_at(now),
            test_bounds(),
            mouse::Cursor::Unavailable,
        );
        assert!(state.solve_playback.is_some());

        let message = published(solve_program(&state, 2, SolveCommand::Stop).update(
            &mut state,
            &redraw_at(now),
            test_bounds(),
            mouse::Cursor::Unavailable,
        ));

        assert!(message.is_none());
        assert!(state.solve_playback.is_none());
        assert!(state.solve_outbox.is_none());
        assert!(state.animating_move.is_some());
    }

    #[test]
    fn reset_random_moves_and_a_successful_load_cancel_solve_playback() {
        // (reset, random moves, load, pending load, expect playback left)
        let cases = [
            (1, 0, 0, None, false),
            (0, 1, 0, None, false),
            (0, 0, 1, Some(Hypercube::solved()), false),
            (0, 0, 1, None, true),
        ];
        for (reset, random, load, pending, survives) in cases {
            let mut state = scrambled_state(4);
            solve_program(&state, 1, SolveCommand::Start).update(
                &mut state,
                &redraw_at(Instant::now()),
                test_bounds(),
                mouse::Cursor::Unavailable,
            );
            let program = HypercubeShaderProgram::new(
                0.9,
                0.0,
                1.0,
                VIEWER_DISTANCE,
                RenderMode::Standard,
                Theme::Classic,
                AABBMode::None,
                false,
                true,
                RotateButton::default(),
                250,
                reset,
                random,
                0,
                0,
                false,
                0,
                load,
                pending,
                0,
                1,
                SolveCommand::Start,
            );
            program.update(
                &mut state,
                &redraw_at(Instant::now()),
                test_bounds(),
                mouse::Cursor::Unavailable,
            );
            assert_eq!(
                state.solve_playback.is_some(),
                survives,
                "reset={reset} random={random} load={load}"
            );
        }
    }

    #[test]
    fn turn_clicks_are_ignored_during_solve_playback() {
        let mut state = scrambled_state(5);
        let program = solve_program(&state, 1, SolveCommand::Start);
        program.update(
            &mut state,
            &redraw_at(Instant::now()),
            test_bounds(),
            mouse::Cursor::Unavailable,
        );
        // Isolate the playback guard from the move-in-flight one.
        state.animating_move = None;
        state.hovered_sticker = FACET_TABLE.iter().position(|f| f.is_actionable);
        let before = state.hypercube.clone();

        program.update(
            &mut state,
            &Event::Mouse(mouse::Event::ButtonPressed(
                RotateButton::default().click_button(),
            )),
            test_bounds(),
            mouse::Cursor::Available(Point::new(10.0, 10.0)),
        );

        assert!(state.animating_move.is_none());
        assert_eq!(state.hypercube, before);
    }

    #[test]
    fn reveal_completion_and_solve_end_on_one_tick_are_both_published() {
        let mut state = HypercubeShaderState {
            animating_reveal: Some(AnimatingReveal {
                start_scale: 1.0 - PRIMARY_STICKER_SCALE,
                target_scale: 1.0 - SECONDARY_STICKER_SCALE,
                start_gap: PRIMARY_FACE_GAP,
                target_gap: SECONDARY_FACE_GAP,
                start_gap_4d: PRIMARY_FACE_GAP_4D,
                target_gap_4d: SECONDARY_FACE_GAP_4D,
                start_yaw: 0.0,
                target_yaw: REVEAL_YAW_SPIN_DEGREES,
                elapsed: REVEAL_ANIMATION_DURATION + Duration::from_millis(100),
                duration: REVEAL_ANIMATION_DURATION,
            }),
            solve_command_generation: 1,
            solve_playback: Some(SolvePlayback {
                generation: 1,
                queue: VecDeque::new(),
                total: 7,
            }),
            ..Default::default()
        };
        let program = HypercubeShaderProgram::new(
            1.0 - PRIMARY_STICKER_SCALE,
            PRIMARY_FACE_GAP,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            true,
            RotateButton::default(),
            250,
            0,
            0,
            0,
            0,
            true,
            0,
            0,
            None,
            0,
            1,
            SolveCommand::Start,
        );
        let now = Instant::now();
        let first = published(program.update(
            &mut state,
            &redraw_at(now),
            test_bounds(),
            mouse::Cursor::Unavailable,
        ));
        let second = published(program.update(
            &mut state,
            &redraw_at(now),
            test_bounds(),
            mouse::Cursor::Unavailable,
        ));

        assert!(matches!(
            first,
            Some(Message::RevealAnimationComplete { .. })
        ));
        assert!(matches!(
            second,
            Some(Message::SolveEnded {
                generation: 1,
                outcome: SolveOutcome::Completed { total: 7 }
            })
        ));
    }

    /// Camera-drag orbit start and facet turn-clicks must both be ignored
    /// while a reveal/hide flourish is playing, per the "locked cutscene"
    /// requirement.
    #[test]
    fn mouse_input_is_ignored_while_reveal_animation_plays() {
        let mut state = HypercubeShaderState {
            animating_reveal: Some(AnimatingReveal {
                start_scale: 0.9,
                target_scale: 0.98,
                start_gap: 0.0,
                target_gap: 1.0,
                start_gap_4d: 1.0,
                target_gap_4d: 2.0,
                start_yaw: 0.0,
                target_yaw: REVEAL_YAW_SPIN_DEGREES,
                elapsed: Duration::ZERO,
                duration: REVEAL_ANIMATION_DURATION,
            }),
            ..Default::default()
        };
        let rotate_button = RotateButton::default();
        let program = HypercubeShaderProgram::new(
            0.9,
            0.0,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            true,
            rotate_button,
            250,
            0,
            0,
            0,
            0,
            true,
            0,
            0,
            None,
            0,
            0,
            SolveCommand::Stop,
        );
        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));
        let cursor = mouse::Cursor::Available(Point::new(10.0, 10.0));

        program.update(
            &mut state,
            &Event::Mouse(mouse::Event::ButtonPressed(rotate_button.to_mouse_button())),
            bounds,
            cursor,
        );
        assert!(
            !state.mouse_pressed,
            "camera drag must not start during the reveal flourish"
        );
        assert!(state.rotate_press.is_none());

        state.hovered_sticker = Some(0);
        let pieces_before = state.hypercube.pieces.clone();
        program.update(
            &mut state,
            &Event::Mouse(mouse::Event::ButtonPressed(rotate_button.click_button())),
            bounds,
            cursor,
        );
        assert_eq!(
            state.hypercube.pieces, pieces_before,
            "facet turn must not apply during the reveal flourish"
        );
    }

    /// A plain (non-Shift) camera-orbit drag only ever touches
    /// `camera_controller` - it must not cancel an in-progress "center this
    /// face" animation, which keeps playing on its own via
    /// `advance_focus_animation`.
    #[test]
    fn non_shift_drag_preserves_focus_animation_and_still_orbits_camera() {
        let mut state = HypercubeShaderState {
            animating_focus: Some(AnimatingFocus {
                start_rotation: Matrix4::identity(),
                plane: (
                    Vector4::new(1.0, 0.0, 0.0, 0.0),
                    Vector4::new(0.0, 1.0, 0.0, 0.0),
                ),
                total_angle: 90.0,
                elapsed: Duration::ZERO,
                duration: Duration::from_millis(250),
            }),
            shift_pressed: false,
            last_mouse_pos: Some(Point::new(10.0, 10.0)),
            ..Default::default()
        };
        let yaw_before = state.camera_controller.yaw;
        let pitch_before = state.camera_controller.pitch;

        let rotate_button = RotateButton::default();
        let program = HypercubeShaderProgram::new(
            0.9,
            0.0,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            true,
            rotate_button,
            250,
            0,
            0,
            0,
            0,
            true,
            0,
            0,
            None,
            0,
            0,
            SolveCommand::Stop,
        );
        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));

        program.update(
            &mut state,
            &Event::Mouse(mouse::Event::ButtonPressed(rotate_button.to_mouse_button())),
            bounds,
            mouse::Cursor::Available(Point::new(10.0, 10.0)),
        );
        assert!(state.mouse_pressed);
        assert!(
            state.animating_focus.is_some(),
            "a plain camera-orbit press must not cancel the focus animation"
        );

        program.update(
            &mut state,
            &Event::Mouse(mouse::Event::CursorMoved {
                position: iced::Point::new(30.0, 20.0),
            }),
            bounds,
            mouse::Cursor::Available(Point::new(30.0, 20.0)),
        );

        assert!(
            state.animating_focus.is_some(),
            "the focus animation must keep playing through a non-Shift drag"
        );
        assert_ne!(state.camera_controller.yaw, yaw_before);
        assert_ne!(state.camera_controller.pitch, pitch_before);
    }

    /// A Shift+drag writes `rotation_4d` directly, so it must take over from,
    /// and cancel, any in-progress "center this face" animation the instant
    /// it starts moving, avoiding the frame-by-frame fight the two would
    /// otherwise have over `rotation_4d`.
    #[test]
    fn shift_drag_cancels_focus_animation() {
        let mut state = HypercubeShaderState {
            animating_focus: Some(AnimatingFocus {
                start_rotation: Matrix4::identity(),
                plane: (
                    Vector4::new(1.0, 0.0, 0.0, 0.0),
                    Vector4::new(0.0, 1.0, 0.0, 0.0),
                ),
                total_angle: 90.0,
                elapsed: Duration::ZERO,
                duration: Duration::from_millis(250),
            }),
            shift_pressed: true,
            last_mouse_pos: Some(Point::new(10.0, 10.0)),
            ..Default::default()
        };

        let rotate_button = RotateButton::default();
        let program = HypercubeShaderProgram::new(
            0.9,
            0.0,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            true,
            rotate_button,
            250,
            0,
            0,
            0,
            0,
            true,
            0,
            0,
            None,
            0,
            0,
            SolveCommand::Stop,
        );
        let bounds = Rectangle::new(Point::ORIGIN, iced::Size::new(800.0, 600.0));

        program.update(
            &mut state,
            &Event::Mouse(mouse::Event::ButtonPressed(rotate_button.to_mouse_button())),
            bounds,
            mouse::Cursor::Available(Point::new(10.0, 10.0)),
        );
        assert!(state.animating_focus.is_some());

        program.update(
            &mut state,
            &Event::Mouse(mouse::Event::CursorMoved {
                position: iced::Point::new(30.0, 20.0),
            }),
            bounds,
            mouse::Cursor::Available(Point::new(30.0, 20.0)),
        );

        assert!(
            state.animating_focus.is_none(),
            "a live Shift-drag must cancel the focus animation"
        );
        assert!(state.active_shift_drag.is_some());
    }

    #[test]
    fn combined_drag_degenerates_to_single_axis_case() {
        let right_4d = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let up_4d = Vector4::new(0.0, 1.0, 0.0, 0.0);
        let w_axis = Vector4::new(0.0, 0.0, 0.0, 1.0);

        let drag = ActiveShiftDrag {
            horizontal_angle: 0.37,
            vertical_angle: 0.0,
        };
        let (plane, phase_angle) = combined_drag_plane_and_phase(drag, right_4d, up_4d)
            .expect("angle exceeds GIZMO_MIN_DRAG_ANGLE");
        let expected_plane = orthogonal_complement_plane(right_4d, w_axis);

        assert!((phase_angle - 0.37).abs() < 1e-6);
        assert!((plane.0 - expected_plane.0).norm() < 1e-6);
        assert!((plane.1 - expected_plane.1).norm() < 1e-6);
    }

    #[test]
    fn combined_drag_below_threshold_is_hidden() {
        let right_4d = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let up_4d = Vector4::new(0.0, 1.0, 0.0, 0.0);
        let drag = ActiveShiftDrag {
            horizontal_angle: GIZMO_MIN_DRAG_ANGLE * 0.5,
            vertical_angle: 0.0,
        };
        assert!(combined_drag_plane_and_phase(drag, right_4d, up_4d).is_none());
    }

    #[test]
    fn oriented_phase_angle_flips_sign_when_first_argument_negates() {
        let u = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let w = Vector4::new(0.0, 0.0, 0.0, 1.0);
        let complement = orthogonal_complement_plane(u, w);

        let positive = oriented_phase_angle(u, w, 0.7, complement);
        let negative = oriented_phase_angle(-u, w, 0.7, complement);

        assert!((positive + negative).abs() < 1e-6);
        assert!(positive.abs() > 1e-6);
    }

    /// Encodes the reported bug directly: from the default (identity)
    /// orientation, centering one face of an opposite pair (e.g. green)
    /// must creep its focus gizmo's arrows in the exact reverse direction
    /// from centering the other (blue), since `create_4d_plane_rotation(-u,
    /// v, angle) == create_4d_plane_rotation(u, v, -angle)` and the two
    /// faces' normals are exact negations of each other.
    #[test]
    fn focus_animation_for_opposite_faces_has_opposite_phase_sign() {
        let target = FACE_CENTERS[0];
        for &(a, b) in &[(1usize, 6usize), (2, 5), (3, 4)] {
            let (u_a, v_a, angle_a) = shortest_arc_plane(FACE_CENTERS[a], target);
            let (u_b, v_b, angle_b) = shortest_arc_plane(FACE_CENTERS[b], target);

            let (_, phase_a) = focus_plane_and_phase((u_a, v_a), angle_a, 1.0);
            let (_, phase_b) = focus_plane_and_phase((u_b, v_b), angle_b, 1.0);

            assert!(
                (phase_a + phase_b).abs() < 1e-5,
                "faces {a}/{b}: expected opposite phase angles, got {phase_a} and {phase_b}"
            );
            assert!(
                phase_a.abs() > 1e-5,
                "faces {a}/{b}: phase angle collapsed to zero"
            );
        }
    }

    #[test]
    fn reset_plane_and_phase_matches_pure_single_plane_rotation() {
        let x = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let w = Vector4::new(0.0, 0.0, 0.0, 1.0);
        let angle = 0.5_f32;
        let (start_p, start_q) = decompose_so4(&create_4d_plane_rotation(x, w, angle));

        let (plane, phase_angle) =
            reset_plane_and_phase(start_p, start_q).expect("angle exceeds threshold");
        let expected_plane = orthogonal_complement_plane(x, w);

        assert!((phase_angle - angle).abs() < 1e-4);
        assert!((plane.0 - expected_plane.0).norm() < 1e-4);
        assert!((plane.1 - expected_plane.1).norm() < 1e-4);
    }

    #[test]
    fn reset_plane_and_phase_ignores_w_free_component_of_double_rotation() {
        let x = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let y = Vector4::new(0.0, 1.0, 0.0, 0.0);
        let z = Vector4::new(0.0, 0.0, 1.0, 0.0);
        let w = Vector4::new(0.0, 0.0, 0.0, 1.0);
        let xw_angle = 1.2_f32;

        for &yz_angle in &[0.0_f32, 0.7, -0.4] {
            let m =
                create_4d_plane_rotation(x, w, xw_angle) * create_4d_plane_rotation(y, z, yz_angle);
            let (start_p, start_q) = decompose_so4(&m);
            let (_, phase_angle) =
                reset_plane_and_phase(start_p, start_q).expect("angle exceeds threshold");
            assert!(
                (phase_angle - xw_angle).abs() < 1e-4,
                "yz_angle={yz_angle}: expected {xw_angle}, got {phase_angle}"
            );
        }
    }

    #[test]
    fn reset_plane_and_phase_below_threshold_is_hidden() {
        let identity = UnitQuaternion::identity();
        assert!(reset_plane_and_phase(identity, identity).is_none());
    }

    #[test]
    fn reset_phase_scales_linearly_with_eased_progress() {
        let x = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let y = Vector4::new(0.0, 1.0, 0.0, 0.0);
        let z = Vector4::new(0.0, 0.0, 1.0, 0.0);
        let w = Vector4::new(0.0, 0.0, 0.0, 1.0);
        let m = create_4d_plane_rotation(x, w, 1.2) * create_4d_plane_rotation(y, z, -0.4);
        let (start_p, start_q) = decompose_so4(&m);
        let (_, base_phase_angle) =
            reset_plane_and_phase(start_p, start_q).expect("angle exceeds threshold");

        let identity = UnitQuaternion::identity();
        for &t in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let eased = ease(t);
            let live_p = quat_slerp_exact(start_p, identity, eased);
            let live_q = quat_slerp_exact(start_q, identity, eased);
            let expected = reset_plane_and_phase(live_p, live_q)
                .map(|(_, phase)| phase)
                .unwrap_or(0.0);
            let actual = reset_phase_at(base_phase_angle, t);
            assert!(
                (actual - expected).abs() < 1e-3,
                "t={t}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn build_gizmo_vertices_produces_ring_during_reset_animation() {
        let mut state = HypercubeShaderState::default();
        let x = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let w = Vector4::new(0.0, 0.0, 0.0, 1.0);
        let (start_p, start_q) = decompose_so4(&create_4d_plane_rotation(x, w, 1.0));
        state.animating_reset = Some(AnimatingReset {
            start_p,
            start_q,
            elapsed: Duration::ZERO,
            duration: Duration::from_millis(250),
        });

        let program = HypercubeShaderProgram::new(
            0.5,
            2.0,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            true,
            RotateButton::default(),
            250,
            1,
            0,
            0,
            0,
            false,
            0,
            0,
            None,
            0,
            0,
            SolveCommand::Stop,
        );

        let vertices = program.build_gizmo_vertices(&state, 0.0, 1.0);
        assert!(!vertices.is_empty());
    }

    #[test]
    fn build_gizmo_vertices_hides_ring_when_toggle_is_off() {
        let mut state = HypercubeShaderState::default();
        let x = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let w = Vector4::new(0.0, 0.0, 0.0, 1.0);
        let (start_p, start_q) = decompose_so4(&create_4d_plane_rotation(x, w, 1.0));
        state.animating_reset = Some(AnimatingReset {
            start_p,
            start_q,
            elapsed: Duration::ZERO,
            duration: Duration::from_millis(250),
        });

        let program = HypercubeShaderProgram::new(
            0.5,
            2.0,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            false,
            RotateButton::default(),
            250,
            1,
            0,
            0,
            0,
            false,
            0,
            0,
            None,
            0,
            0,
            SolveCommand::Stop,
        );

        let vertices = program.build_gizmo_vertices(&state, 0.0, 1.0);
        assert!(vertices.is_empty());
    }

    #[test]
    fn build_gizmo_vertices_hides_ring_for_near_identity_reset() {
        let mut state = HypercubeShaderState::default();
        let (start_p, start_q) = decompose_so4(&Matrix4::identity());
        state.animating_reset = Some(AnimatingReset {
            start_p,
            start_q,
            elapsed: Duration::ZERO,
            duration: Duration::from_millis(250),
        });

        let program = HypercubeShaderProgram::new(
            0.5,
            2.0,
            1.0,
            VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Classic,
            AABBMode::None,
            false,
            true,
            RotateButton::default(),
            250,
            1,
            0,
            0,
            0,
            false,
            0,
            0,
            None,
            0,
            0,
            SolveCommand::Stop,
        );

        let vertices = program.build_gizmo_vertices(&state, 0.0, 1.0);
        assert!(vertices.is_empty());
    }

    #[test]
    fn lerp_color_is_identity_at_endpoints_and_averages_at_midpoint() {
        let a = [0.0, 0.2, 0.4, 1.0];
        let b = [1.0, 0.8, 0.0, 0.5];

        assert_eq!(lerp_color(a, b, 0.0), a);
        assert_eq!(lerp_color(a, b, 1.0), b);

        let mid = lerp_color(a, b, 0.5);
        for i in 0..4 {
            assert!((mid[i] - (a[i] + b[i]) / 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn gizmo_ring_radius_matches_face_gap_plus_face_gap_4d() {
        assert!((gizmo_ring_radius(0.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((gizmo_ring_radius(0.45, 2.0) - 2.45).abs() < 1e-6);
        assert!((gizmo_ring_radius(1.5, 2.0) - 3.5).abs() < 1e-6);
    }

    #[test]
    fn recompute_flat_normals_matches_cross_product_and_is_shared_across_a_triangles_corners() {
        let mut vertices = vec![
            gizmo_vertex(Point3::new(0.0, 0.0, 0.0), [0.0; 4]),
            gizmo_vertex(Point3::new(1.0, 0.0, 0.0), [0.0; 4]),
            gizmo_vertex(Point3::new(0.0, 1.0, 0.0), [0.0; 4]),
        ];

        recompute_flat_normals(&mut vertices);

        for vertex in &vertices {
            assert!((Vector3::from(vertex.normal) - Vector3::z()).norm() < 1e-6);
        }
    }

    #[test]
    fn recompute_flat_normals_falls_back_to_z_for_a_degenerate_triangle() {
        let mut vertices = vec![
            gizmo_vertex(Point3::new(0.0, 0.0, 0.0), [0.0; 4]),
            gizmo_vertex(Point3::new(1.0, 0.0, 0.0), [0.0; 4]),
            gizmo_vertex(Point3::new(2.0, 0.0, 0.0), [0.0; 4]),
        ];

        recompute_flat_normals(&mut vertices);

        for vertex in &vertices {
            assert!((Vector3::from(vertex.normal) - Vector3::z()).norm() < 1e-6);
        }
    }
}

#[cfg(test)]
mod clockwise_sign_tests {
    use super::*;
    use crate::geometry::FACE_CENTERS;
    use crate::math::project_4d_to_3d;
    use crate::moves::clockwise_sign;
    use nalgebra::Point3;

    /// For every actionable facet, `moves::clockwise_sign` must agree with
    /// an oracle built independently of the cofactor formula it uses
    /// internally: two probes `p1`, `p2 = axis x p1` (perpendicular to the
    /// rotation axis) have true rendered velocities `v1`, `v2` under a small
    /// rotation, and `v1 x v2` is the true spin axis, measured with no axis
    /// transform formula at all - just how two points actually move.
    ///
    /// Clockwise (as viewed from beyond the facet, looking back in) means
    /// that true spin axis points away from a viewer standing further out
    /// along the same ray the facet sits on - the direction given by
    /// transforming `local_coords` as an ordinary vector, not a pseudovector.
    #[test]
    fn clockwise_sign_matches_velocity_oracle() {
        let rotation_4d = Matrix4::identity();

        for facet in FACET_TABLE.iter().filter(|f| f.is_actionable) {
            let position_4d = Vector4::from(facet.position_4d);
            let base = project_4d_to_3d(position_4d, &rotation_4d, VIEWER_DISTANCE);

            const EPSILON: f32 = 1e-3;
            let tangent = |axis: usize| -> Vector3<f32> {
                let mut offset_4d = position_4d;
                offset_4d[axis] += EPSILON;
                (project_4d_to_3d(offset_4d, &rotation_4d, VIEWER_DISTANCE) - base) / EPSILON
            };
            let local_coords = facet.local_coords.map(|c| c as f32);
            let d = facet.free_axes.map(tangent);
            let ordinary_direction =
                d[0] * local_coords[0] + d[1] * local_coords[1] + d[2] * local_coords[2];

            let axis_unit =
                Vector3::new(local_coords[0], local_coords[1], local_coords[2]).normalize();
            let candidate = if axis_unit.x.abs() < 0.9 {
                Vector3::x()
            } else {
                Vector3::y()
            };
            let p1 = (candidate - axis_unit * candidate.dot(&axis_unit)).normalize();
            let p2 = axis_unit.cross(&p1).normalize();

            const ANGLE_EPSILON: f32 = 1e-5;
            let velocity = |p: Vector3<f32>| -> Vector3<f32> {
                let pre = project_cube_point(
                    p,
                    position_4d,
                    facet.axis,
                    &rotation_4d,
                    VIEWER_DISTANCE,
                    Vector4::zeros(),
                );
                let rotated = rotate_local_position(facet.local_coords, ANGLE_EPSILON, p.into());
                let post = project_cube_point(
                    Vector3::from(rotated),
                    position_4d,
                    facet.axis,
                    &rotation_4d,
                    VIEWER_DISTANCE,
                    Vector4::zeros(),
                );
                (post - pre) / ANGLE_EPSILON
            };
            let ground_truth_axis = velocity(p1).cross(&velocity(p2));

            let expected_sign = if ground_truth_axis.dot(&ordinary_direction) > 0.0 {
                -1.0
            } else {
                1.0
            };
            let actual_sign = clockwise_sign(facet);
            assert_eq!(
                actual_sign, expected_sign,
                "facet piece_slot={} axis={} side_sign={} local_coords={:?}: \
                 clockwise_sign returned {actual_sign}, oracle expected {expected_sign}",
                facet.piece_slot, facet.axis, facet.side_sign, facet.local_coords
            );
        }
    }

    /// The instances, camera and 4D rotation `depth_draw_order` is exercised
    /// against, all taken from a freshly-solved puzzle.
    fn fire_order_fixture() -> (Vec<StickerInstance>, Camera, Matrix4<f32>) {
        let state = HypercubeShaderState::default();
        let instances = sticker_instances_for_render(&state);
        (instances, state.camera.clone(), state.rotation_4d)
    }

    /// A sticker scale close to, but not exceeding, 1.0 (the scale at which
    /// neighbouring cells' cubes exactly touch with no 3D overlap): large
    /// enough that many camera angles show two neighbours' screen-space
    /// footprints genuinely overlapping, without ever letting the cubes
    /// themselves interpenetrate in 3D, which would make "which is in
    /// front" locally ill-defined even for a pair a single flat plane truly
    /// separates.
    const TEST_STICKER_SCALE: f32 = 0.95;

    /// The Fire-tagged subsequence of the combined `depth_draw_order`,
    /// preserving relative order, so existing Fire-only assertions still
    /// apply against it.
    fn fire_order_for(
        instances: &[StickerInstance],
        camera: &Camera,
        rotation_4d: &Matrix4<f32>,
        visible: &[bool; 8],
    ) -> Vec<u32> {
        depth_draw_order(
            instances,
            visible,
            rotation_4d,
            camera,
            true,
            TEST_STICKER_SCALE,
            SECONDARY_FACE_GAP,
            SECONDARY_FACE_GAP_4D,
            VIEWER_DISTANCE,
        )
        .into_iter()
        .flatten()
        .filter_map(|layer| match layer {
            DepthLayer::Fire(index) => Some(index),
            DepthLayer::Ice(_) | DepthLayer::Light(_) | DepthLayer::Dirt(_) => None,
        })
        .collect()
    }

    /// Builds the `FireSticker` `depth_draw_order` would for `index`, so tests
    /// can call `FireSticker::is_behind` directly instead of inferring its
    /// verdict from final draw ranks.
    fn fire_sticker_for(
        index: usize,
        instances: &[StickerInstance],
        rotation_4d: &Matrix4<f32>,
        camera: &Camera,
        sticker_scale: f32,
    ) -> FireSticker {
        let instance = &instances[index];
        let face_id = index / (instances.len() / 8);
        let position_4d = Vector4::from(instance.position_4d);
        let forward = (camera.target - camera.eye).normalize();
        let center = project_face_point(
            position_4d,
            Vector4::from(instance.face_normal_4d),
            rotation_4d,
            SECONDARY_FACE_GAP,
            SECONDARY_FACE_GAP_4D,
            VIEWER_DISTANCE,
        );
        let world_vertices = transform_sticker_vertices_to_3d(
            position_4d,
            face_id,
            rotation_4d,
            sticker_scale,
            SECONDARY_FACE_GAP,
            SECONDARY_FACE_GAP_4D,
            VIEWER_DISTANCE,
        );
        let aabb = calculate_sticker_aabb(&world_vertices);

        FireSticker {
            instance_index: index as u32,
            depth: (center - camera.eye).dot(&forward),
            world_vertices,
            aabb,
        }
    }

    /// Places a 4D point of `face_id` in world space the way the renderer
    /// does, at the fixture's slider values.
    fn place(point_4d: Vector4<f32>, face_id: usize, rotation_4d: &Matrix4<f32>) -> Point3<f32> {
        project_face_point(
            point_4d,
            FACE_CENTERS[face_id],
            rotation_4d,
            SECONDARY_FACE_GAP,
            SECONDARY_FACE_GAP_4D,
            VIEWER_DISTANCE,
        )
    }

    /// Which side of the plane through `a`, `b` and `c` the point `p` lies on,
    /// as a signed volume. Zero means coplanar.
    fn side_of_plane(p: Point3<f32>, a: Point3<f32>, b: Point3<f32>, c: Point3<f32>) -> f32 {
        (p - a).dot(&(b - a).cross(&(c - a)))
    }

    #[test]
    fn fire_order_covers_every_fire_sticker_exactly_once() {
        let (instances, camera, rotation_4d) = fire_order_fixture();
        let drawn = fire_order_for(&instances, &camera, &rotation_4d, &[true; 8]);

        let mut expected: Vec<u32> = instances
            .iter()
            .enumerate()
            .filter(|(_, instance)| instance.kind == ELEMENTAL_FIRE_KIND)
            .map(|(index, _)| index as u32)
            .collect();
        assert!(
            !expected.is_empty(),
            "a solved puzzle should have fire stickers to draw"
        );

        let mut sorted = drawn.clone();
        sorted.sort_unstable();
        expected.sort_unstable();
        assert_eq!(sorted, expected);
    }

    /// Two cells of one face that neighbour each other along a single free
    /// axis are separated by the lattice slab boundary between them, and under
    /// the 4D projection that boundary is a genuine plane: restricted to a
    /// face's own 3-flat, projection from the 4D viewer is a projectivity -
    /// the flat never approaches the `viewer_distance - w = 0` singularity,
    /// since a point of the puzzle has `|R*p| <= sqrt(1 + 3 * GRID_EXTENT^2)`,
    /// well under `VIEWER_DISTANCE` - and a projectivity carries 2-flats to
    /// planes. Both pushes preserve that: the 4D one translates the flat
    /// before the divide, the 3D one translates the whole face after it.
    ///
    /// So whichever of the two cells lies on the camera's side of that plane
    /// is in front of the other and, whenever the ray-cast test in
    /// `FireSticker::is_behind` finds a screen-space overlap for the pair at
    /// all, must not come back behind it. This asserts that geometry
    /// directly, rather than recomputing `is_behind`'s own key, so it can
    /// actually catch a wrong one.
    #[test]
    fn fire_order_respects_separating_planes() {
        let (instances, base_camera, base_rotation) = fire_order_fixture();

        // One orientation proves very little: a key can satisfy every
        // separating plane from one viewpoint and invert pairs from the next.
        // Sweep a few 4D rotations against a ring of camera azimuths.
        for (plane_a, plane_b) in [(0, 3), (1, 3), (0, 1), (2, 3)] {
            for turn in 0..4 {
                let mut axis_a = Vector4::zeros();
                let mut axis_b = Vector4::zeros();
                axis_a[plane_a] = 1.0;
                axis_b[plane_b] = 1.0;
                let rotation_4d = create_4d_plane_rotation(
                    axis_a,
                    axis_b,
                    std::f32::consts::FRAC_PI_2 * 0.37 * turn as f32,
                ) * base_rotation;

                for azimuth in 0..6 {
                    let angle = std::f32::consts::TAU * azimuth as f32 / 6.0;
                    let radius = (base_camera.eye - base_camera.target).norm();
                    let mut camera = base_camera.clone();
                    camera.eye = base_camera.target
                        + Vector3::new(radius * angle.cos(), radius * 0.4, radius * angle.sin());

                    assert_separating_planes_respected(&instances, &camera, &rotation_4d);
                }
            }
        }
    }

    fn assert_separating_planes_respected(
        instances: &[StickerInstance],
        camera: &Camera,
        rotation_4d: &Matrix4<f32>,
    ) {
        let facets_per_face = instances.len() / 8;
        let fire_indices: Vec<usize> = instances
            .iter()
            .enumerate()
            .filter(|(_, instance)| instance.kind == ELEMENTAL_FIRE_KIND)
            .map(|(index, _)| index)
            .collect();
        let mut checked = 0;

        for &a_index in &fire_indices {
            for &b_index in &fire_indices {
                let face_id = a_index / facets_per_face;
                if b_index / facets_per_face != face_id {
                    continue;
                }

                let a_position = Vector4::from(instances[a_index].position_4d);
                let b_position = Vector4::from(instances[b_index].position_4d);

                // Neighbours along exactly one free axis, one lattice step
                // apart, so exactly one boundary separates them.
                let free_axes: Vec<usize> =
                    (0..4).filter(|&axis| axis != FIXED_DIMS[face_id]).collect();
                let differing: Vec<usize> = free_axes
                    .iter()
                    .copied()
                    .filter(|&axis| a_position[axis] != b_position[axis])
                    .collect();
                let [axis] = differing[..] else {
                    continue;
                };
                if (a_position[axis] - b_position[axis]).abs() > GRID_EXTENT * 1.5 {
                    continue;
                }

                // Three points spanning the boundary 2-flat: fixed at the
                // midpoint along `axis`, swept along the other two free axes.
                let spanning: Vec<usize> = free_axes
                    .iter()
                    .copied()
                    .filter(|&other| other != axis)
                    .collect();
                let boundary = (a_position[axis] + b_position[axis]) / 2.0;
                let corner = |offsets: [f32; 2]| {
                    let mut point = a_position;
                    point[axis] = boundary;
                    point[spanning[0]] = offsets[0];
                    point[spanning[1]] = offsets[1];
                    place(point, face_id, rotation_4d)
                };
                let (p, q, r) = (corner([0.0, 0.0]), corner([1.0, 0.0]), corner([0.0, 1.0]));

                let eye_side = side_of_plane(camera.eye, p, q, r);
                let a_side = side_of_plane(place(a_position, face_id, rotation_4d), p, q, r);
                let b_side = side_of_plane(place(b_position, face_id, rotation_4d), p, q, r);

                // The plane has to actually separate the two cells, and the
                // camera has to be off it, for the comparison to mean
                // anything. Only assert for the cell facing the camera.
                if a_side * b_side >= 0.0 || eye_side == 0.0 || a_side.signum() != eye_side.signum()
                {
                    continue;
                }

                // Only a claim `is_behind` actually makes can be wrong; a
                // `None` just means this pair's screen footprints didn't
                // overlap for this camera/rotation, which says nothing about
                // draw order since they wouldn't visually interact anyway.
                let sticker_a =
                    fire_sticker_for(a_index, instances, rotation_4d, camera, TEST_STICKER_SCALE);
                let sticker_b =
                    fire_sticker_for(b_index, instances, rotation_4d, camera, TEST_STICKER_SCALE);
                let Some(ordering) = sticker_a.is_behind(camera.eye, &sticker_b) else {
                    continue;
                };

                checked += 1;
                assert_ne!(
                    ordering,
                    Ordering::Greater,
                    "instance {a_index} is on the camera's side of the boundary separating it \
                     from {b_index} on face {face_id}, so it is in front and must not be behind \
                     it, but is_behind returned {ordering:?}"
                );
            }
        }

        assert!(
            checked > 0,
            "the fixture produced no separated, screen-overlapping neighbour pairs to check"
        );
    }

    #[test]
    fn fire_order_skips_stickers_on_invisible_faces() {
        let (instances, camera, rotation_4d) = fire_order_fixture();
        let mut visible = [true; 8];
        visible[0] = false;

        let drawn = fire_order_for(&instances, &camera, &rotation_4d, &visible);
        let facets_per_face = instances.len() / 8;
        assert!(
            drawn
                .iter()
                .all(|&index| index as usize / facets_per_face != 0),
            "face 0 is hidden, so none of its stickers should be drawn"
        );
        assert!(
            !drawn.is_empty(),
            "hiding one face should not hide every fire sticker"
        );
    }

    /// `DepthLayer`'s carried instance index, regardless of kind.
    fn depth_layer_instance_index(layer: &DepthLayer) -> u32 {
        match *layer {
            DepthLayer::Fire(index)
            | DepthLayer::Ice(index)
            | DepthLayer::Light(index)
            | DepthLayer::Dirt(index) => index,
        }
    }

    #[test]
    fn depth_batches_never_group_stickers_with_an_established_order() {
        let (instances, camera, rotation_4d) = fire_order_fixture();
        let batches = depth_draw_order(
            &instances,
            &[true; 8],
            &rotation_4d,
            &camera,
            true,
            TEST_STICKER_SCALE,
            SECONDARY_FACE_GAP,
            SECONDARY_FACE_GAP_4D,
            VIEWER_DISTANCE,
        );
        assert!(
            batches.len() > 1,
            "a solved puzzle's fire/ice stickers should span more than one batch"
        );

        for batch in &batches {
            for (i, a) in batch.iter().enumerate() {
                for b in &batch[i + 1..] {
                    let sticker_a = fire_sticker_for(
                        depth_layer_instance_index(a) as usize,
                        &instances,
                        &rotation_4d,
                        &camera,
                        TEST_STICKER_SCALE,
                    );
                    let sticker_b = fire_sticker_for(
                        depth_layer_instance_index(b) as usize,
                        &instances,
                        &rotation_4d,
                        &camera,
                        TEST_STICKER_SCALE,
                    );
                    assert_eq!(
                        sticker_a.is_behind(camera.eye, &sticker_b),
                        None,
                        "instances {} and {} share a batch, but is_behind found an order \
                         between them - they aren't safe to share a pass",
                        depth_layer_instance_index(a),
                        depth_layer_instance_index(b)
                    );
                }
            }
        }
    }

    #[test]
    fn depth_batches_respect_established_order_across_batches() {
        let (instances, camera, rotation_4d) = fire_order_fixture();
        let batches = depth_draw_order(
            &instances,
            &[true; 8],
            &rotation_4d,
            &camera,
            true,
            TEST_STICKER_SCALE,
            SECONDARY_FACE_GAP,
            SECONDARY_FACE_GAP_4D,
            VIEWER_DISTANCE,
        );

        let mut batch_of = std::collections::HashMap::new();
        for (batch_index, batch) in batches.iter().enumerate() {
            for layer in batch {
                batch_of.insert(depth_layer_instance_index(layer), batch_index);
            }
        }

        let participants: Vec<usize> = instances
            .iter()
            .enumerate()
            .filter(|(_, instance)| {
                instance.kind == ELEMENTAL_FIRE_KIND || instance.kind == ELEMENTAL_ICE_KIND
            })
            .map(|(index, _)| index)
            .collect();

        let mut checked = 0;
        for &a_index in &participants {
            for &b_index in &participants {
                if a_index >= b_index {
                    continue;
                }
                let sticker_a = fire_sticker_for(
                    a_index,
                    &instances,
                    &rotation_4d,
                    &camera,
                    TEST_STICKER_SCALE,
                );
                let sticker_b = fire_sticker_for(
                    b_index,
                    &instances,
                    &rotation_4d,
                    &camera,
                    TEST_STICKER_SCALE,
                );
                let Some(ordering) = sticker_a.is_behind(camera.eye, &sticker_b) else {
                    continue;
                };
                checked += 1;
                let (before, after) = match ordering {
                    Ordering::Greater => (a_index as u32, b_index as u32),
                    Ordering::Less => (b_index as u32, a_index as u32),
                    Ordering::Equal => unreachable!("is_behind never returns Equal"),
                };
                let (before_batch, after_batch) = (batch_of[&before], batch_of[&after]);
                assert!(
                    before_batch < after_batch,
                    "instance {before} must draw before {after}, but batching put them in \
                     batches {before_batch} and {after_batch}"
                );
            }
        }

        assert!(
            checked > 0,
            "the fixture produced no established orderings to check"
        );
    }

    #[test]
    fn depth_batches_are_singletons_when_lattice_is_not_aligned() {
        let (instances, camera, rotation_4d) = fire_order_fixture();
        let batches = depth_draw_order(
            &instances,
            &[true; 8],
            &rotation_4d,
            &camera,
            false,
            TEST_STICKER_SCALE,
            SECONDARY_FACE_GAP,
            SECONDARY_FACE_GAP_4D,
            VIEWER_DISTANCE,
        );

        assert!(!batches.is_empty());
        assert!(
            batches.iter().all(|batch| batch.len() == 1),
            "mid-animation batches must stay singletons: with no lattice-aligned relation \
             built, there's no overlap evidence to batch on"
        );
    }

    /// One instance per face_id, kind `ELEMENTAL_FIRE_KIND` on `fire_faces`
    /// and an arbitrary non-Fire kind everywhere else.
    fn fire_kind_instances(fire_faces: &[u32]) -> Vec<StickerInstance> {
        (0..8u32)
            .map(|face_id| StickerInstance {
                position_4d: [0.0; 4],
                basis: [[0.0; 4]; 3],
                face_normal_4d: [0.0; 4],
                kind: if fire_faces.contains(&face_id) {
                    ELEMENTAL_FIRE_KIND
                } else {
                    0
                },
                _padding: [0; 3],
            })
            .collect()
    }

    #[test]
    fn ground_truth_debug_face_is_none_when_disabled() {
        let instances = fire_kind_instances(&[3]);
        assert_eq!(
            ground_truth_debug_face(false, Theme::Elemental, &instances, 0.0),
            None
        );
    }

    #[test]
    fn ground_truth_debug_face_is_none_under_classic_theme() {
        let instances = fire_kind_instances(&[3]);
        assert_eq!(
            ground_truth_debug_face(true, Theme::Classic, &instances, 0.0),
            None
        );
    }

    #[test]
    fn ground_truth_debug_face_is_none_with_no_fire_stickers() {
        let instances = fire_kind_instances(&[]);
        assert_eq!(
            ground_truth_debug_face(true, Theme::Elemental, &instances, 0.0),
            None
        );
    }

    #[test]
    fn ground_truth_debug_face_cycles_through_fire_faces_in_order() {
        let instances = fire_kind_instances(&[2, 5, 7]);

        assert_eq!(
            ground_truth_debug_face(true, Theme::Elemental, &instances, 0.0),
            Some(2)
        );
        assert_eq!(
            ground_truth_debug_face(
                true,
                Theme::Elemental,
                &instances,
                GROUND_TRUTH_DEBUG_FACE_CYCLE_SECONDS
            ),
            Some(5)
        );
        assert_eq!(
            ground_truth_debug_face(
                true,
                Theme::Elemental,
                &instances,
                GROUND_TRUTH_DEBUG_FACE_CYCLE_SECONDS * 2.0
            ),
            Some(7)
        );
        // Wraps back to the first face after the last.
        assert_eq!(
            ground_truth_debug_face(
                true,
                Theme::Elemental,
                &instances,
                GROUND_TRUTH_DEBUG_FACE_CYCLE_SECONDS * 3.0
            ),
            Some(2)
        );
    }
}
