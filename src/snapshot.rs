//! Explicit save of a full view snapshot: the current frame's rendered
//! pixels alongside every piece of state needed to reproduce that exact
//! view. Debug-only, for collecting fixtures while chasing rendering bugs -
//! unlike `puzzle_state`, saves are timestamped and additive rather than a
//! single overwritten file.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use nalgebra::Matrix4;
use serde::{Deserialize, Serialize};

use crate::app::{AABBMode, RenderMode};
use crate::camera::{Camera, CameraController, Projection};
use crate::piece::Hypercube;
use crate::theme::Theme;

/// Everything needed to reproduce one rendered view: the puzzle arrangement,
/// the 4D rotation, the 3D camera (including zoom, via `camera_controller`),
/// and every render-affecting slider/toggle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ViewSnapshot {
    pub(crate) puzzle: Hypercube,
    pub(crate) rotation_4d: Matrix4<f32>,
    pub(crate) camera: Camera,
    pub(crate) camera_controller: CameraController,
    pub(crate) projection: Projection,
    pub(crate) sticker_scale: f32,
    pub(crate) face_gap: f32,
    pub(crate) face_gap_4d: f32,
    pub(crate) viewer_distance: f32,
    pub(crate) render_mode: RenderMode,
    pub(crate) theme: Theme,
    pub(crate) aabb_mode: AABBMode,
    pub(crate) timestamp_millis: u128,
}

impl ViewSnapshot {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture(
        puzzle: Hypercube,
        rotation_4d: Matrix4<f32>,
        camera: Camera,
        camera_controller: CameraController,
        projection: Projection,
        sticker_scale: f32,
        face_gap: f32,
        face_gap_4d: f32,
        viewer_distance: f32,
        render_mode: RenderMode,
        theme: Theme,
        aabb_mode: AABBMode,
    ) -> Self {
        let timestamp_millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis())
            .unwrap_or(0);

        Self {
            puzzle,
            rotation_4d,
            camera,
            camera_controller,
            projection,
            sticker_scale,
            face_gap,
            face_gap_4d,
            viewer_distance,
            render_mode,
            theme,
            aabb_mode,
            timestamp_millis,
        }
    }
}

fn snapshot_dir() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "hypercube")
        .map(|dirs| dirs.data_dir().join("snapshots"))
}

/// Persists a snapshot's state as `<timestamp>.json` and its pixels as
/// `<timestamp>.png` alongside it, logging a warning on failure rather than
/// propagating an error, mirroring `puzzle_state::save`.
pub(crate) fn save(snapshot: &ViewSnapshot, rgba: &[u8], width: u32, height: u32) {
    let Some(dir) = snapshot_dir() else {
        log::warn!("Could not determine data directory; snapshot not saved");
        return;
    };

    if let Err(err) = std::fs::create_dir_all(&dir) {
        log::warn!("Failed to create snapshot directory {dir:?}: {err}");
        return;
    }

    let basename = format!("snapshot-{}", snapshot.timestamp_millis);

    match image::RgbaImage::from_raw(width, height, rgba.to_vec()) {
        Some(image) => {
            let png_path = dir.join(format!("{basename}.png"));
            if let Err(err) = image.save(&png_path) {
                log::warn!("Failed to write snapshot image to {png_path:?}: {err}");
            }
        }
        None => log::warn!(
            "Snapshot pixel buffer ({} bytes) doesn't match {width}x{height}; image not saved",
            rgba.len()
        ),
    }

    match serde_json::to_string_pretty(snapshot) {
        Ok(contents) => {
            let json_path = dir.join(format!("{basename}.json"));
            if let Err(err) = std::fs::write(&json_path, contents) {
                log::warn!("Failed to write snapshot state to {json_path:?}: {err}");
            }
        }
        Err(err) => log::warn!("Failed to serialize snapshot state: {err}"),
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point3, Vector3};

    use super::*;

    fn sample_snapshot() -> ViewSnapshot {
        ViewSnapshot::capture(
            Hypercube::solved(),
            Matrix4::identity(),
            Camera {
                eye: Point3::new(0.0, 0.0, 15.0),
                target: Point3::new(0.0, 0.0, 0.0),
                up: Vector3::new(0.0, 1.0, 0.0),
            },
            CameraController::new(15.0),
            Projection {
                aspect: 800.0 / 600.0,
                fovy: std::f32::consts::FRAC_PI_4,
                znear: 0.1,
                zfar: 100.0,
            },
            0.5,
            0.1,
            1.0,
            crate::math::VIEWER_DISTANCE,
            RenderMode::Standard,
            Theme::Elemental,
            AABBMode::None,
        )
    }

    #[test]
    fn round_trips_through_json() {
        let original = sample_snapshot();

        let json = serde_json::to_string_pretty(&original).expect("serialize");
        let restored: ViewSnapshot = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.puzzle, original.puzzle);
        assert_eq!(restored.rotation_4d, original.rotation_4d);
        assert_eq!(restored.camera.eye, original.camera.eye);
        assert_eq!(restored.camera.target, original.camera.target);
        assert_eq!(restored.camera.up, original.camera.up);
        assert_eq!(
            restored.camera_controller.distance,
            original.camera_controller.distance
        );
        assert_eq!(restored.sticker_scale, original.sticker_scale);
        assert_eq!(restored.render_mode, original.render_mode);
        assert_eq!(restored.theme, original.theme);
        assert_eq!(restored.aabb_mode, original.aabb_mode);
        assert_eq!(restored.timestamp_millis, original.timestamp_millis);
    }
}
