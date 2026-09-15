# settings.rs

`AppSettings` persisted via `serde`/`toml`/`directories`: `rotate_button` (`RotateButton`), `animation_duration_ms`, `theme` (`Theme`), `debug_mode`, `show_gizmo_ring`, and `show_tutorial_on_launch` (all `#[serde(default)]` so an older `settings.toml` still deserializes, falling back to `false`).
