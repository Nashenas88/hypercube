# settings.rs

`AppSettings` persisted via `serde`/`toml`/`directories`. Includes `rotation_gizmo_display: RotationGizmoDisplay` (`DominantAxis` default, or `BothAxes`), controlling how the rotation-axis gizmo shows a diagonal Shift+drag's two independent component rotations - see `shader_widget.md`.
