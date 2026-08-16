# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A 4D Rubik's-cube visualizer built in Rust with `iced` 0.14 (wgpu + advanced features) and `nalgebra`. No workspace. `src/lib.rs` holds all modules and logic behind `pub fn run()`; `src/main.rs` is a one-line binary entry point that calls it. The split exists solely so `benches/` has a library target to link against — it isn't a public API boundary.

## Commands

- Build: `cargo build`
- Run: `cargo run`
- Test all: `cargo test`
- Test one: `cargo test <test_name>` (all tests are inline `#[cfg(test)] mod tests` blocks — no `tests/` directory — in `app.rs`, `camera.rs`, `piece.rs`, `math.rs`, `moves.rs`, `shader_widget.rs`)
- Format: `cargo fmt`
- Lint: `cargo clippy --all-targets`

No CI config, justfile, or Makefile exists — the commands above are the full local dev loop.

### Performance measurement

- `cargo bench` — criterion benchmarks (`benches/instances.rs`) for the deterministic CPU-only per-frame allocation sites named in `perf_improvements.md` item #3 (`HypercubeShaderProgram::calculate_indices`, `sticker_instances_for_render`), run against `HypercubeShaderState::default()`.
- `cargo run --release --features gpu-capture-hooks` — for GPU-bound measurement (`perf`, RenderDoc): the `gpu-capture-hooks` feature makes the app kick off 5 back-to-back reveal/hide flourishes on boot and then call `iced::exit()`, giving a profiler attached to the running process (e.g. `perf record -p <pid>`) a fixed, reproducible workload instead of relying on manual clicking.

## Architecture

UI, 3D/4D logic, and GPU rendering are deliberately kept in separate layers:

- `main.rs` — one-line binary entry point (`hypercube::run()`).
- `lib.rs` — declares all modules and exposes `pub fn run() -> iced::Result`, the thin `iced::application` setup that wires `app::HypercubeApp::new/update/view` together. Contains no UI or 3D/4D logic itself.
- `app.rs` — `HypercubeApp` holds only UI-control state (scale/gap sliders, render mode, settings, the reveal toggle's runtime state). Builds a left control panel plus a right `Shader::new(HypercubeShaderProgram)` viewport. Contains no 3D/4D logic. The sticker-scale/face-gap sliders are hidden behind a "Reveal"/"Hide" toggle button: pressing it bumps a `reveal_generation` counter (same one-shot generation-counter pattern `reset_generation` uses to reach `HypercubeShaderState`) and flips `revealed` immediately, while `reveal_animating` disables the button and hides the sliders until `shader_widget.rs` publishes `Message::RevealAnimationComplete` back once the flourish settles.
- `shader_widget.rs` — the custom iced `shader::Program`/`Primitive` that owns essentially all rendering and interaction state, independent of `HypercubeApp`. `HypercubeShaderProgram` (per-frame config), `HypercubeShaderState` (persistent: camera, 4D rotation matrix, hover/click/double-click bookkeeping, animation state, the `Hypercube` puzzle state), `HypercubePrimitive` (per-draw snapshot). Mouse/keyboard events are handled here; a `RotateButton` setting assigns one mouse button to camera orbit (+Shift for 4D rotation) and the other to puzzle turn-clicks, so the two never conflict. A turn-click's direction is resolved by `moves::clockwise_sign` to always turn clockwise as viewed along the clicked facet's own rotation axis; Shift reverses it to counterclockwise. Double-click on a face triggers a "center this face" animation via `math::shortest_arc_plane`. A reveal/hide flourish (`AnimatingReveal`) spins the camera 720° in yaw while sticker scale/face gap sweep toward secondary/primary defaults, driven by the same `RedrawRequested`-tick loop as the move/focus animations; camera-drag and turn-click input are ignored while it plays.
- `renderer.rs` — owns all wgpu resources (buffers, pipelines for standard/normal/depth/debug/sky, textures). Draws all 216 sticker facets with one instanced draw call per pipeline from a single static cube mesh.
- `camera.rs` — 3D orbit camera (`Camera`, `CameraController`, `Projection`).
- `math.rs` — CPU-side 4D rotation matrices for the 6 rotation planes, generic `create_4d_plane_rotation`, 4D→3D perspective projection (`project_cube_point`).
- `geometry.rs` — static, puzzle-state-independent tables (face centers, base cube vertices, winding/index tables). Puzzle state itself lives in `piece.rs`.
- `piece.rs` — core domain model. `Piece { position: [i8;4], colors: [Option<Color>; 4] }`; `position` is a lattice point in `{-1,0,1}^4`, `colors[axis]` is set only for nonzero axes. `Hypercube` always holds exactly 81 pieces in a canonical order (`index_of`/`position_of`), so two states can be compared with `assert_eq!` directly — this piece-based model replaced an earlier sticker-based one. `FACET_TABLE` (216 entries) and `generate_sticker_instances()` derive per-frame GPU instance data from piece state.
- `moves.rs` — move application. A move rotates one "side" (27 pieces sharing a fixed coordinate on one axis) as a rigid 3×3×3 subcube; the rotation axis comes from the clicked piece's local coordinates on the 3 free axes, and turn angle (90°/180°/120°) depends on how many of those are nonzero. `discrete_rotation()` snaps a continuous rotation matrix to an exact signed permutation.
- `ray_casting.rs` — CPU-side ray/AABB/triangle intersection against 4D→3D-projected stickers, for hover and click picking.
- `settings.rs` — `AppSettings` persisted via `serde`/`toml`/`directories`.
- `shaders/*.wgsl` — WGSL shaders sharing `Transform4D` (`rotation_matrix`, `viewer_distance`, `sticker_scale`, `face_gap`), `CameraUniform`, and `StickerInstance` structs plus 4D math functions, all defined once in `math4d.wgsl` and pulled into each pipeline shader via `naga_oil`'s `#import` (composed in `renderer.rs` through a `naga_oil::compose::Composer`, since WGSL itself has no import mechanism).

**Render/interaction flow:** `main.rs` → `app.rs`'s `HypercubeApp::view()` embeds the `Shader` widget → iced calls `HypercubeShaderProgram::update()` per event (mutates `HypercubeShaderState`) → `draw()` builds a `HypercubePrimitive` snapshot → iced calls `Primitive::prepare()` (uploads buffers via `Renderer::update_*`) then `Primitive::render()` (`Renderer::render()` / `render_debug_aabb()` encode the actual wgpu render passes).


## Guidance

When making changes to the code, make sure to keep this file in sync.