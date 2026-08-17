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

UI, 3D/4D logic, and GPU rendering are deliberately kept in separate layers.
Full per-file descriptions live under `context/`; each bullet below is a
short pointer:

- `main.rs` — one-line binary entry point. See `context/main.md`.
- `lib.rs` — declares modules, exposes `run()`, wires up `HypercubeApp`; no UI/3D logic itself. See `context/lib.md`.
- `app.rs` — UI-control state only (sliders, render mode, reveal toggle); no 3D/4D logic. See `context/app.md`.
- `shader_widget.rs` — owns rendering/interaction state and input handling, independent of `HypercubeApp`. See `context/shader_widget.md`.
- `renderer.rs` — owns wgpu resources; per-face instanced draws; generation-gated buffer uploads. See `context/renderer.md`.
- `camera.rs` — 3D orbit camera (`Camera`, `CameraController`, `Projection`). See `context/camera.md`.
- `math.rs` — CPU-side 4D rotation matrices and 4D→3D perspective projection. See `context/math.md`.
- `geometry.rs` — static, puzzle-state-independent tables (face centers, vertices, winding). See `context/geometry.md`.
- `piece.rs` — core domain model (`Piece`, `Hypercube`, `FACET_TABLE`, sticker instance generation). See `context/piece.md`.
- `moves.rs` — move application: rotates a 3×3×3 side, snaps to an exact permutation. See `context/moves.md`.
- `ray_casting.rs` — CPU-side ray/AABB/triangle intersection for hover and click picking. See `context/ray_casting.md`.
- `settings.rs` — `AppSettings` persisted via `serde`/`toml`/`directories`. See `context/settings.md`.
- `shaders/*.wgsl` — WGSL shaders sharing structs/math via `math4d.wgsl`, composed with `naga_oil`. See `context/shaders.md`.

**Render/interaction flow:** `main.rs` → `app.rs`'s `HypercubeApp::view()` embeds the `Shader` widget → iced calls `HypercubeShaderProgram::update()` per event (mutates `HypercubeShaderState`) → `draw()` builds a `HypercubePrimitive` snapshot → iced calls `Primitive::prepare()` (uploads buffers via `Renderer::update_*`) then `Primitive::render()` (`Renderer::render()` / `render_debug_aabb()` encode the actual wgpu render passes).

## Guidance

When making changes to the code, make sure to keep this file in sync. If a
change affects a file described under `context/`, keep that file's
description in sync too.