# Rendering performance improvements

Findings from a review of the render hot path. Line numbers drift — grep for
the quoted code.

**Status:** #1 is implemented (see §1) — it turned out to require pulling
forward the core of #5's instance reorder, not the standalone fix originally
sketched below. #3 is implemented (see §3): index/sticker-instance
regeneration and GPU upload are generation-gated, and
`update_debug_instances` reuses a scratch buffer instead of allocating one
per frame. #2 is implemented (see §2): the per-vertex rotation matmuls in
`vs_main` are hoisted and reused across the visibility test, normal, and
position/push paths, and `compute_world_normal`'s dynamic array indexing is
gone. #4 and the bonus item are still open. #4 (skip invisible-face draws)
is still open — the per-face draw loop #1 added is exactly where it would
plug in.

**Suggested order (for what's left):** #4 next (now cheap given #1's
per-face draws already exist).

---

## How to collect perf numbers

There's no benchmarking or profiling infrastructure in the project yet (no
`criterion`, no FPS counter, no GPU timestamp queries) and redraws are
event-driven, not continuous — the app renders nothing at true idle (see
`Event::Window(RedrawRequested)` handling in `src/shader_widget.rs`). "Leave
it idle and watch the FPS" is not a usable baseline here. A real measurement
needs a workload and a way to read the result.

1. **Always measure `--release`.** `cargo build --release` /
   `cargo run --release`. Debug-build timings aren't representative of
   anything in this doc.

2. **Use a reproducible, self-driving workload.** The Reveal/Hide flourish
   (`AnimatingReveal`, a fixed-duration 720° yaw spin — see the reveal
   button in the UI) keeps requesting redraws on its own for its whole
   duration with no mouse input needed, so it's the natural repeatable
   benchmark scenario: same duration and frame count every run, no
   human-input variance between samples. Repeatedly triggering a move
   animation works too. Avoid ad hoc mouse-dragging as your measured
   workload — it's not reproducible run to run.

3. **Two complementary techniques:**
   - **Coarse wall-clock frame timing**, good for before/after deltas on
     #1, #2, and #4: temporarily add an `Instant`-based rolling frame-time
     average logged via `log::info!` (matching the existing
     `log::debug!`/`log::warn!` calls already in `calculate_indices`), run
     with `RUST_LOG=info cargo run --release`, and diff the average over one
     full reveal-animation run before vs. after the change. This is
     temporary local instrumentation, not something to commit, and it
     measures CPU submission time, not pure GPU time.
   - **GPU ground truth via an external frame-capture profiler** (RenderDoc
     on the Vulkan backend, or a platform equivalent) — the only way to
     directly confirm the invocation-count claims in this doc, e.g. item
     #1's 62,208 vs. 7,776 vertex shader invocations, or item #5's up to 8
     separate draws. Wall-clock frame time alone conflates GPU cost with CPU
     submission time, vsync capping, and OS scheduler noise.

4. **Controls:** same window size and same puzzle state at the start of each
   run (solved cube, no pending move) so instance/vertex counts match across
   runs; take several samples (e.g. 5 full reveal runs) and compare the
   median, not one run — 216 instances is small enough that scheduler/driver
   noise can dominate a single sample; measure on the lowest-spec hardware
   you actually care about, since a modern desktop GPU may hide all of this
   under vsync.

5. **#3 needs a different measurement.** It's CPU/upload-bound, not
   GPU-bound, so the GPU-profiler technique above won't show it at all. Its
   win is clearest by comparing CPU time spent in `draw()`/
   `Primitive::prepare()` before and after adding dirty flags, for a workload
   where `Hypercube` state and `animating_move` don't change — a plain
   camera-drag is the intuitive example, but per item 2 above it isn't
   reproducible run to run; the reveal/hide flourish (scale/gap/yaw sweep,
   no `Hypercube`/`animating_move` change either) is the reproducible
   stand-in, and it's what the `gpu-capture-hooks` feature already drives.

---

## 1. Every instance draws the cube 8 times (8× overdraw) — FIXED

**Impact: high. Confidence: verified. Status: fixed**, by grouping
`FACET_TABLE` face-major and issuing 8 per-face draws — see "Fix" below.
Confirmed via RenderDoc: 8 draws of `count=36, instancecount=27` each
(was 1 draw of `count=288, instancecount=216`), same 7,776 total vertex
invocations §5 predicted, correctly distributed this time. The rest of
this section is kept for the diagnosis, which is still accurate background
for §4 and the bonus item.

`src/renderer.rs`, in `render()` (before the fix):

```rust
render_pass.draw_indexed(
    0..VERTEX_NORMAL_INDICES.len() as u32 * 8,  // 0..288
    0,
    0..self.num_stickers as u32,                 // 0..216
);
```

The index buffer holds 8 chunks of 36 indices — one winding-order variant per
`face_id`. But this draw feeds **all 288 indices to every one of the 216
instances**. At startup the buffer is built as
`VERTEX_NORMAL_INDICES.into_iter().cycle().take(36 * 8)`, i.e. 8 *identical*
copies.

In an indexed draw, `@builtin(vertex_index)` is the value fetched from the
index buffer (0..35), not the draw-sequence position. So all 8 chunks produce
the same `face_3d = vertex_index / 6u` mapping, and each instance rasterizes
the same cube 8 times at identical depth. It renders correctly — backface
culling plus equal-depth Z testing hide it — but costs **62,208 vertex shader
invocations per frame where 7,776 suffice**.

### Important corollary

Because every instance receives every chunk, the winding correction computed in
`HypercubeShaderProgram::calculate_indices` (`src/shader_widget.rs`) **never
reaches the instances it was computed for**. Fixing this draw call is what
makes that correction start mattering, which may change how culling looks.

The heuristic it uses is itself suspect:

```rust
let centroid = transformed_vertices.iter().sum::<Vector3<f32>>() / 8.0;
if normal.dot(&centroid) < 0.0 { triangle_indices.swap(1, 2); }
```

`centroid` is the mean of the *projected* cube corners treated as a direction
from the world origin. The correct outward reference for a cube face is
`face_center - cube_center`. What this actually tests is "does this face point
into the same hemisphere as the cube's offset from origin", which trips for
roughly 3 of 6 local faces on every `face_id`. Tell-tale: the trip set changes
as the 4D drag rotation changes, whereas a genuine chirality property would be
rotation-invariant. `calculate_indices` now uses `face_center - cube_center`
directly (computed per local face from its 4 unique transformed corners).

### Fix

Per-face draws with the right index offset (see #5, whose reorder mechanism
this needs — it turned out cheaper than §5 anticipated, see the note there).
Collapsing to a single 36-index buffer (`draw_indexed(0..36, 0, 0..216)`)
does **not** work: it bakes in face_id 0's winding for all 216 instances,
correct for only the ~27 that belong to that face.

---

## 2. Hoist the 4D rotation out of the per-vertex normal math — FIXED

**Impact: medium. Confidence: verified. Status: fixed**, by hoisting the
rotated face normal, sticker center, and basis vectors once per vertex in
`vs_main`, in `src/shaders/math4d.wgsl`/`shader.wgsl`/`normal_shader.wgsl`/
`depth_shader.wgsl`.

The shared 4D math (`Transform4D`, `is_face_visible`, `compute_world_normal`)
lives once in `src/shaders/math4d.wgsl` and is pulled into `shader.wgsl`,
`normal_shader.wgsl`, and `depth_shader.wgsl` via naga_oil `#import`.

Before the fix, `vs_main` performed **7** `mat4x4 * vec4` multiplies for a
visible vertex:

- 4 inside `compute_world_normal` (`p0`, `pi`, `pj`, `pk`)
- 1 for the vertex position (`transform.rotation_matrix * vertex_4d`)
- 1 in `is_face_visible`
- 1 in the face-gap push (previously `face_push_offset_3d`)

### Fix

`is_face_visible` and the face-gap push both used the same
`instance.face_normal_4d` rotated separately — the same multiplication done
twice, every vertex. `vs_main` now rotates it once:
`let rotated_face_normal = transform.rotation_matrix * instance.face_normal_4d;`
and passes the result to `is_face_visible`, then reuses it for the push.
That rotation happens *before* the visibility check's early-out, so a
culled vertex still pays only this one multiply, same as before the fix —
the win is entirely on the visible path.

By linearity, `R * (center + basis[i]) == R*center + R*basis[i]`, so `vs_main`
also hoists `R*center` and the three `R*basis[i]` once per visible vertex,
after the visibility check, and reuses them for both the normal and the
position path:

```wgsl
let rc  = transform.rotation_matrix * sticker_center_4d;
let rb0 = transform.rotation_matrix * instance.basis[0];
let rb1 = transform.rotation_matrix * instance.basis[1];
let rb2 = transform.rotation_matrix * instance.basis[2];

// position: no further matmul
let rotated_vertex_4d = rc + v.x * rb0 + v.y * rb1 + v.z * rb2;
// normal: p0 = proj(rc), pi = proj(rc + rb_i), etc.
```

`compute_world_normal` took `sticker_center_4d`/`basis`/`rotation_matrix` and
did the rotation internally with a **runtime**-indexed `basis[i]` (forces the
array into scratch memory on most drivers); it now takes the four hoisted,
already-rotated `rc`/`rb0`/`rb1`/`rb2` directly, and its `switch` selects
among the three `rb*` locals instead of indexing an array, eliminating the
dynamic indexing entirely.

A visible vertex now performs **5** `mat4x4 * vec4` multiplies
(`rotated_face_normal`, `rc`, `rb0`, `rb1`, `rb2`), each reused across the
visibility test, normal, position, and push — down from 7, with no separate
multiply duplicated across call sites. `face_push_offset_3d` became a
one-line wrapper around `project_4d_to_3d` once its rotation moved to the
caller, so it was removed and its call sites inlined; its "deliberately not
normalized" doc comment moved to the push call site in each shader.

GPU-side confirmation of the invocation-count reduction needs RenderDoc (see
§"How to collect perf numbers" item 3); this is vertex-shader work, so the
`perf`/CPU-sampling technique used for #3 doesn't show it.

### CPU-side pre-rotation — considered, not pursued

Pre-rotating `basis` and the sticker center **on the CPU** and storing them in
the instance buffer looked attractive (216 instances vs 7,776 vertices), but
`is_face_visible` and the face-gap push both need `R * (a live per-instance
vector)` and would stay GPU-side regardless. More importantly, `rotation_4d`
changes every frame during a drag, so CPU-side pre-rotation would mean
re-deriving and re-uploading all 216 instances' rotated basis/center on every
dragged frame — directly in tension with #3's goal of *not* re-uploading
instances when only the camera/rotation moves. The shader-side hoist above
gets the benefit without that tradeoff.

---

## 3. Per-frame allocations and uploads that rarely change

**Impact: medium. Confidence: verified. Status: fixed.**

All of these used to run every frame regardless of whether anything changed:

| Location | Waste | Status |
|---|---|---|
| `src/shader_widget.rs`, `draw()` | `state.cached_indices.clone()` — 576 B; only changes when `rotation_changed` | Fixed — `cached_indices` is now `Arc<[u16]>`, so this clone is a refcount bump |
| `src/renderer.rs`, `update_indices` | re-uploads all 288 indices every frame, same story | Fixed — generation-gated, skips `queue.write_buffer` when unchanged |
| `src/renderer.rs`, `update_debug_instances` | allocates a fresh `Vec<DebugInstance>` per frame to strip the `distance` field | Fixed — reuses a scratch `Vec` on `Renderer`, `clear()`+`extend()`-ed each frame instead of reallocated |
| `src/shader_widget.rs`, `draw()` | `sticker_instances_for_render` allocates 216 × 96 B ≈ 21 KB every frame | Fixed — regeneration only happens when a move is animating or `Hypercube` state changed; `draw()`'s per-frame copy is now an `Arc<[StickerInstance]>` refcount bump |
| `src/renderer.rs`, `update_sticker_instances` | re-uploads that full ~21 KB every frame | Fixed — generation-gated, same mechanism as `update_indices` |

The instance buffer was the valuable one. When idle (no animation, no move
committed) the instance data is bit-identical frame to frame — only
`hovered_sticker` changes, and that already lives in its own uniform.

### Fix

Generation counters, matching the project's existing `reset_generation`/
`reveal_generation` pattern. `HypercubeShaderState` tags `cached_indices`
and `cached_sticker_instances` (both `Arc<[T]>`) with `indices_generation`/
`sticker_generation`, bumped only when the underlying data is actually
recomputed (indices: `rotation_changed`; instances: a move animation
started, ended, or is still in progress, or the `Hypercube` state changed
via reset). The generation is carried on `HypercubePrimitive` and compared
against `Renderer`'s `last_indices_generation`/`last_sticker_generation` in
`update_indices`/`update_sticker_instances`, which skip the
`queue.write_buffer` call when nothing changed since the last upload.

`update_debug_instances` uses the same reuse-instead-of-reallocate idea:
its `Vec<DebugInstance>` scratch buffer lives on `Renderer` and is
`clear()`+`extend()`-ed each frame rather than allocated fresh — usually a
no-op allocation-wise, since AABB debug mode is off by default and
`debug_instances` is empty.

---

## 4. Only about half the tesseract faces are ever visible

**Impact: medium-high. Confidence: verified. Subsumed by #5.**

`is_face_visible` in the vertex shader culls by emitting off-screen vertices —
the vertex shader still runs for every vertex of every culled instance.

Visibility depends only on `face_id` and `rotation_4d`: **8 booleans per
frame**. `find_intersected_sticker` in `src/ray_casting.rs` already computes
exactly this set on the CPU (`for face_id in 0..8 { if is_face_visible(...) }`)
and discards it.

Skipping invisible faces' instances entirely removes both the vertex work and
`is_face_visible` from the shader. Requires the reordering in #5.

---

## 5. Group instances by `face_id` — unlocks #1 and #4 together

**Impact: highest. Confidence: high. Status: core mechanism (steps 1-2
below) done, cheaper than this section originally assumed — see "What
actually happened."**

### The blocker (as originally assessed)

Instance-buffer order is load-bearing. `find_intersected_sticker`
(`src/ray_casting.rs`) returns a **positional index into `FACET_TABLE`**:

```rust
for (sticker_index, sticker) in FACET_TABLE.iter().enumerate() { ... }
closest_sticker = if sticker.is_actionable { Some(sticker_index) } else { None };
```

That value becomes `state.hovered_sticker` and is used for two things: the
shader compares it against `@builtin(instance_index)` for the hover highlight,
and the click handler in `src/shader_widget.rs` uses it as a `FACET_TABLE`
index to decide which move to trigger. This section originally assumed
reordering the instance buffer would desync it from `FACET_TABLE` and break
hit-testing, and proposed spending a `facet_index: u32` field (96 → 112
bytes/instance, ~17% more upload bandwidth) to bridge the two.

### What actually happened

That assumption was wrong: `build_facet_table()` (`src/piece.rs`) was
reordered face-major *at the source*, so `FACET_TABLE` itself — the one
array every consumer (`generate_sticker_instances`, `find_intersected_sticker`,
the tests) reads live — carries the new order. `sticker_index`,
`@builtin(instance_index)`, and the render order all stay derived from the
same table, so they never desync. No `facet_index` field, no shader change,
no bandwidth cost.

1. ~~Sort instances by `face_id`~~ — done via `build_facet_table()`'s
   iteration order (8 contiguous blocks of 27; see `FACET_TABLE`'s doc
   comment).
2. ~~Issue up to 8 draws, each over one instance range with its own 36-index
   chunk offset~~ — done in `Renderer::render()`. **Fixes #1.**
3. Skip the ranges for faces `is_face_visible` rejects, so `is_face_visible`
   can leave the vertex shader entirely — **still open, fixes #4**. The loop
   this plugs into already exists in `Renderer::render()`.

Step 3 alone (once done) is roughly another 2× less vertex work on top of
#1's 8×.

---

## Bonus, low value

`StickerInstance.basis` is 48 of the struct's 96 bytes, but statically there are
only **4 distinct values** — `unit_vectors(free_axes(axis))` for `axis` in
0..4. A `u32` basis-id plus a small uniform lookup table would halve instance
bandwidth. It conflicts with animation, though: mid-turn instances carry
arbitrary rotated bases, so it needs a flag or a separate path. Probably not
worth it unless the instance upload shows up in a profile after #3.

This competes with #5 for the same struct bytes — neither is free. If both
ever land, do this one first and let #5's `facet_index` reuse the bandwidth
this frees up instead of growing the struct further.
