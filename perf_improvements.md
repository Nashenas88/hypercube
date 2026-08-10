# Rendering performance improvements

Findings from a review of the render hot path (2026-08-09), written up for a
later session. Nothing here is implemented yet.

Context: these were found while fixing the sticker-normal desync, which moved
normal computation out of a static 48-entry uniform table and into
`compute_world_normal` in the vertex shader, derived per-instance from
`instance.basis`. That change removed the `NormalsUniform` /  `cached_normals`
system entirely; `calculate_normals_and_indices` became `calculate_indices`.

Line numbers are from that commit and will drift — grep for the quoted code.

**Suggested order:** #1 and #3 first (safe, independent, high value), then #2
(contained shader edit), then #5 last. #5 is the real payoff but is the only
restructure, and doing it after #1 means the winding-correction heuristic gets
exercised for the first time before you build on it. #4 is subsumed by #5.

---

## 1. Every instance draws the cube 8 times (8× overdraw)

**Impact: high. Confidence: verified.**

`src/renderer.rs`, in `render()`:

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

Note the heuristic it uses is itself suspect and was deliberately left alone
during the normals fix:

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
rotation-invariant. Expect to revisit this once #1 makes it live.

### Fix

Either issue per-face draws with the right index offset (see #5, which is the
clean way to get there), or — if the per-face winding variants turn out to be
unnecessary once normals are geometric — collapse to a single 36-index buffer
and `draw_indexed(0..36, 0, 0..216)`.

---

## 2. Hoist the 4D rotation out of the per-vertex normal math

**Impact: medium. Confidence: safe, contained.**

`src/shaders/shader.wgsl` (and the identical copy in `normal_shader.wgsl`).
Per vertex, `vs_main` currently performs 6 `mat4x4 * vec4` multiplies:

- 4 inside `compute_world_normal` (`p0`, `pi`, `pj`, `pk`)
- 1 for the vertex position (`transform.rotation_matrix * vertex_4d`)
- 1 in `is_face_visible`

By linearity, `R * (center + basis[i]) == R*center + R*basis[i]`. Hoisting
`R*center` and the three `R*basis[i]` once per vertex serves **both** the
normal and the position path:

```wgsl
let rc  = transform.rotation_matrix * sticker_center_4d;
let rb0 = transform.rotation_matrix * instance.basis[0];
let rb1 = transform.rotation_matrix * instance.basis[1];
let rb2 = transform.rotation_matrix * instance.basis[2];

// position: no further matmul
let rotated_vertex_4d = rc + v.x * rb0 + v.y * rb1 + v.z * rb2;
// normal: p0 = proj(rc), pi = proj(rc + rb_i), etc.
```

### The bigger win hiding in there

`compute_world_normal` indexes `basis[i]` with a **runtime** `i` produced by its
`switch`. Dynamic indexing of an `array<vec4<f32>, 3>` forces it into scratch
memory on most drivers. Rewriting the six switch arms to select directly among
the three hoisted `vec4` locals eliminates the dynamic indexing entirely.

### Rejected alternative — do not do this

Pre-rotating `basis` and the sticker center **on the CPU** and storing them in
the instance buffer looks attractive (216 instances vs 7,776 vertices) but does
not work cleanly: `sticker_center_4d` is derived in-shader from
`transform.face_spacing`, a live UI slider. Moving it CPU-side couples the CPU
to slider state and pulls the face-spread math out of the shader — precisely
where `sticker_instances_for_render`'s animation compensation assumes it lives
(see the long comment about subtracting the shader's fixed unrotated
contribution back out). `is_face_visible` would still need `R * face_center_4d`
regardless.

---

## 3. Per-frame allocations and uploads that rarely change

**Impact: medium. Confidence: verified. No design risk.**

All of these run every frame regardless of whether anything changed:

| Location | Waste |
|---|---|
| `src/shader_widget.rs`, `draw()` | `state.cached_indices.clone()` — 576 B; only changes when `rotation_changed` |
| `src/renderer.rs`, `update_indices` | re-uploads all 288 indices every frame, same story |
| `src/renderer.rs`, `update_debug_instances` | allocates a fresh `Vec<DebugInstance>` per frame to strip the `distance` field |
| `src/shader_widget.rs`, `draw()` | `sticker_instances_for_render` allocates 216 × 96 B ≈ 21 KB every frame |
| `src/renderer.rs`, `update_sticker_instances` | re-uploads that full ~21 KB every frame |

The instance buffer is the valuable one. When idle (no animation, no move
committed) the instance data is bit-identical frame to frame — only
`hovered_sticker` changes, and that already lives in its own uniform.

Fix: dirty flags. Regenerate/upload sticker instances only when
`animating_move.is_some()` or the `Hypercube` state changed; upload indices
only when `rotation_changed`; keep a reusable scratch `Vec` (or store debug
instances in GPU-ready form) for the debug AABB path.

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

**Impact: highest. Confidence: high, but this is the one restructure.**

### The blocker

Instance-buffer order is load-bearing. `find_intersected_sticker`
(`src/ray_casting.rs`) returns a **positional index into `FACET_TABLE`**:

```rust
for (sticker_index, sticker) in FACET_TABLE.iter().enumerate() { ... }
closest_sticker = if sticker.is_actionable { Some(sticker_index) } else { None };
```

That value becomes `state.hovered_sticker` and is used for two things: the
shader compares it against `@builtin(instance_index)` for the hover highlight,
and the click handler in `src/shader_widget.rs` uses it as a `FACET_TABLE`
index to decide which move to trigger. Reorder or filter the instance buffer
and you break hit-testing, not just highlighting.

### The unlock

`StickerInstance` (`src/piece.rs`) already carries `_padding: [u32; 3]` — **12
free bytes**. Spend 4 of them on a `facet_index: u32`, have the shader compare
*that* against `highlighting.hovered_sticker_index` instead of
`instance_index`, and the instance buffer becomes free to reorder at zero size
cost.

Then:

1. Sort instances by `face_id` (contiguous ranges, 8 groups).
2. Issue up to 8 draws, each over one instance range with its own 36-index
   chunk offset → fixes **#1**, and finally applies each face's winding
   correction to the instances it was computed for.
3. Skip the ranges for faces `is_face_visible` rejects → fixes **#4**, and
   `is_face_visible` can leave the vertex shader entirely.

Combined with #1 that is roughly **8× × 2 ≈ 16× less vertex work**.

Watch out for: `Renderer::num_stickers` assumes one flat range; the hover
comparison in both `shader.wgsl` and `normal_shader.wgsl` must change together;
and the sort should be computed once (it is static — `FACET_TABLE` order and
`face_id` are both fixed at startup), not per frame.

---

## Bonus, low value

`StickerInstance.basis` is 48 of the struct's 96 bytes, but statically there are
only **4 distinct values** — `unit_vectors(free_axes(axis))` for `axis` in
0..4. A `u32` basis-id plus a small uniform lookup table would halve instance
bandwidth. It conflicts with animation, though: mid-turn instances carry
arbitrary rotated bases, so it needs a flag or a separate path. Probably not
worth it unless the instance upload shows up in a profile after #3.
