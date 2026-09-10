# Performance improvement ideas

Tracks optimization ideas evaluated during perf diagnosis but not yet
implemented, so they aren't re-investigated from scratch later. This is
distinct from the rationale comments already inline in `elemental_shader.wgsl`
(e.g. Dirt's step budget, Moss's octave trim), which document *why* a change
that already shipped was made — this file is for ideas still pending a
decision or a prerequisite check.

Note: `CLAUDE.md` and `benches/instances.rs` both reference this file's
"item #3" (the CPU-only per-frame allocation sites benchmarked by
`cargo bench` — `HypercubeShaderProgram::calculate_indices` and
`sticker_instances_for_render`). That original file no longer exists on disk
and its earlier items' content wasn't recoverable; this is a fresh start,
renumbered from 1. The functions those historical benchmarks target are
still real and still benchmarked — only this tracking doc was lost.

## 1. Depth pre-pass to occlusion-cull overdrawn translucent materials (Fire/Ice/Light/Dirt)

**Status:** not implemented. Gated on confirming overdraw is actually
significant for these materials at typical camera/scramble states (see
Prerequisite below) — this is a real architectural addition, not worth
building speculatively.

**Problem:** Fire/Ice/Light/Dirt are drawn without depth write, back-to-front
blended (`group_depth_batches`/`topological_draw_order` in
`shader_widget.rs`/`renderer.rs`), each running an expensive raymarch
(16-36 steps depending on material). If several stickers of these kinds
overlap on screen (plausible under a 4D-projected view), every overlapping
layer pays its full raymarch cost regardless of whether a nearer layer has
already gone fully opaque and is hiding it.

**Approach — two passes per material instead of one:**
1. **Depth-only pre-pass**: run the same raymarch, but instead of writing
   color, write depth (`@builtin(frag_depth)`) at the point the raymarch
   becomes "solid" (e.g. Light already has this as its existing
   `transmittance < 0.02` break condition — that point is a natural
   candidate for the solid depth). Fragments that never reach that
   threshold either discard or write the far span distance (no occlusion
   contributed).
2. **Color/blend pass** (today's pass, mostly unchanged): depth-test
   (`Less`) against the buffer the pre-pass populated, depth write off,
   same raymarch + blend as now.

**Critical caveat — `frag_depth` disables early-Z for the pass that writes
it.** WGSL/WebGPU has no "conservative depth" qualifier (unlike HLSL's
`SV_DepthLessEqual`/GLSL's `layout(depth_less)`), so once a shader writes
`frag_depth`, the GPU can't reject fragments before running that shader.
This means:
- The **pre-pass gets no automatic hardware occlusion benefit among its own
  overlapping boxes** — every box's raymarch still runs once in the
  pre-pass, front-to-back sorting or not. Not free.
- The actual win is entirely in the **second (color) pass**: it doesn't
  write custom depth, so it's a standard depth comparison and hardware
  early-Z genuinely applies there — a box whose nearest possible surface is
  already behind the recorded solid depth from a nearer sticker gets
  rejected before its raymarch shader ever runs.

**Net effect:** for a stack of N overlapping stickers where the front one
goes solid partway through its march, today's cost is N full raymarches;
with this scheme it's N raymarches in the pre-pass (no cheaper — still have
to find each box's solid point) *plus* only the non-occluded ones' raymarches
again in the color pass. Only pays off when there's real depth stacking
*and* the material reliably goes opaque partway through its box (thin/wispy
materials that rarely saturate transmittance won't benefit much).

**Cost of building this:** new depth-only pipeline per material, doubled
draw calls per depth-batch group, need front-to-back ordering info for the
pre-pass in addition to the existing back-to-front order for the color pass's
blending correctness.

**Prerequisite before building:** use RenderDoc's Quad Overdraw overlay
(Texture Viewer → Overlay → Quad Overdraw) to confirm these materials
actually show meaningful screen-space stacking (visibly hot/white regions)
at representative camera angles/scrambled states. If overdraw is low, this
is a lot of new pipeline complexity for little payoff — the iteration-count
lever (raymarch step counts themselves) is a better use of effort in that
case.
