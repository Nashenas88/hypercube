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

**Status:** implemented for Fire, Light and Dirt (`fire_depth_prepass_pipeline`/
`light_depth_prepass_pipeline`/`dirt_depth_prepass_pipeline` in `renderer.rs`,
drawn front-to-back into the existing `depth_view` ahead of the existing
back-to-front color passes). Ice was left untouched — its pipeline already
writes real depth with opaque blending, so it was never an overdraw offender
and needs no prepass.

RenderDoc's Quad Overdraw overlay confirmed real screen-space stacking for
these materials before this was built, which is what motivated going ahead
with it (see the original Prerequisite note below, kept for context).

**The design that shipped is not the one first sketched below.** The
original idea was a prepass that reruns each material's full fragment
shader (skipping only the final color write) to get a depth value. Working
through the cost more carefully surfaced a real flaw: writing
`@builtin(frag_depth)` disables hardware early-Z for the pass that writes
it (see the caveat below — still accurate), so a *full-shading* prepass
would cost nearly as much as the real color pass. For a lightly-occluded
frame (the common case away from a fully scrambled, camera-stacked view),
every visible fragment would then pay for both the prepass's full march
*and* the color pass's full march — a net loss, not a win.

The fix that shipped instead: the prepass fragment shaders
(`fs_fire_depth`/`fs_light_depth`/`fs_dirt_depth` in
`elemental_shader.wgsl`) are **alpha-only** — they run the same hit-finding
march as the color shader (via a shared `fire_plasma_march`/
`light_cloud_march`/`dirt_march` helper) but skip every computation that
only feeds color, never alpha/hit:
- Light's nested self-shadow sub-march (`LIGHT_CLOUD_SHADOW_STEPS`) and the
  `light_color`/`ambient_color`/`scatter` terms it feeds.
- Dirt's `dirt_normal`/`dirt_ao` and its color `dirt_fbm` octaves, all of
  which only run once after the march already breaks.
- Fire's `sun_palette` lookup and per-step color accumulation — a smaller
  saving than Light/Dirt's, since Fire's density calculation feeds color
  and alpha about equally (see its own doc comment in the shader).

`dirt_march` is shared verbatim by both `fs_dirt` and `fs_dirt_depth` (its
march loop was already 100% alpha/hit-only, so there was nothing to
duplicate). `fire_plasma_march`/`light_cloud_march` are used only by the
new depth entry points — `fs_fire`/`fs_light` keep their own original loops
unchanged, since their color accumulation needs every step's transmittance
value, not just the final one, and WGSL has no closures/function pointers
to share one loop body between a shading and a non-shading caller.

**Follow-up attempted and reverted: a manual stand-in for early-Z inside the
prepass itself.** The alpha-only prepass above still can't use hardware
early-Z on itself (writing `frag_depth` disables it for the writing pass,
same caveat as below), so within one prepass batch, a fragment fully
occluded by an *earlier* batch still pays its full march before the GPU's
ordinary post-shader depth test throws the write away. Tried fixing this
with a per-batch snapshot of `depth_view` (`depth_prepass_snapshot_texture`
in `renderer.rs`, copied fresh before every batch, mirroring the pattern
`ice_bg_texture` already uses for Ice) that each batch's shaders would
sample to discard a fragment before marching if its box's nearest possible
point already couldn't beat the snapshot.

This was implemented, passed `cargo test` (byte-identical golden image) and
Vulkan-backed local testing, but **failed under RenderDoc's GL backend in
three successive ways**: `textureLoad` on a depth texture isn't supported by
naga's GLSL backend at all; switching to a plain (non-comparison)
`textureSampleLevel` still failed because naga's GLSL backend always maps a
WGSL depth texture to a GLSL `sampler2DShadow` regardless of the declared
sampler kind, and a `sampler2DShadow` only has comparison-sample overloads;
switching again to a real comparison sample (`textureSampleCompare` with a
`sampler_comparison`, `GreaterEqual`) finally compiled under GL, but at
runtime silently broke the *entire* occlusion effect - RenderDoc showed
full overdraw again, exactly as if nothing had ever been culled. The
working theory (not confirmed against wgpu-hal source, which wasn't
reachable in this environment): `copy_texture_to_texture` between two
`Depth32Float` textures doesn't actually work on the GL backend, so the
snapshot stayed at its near-zero default - making every fragment's
"nearest possible depth" compare as `>= 0`, i.e. always "already occluded,"
so the prepass discarded every fragment before writing any real occlusion
depth at all.

Given three GL-specific failures in a row from a mechanism that samples a
depth texture, versus zero from the base alpha-only prepass (which only
ever *writes* `frag_depth`, a universally-supported operation), this was
reverted rather than debugged further. If revisited, the fix should avoid
sampling a `Depth32Float` texture as a depth texture at all - e.g. mirror
depth into a plain `R32Float` color attachment (via an ordinary
`@location(0)` write, not a depth-format copy) and read *that* back
normally, since ordinary color textures don't hit any of naga's
depth-texture-specific GLSL gaps.

Not yet re-measured with the `gpu-capture-hooks` + RenderDoc workflow after
landing — in particular, whether Fire's prepass draws pay for themselves is
still an open question per its weaker expected win (see its doc comment);
the fallback if not is to drop Fire from the three loops in `render()` and
leave its color pipeline as the sole draw, same as Ice today.

**Status of the analysis below:** kept as background for *why* this needed
more than the naive version — the caveat and net-effect analysis are still
correct, they just describe the design that was rejected in favor of the
alpha-only one above.

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
