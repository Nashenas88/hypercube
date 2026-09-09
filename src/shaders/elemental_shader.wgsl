#import math4d::{camera, compute_sticker_anchor, compute_vertex_geometry, instances, inverse3, transform}
#import sticker_common::{HighlightingUniform, LightUniform, light, highlighting, piece_slots}
#import elemental_common::{hash11, hash21, hash31, value_noise2, value_noise3}

// Ice's own bind group: iChannel0 (gray noise, for the triplanar bump map)
// and iChannel1 (a snapshot of the rendered scene so far, copied fresh by
// `render()` before each Ice depth layer's pass - see `ice_sample_background`).
// `main_bind_group_layout` at group 0 has no free texture slots, so Ice's
// pipeline binds this as a second group instead.
@group(1) @binding(0) var ice_noise_texture: texture_2d<f32>;
@group(1) @binding(1) var ice_noise_sampler: sampler;
@group(1) @binding(2) var ice_background_texture: texture_2d<f32>;
@group(1) @binding(3) var ice_background_sampler: sampler;

// `ice_background_texture` is sized to the whole window, but the 3D scene
// only occupies `Renderer::bounds` within it (a sub-rectangle - the shader
// widget's viewport, e.g. below the menu bar). Maps NDC (spanning that
// sub-rectangle) into the texture's own `[0,1]` UV space, recomputed every
// frame in `Renderer::update_camera` since either `bounds` or the texture's
// size can change between frames.
struct IceBackgroundTransform {
    scale: vec2<f32>,
    offset: vec2<f32>,
}
@group(1) @binding(4) var<uniform> ice_background_transform: IceBackgroundTransform;

// Resolves `vs_main_batched`'s raw `@builtin(instance_index)` to the real
// sticker index it stands in for - see `vs_main_batched`.
@group(0) @binding(8) var<storage, read> depth_batch_remap: array<u32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) instance_index: u32,
    @location(3) piece_slot: u32,
    @location(4) kind: u32,
    // Raw, untransformed mesh-space vertex position (before
    // `transform.sticker_scale`) - the `vertex_position` attribute
    // unmodified. Water uses this to recover each fragment's local face
    // identity and in-plane UV without re-deriving `face_3d`.
    @location(5) local_position: vec3<f32>,
}

fn vs_body(instance_index: u32, vertex_position: vec3<f32>, vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    let geometry = compute_vertex_geometry(instance_index, vertex_position, vertex_index);

    out.clip_position = geometry.clip_position;
    out.world_position = geometry.world_position;
    out.world_normal = geometry.world_normal;
    out.local_position = vertex_position;

    if (!geometry.visible) {
        out.instance_index = instance_index;
        out.piece_slot = 0u;
        out.kind = 0u;
        return out;
    }

    out.instance_index = instance_index;
    out.piece_slot = piece_slots[instance_index];
    out.kind = instances[instance_index].kind;

    return out;
}

@vertex
fn vs_main(
    @location(0) vertex_position: vec3<f32>,
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    return vs_body(instance_index, vertex_position, vertex_index);
}

// Used by `fire_pipeline`/`ice_pipeline` instead of `vs_main`: a depth
// batch's same-kind, same-face stickers draw with one instanced
// `draw_indexed` call over a contiguous run of `depth_batch_remap`
// (`Renderer::update_depth_batches`), rather than one call per sticker, so
// `@builtin(instance_index)` here is a position in that run, not the
// sticker's own identity - resolved through `depth_batch_remap` first, so
// every downstream identity use (highlighting, per-sticker noise seeds,
// `instances`/`piece_slots` lookups) sees the real sticker index.
@vertex
fn vs_main_batched(
    @location(0) vertex_position: vec3<f32>,
    @builtin(instance_index) draw_instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    let instance_index = depth_batch_remap[draw_instance_index];
    return vs_body(instance_index, vertex_position, vertex_index);
}

// Fragment shader

// Blends the hovered-sticker and hovered-piece tints into an already-shaded
// color, the sticker taking precedence over the piece it belongs to.
// `coverage` scales the tint the way the color it is mixed into is scaled:
// 1.0 for an opaque material, and the fragment's own alpha for one whose
// color is premultiplied by it.
fn apply_highlight(color: vec3<f32>, coverage: f32, instance_index: u32, piece_slot: u32) -> vec3<f32> {
    if (instance_index == highlighting.hovered_sticker_index) {
        return mix(color, highlighting.highlight_color.rgb * coverage, highlighting.highlight_color.a);
    }
    if (piece_slot == highlighting.hovered_piece_slot) {
        return mix(color, highlighting.piece_highlight_color.rgb * coverage, highlighting.piece_highlight_color.a);
    }
    return color;
}

// Water: a bump-mapped sea surface applied per sticker facet rather than
// vertex-displaced. Each of a face's 27 water stickers renders its own
// independent-looking wave patch: the local UV domain is
// always the sticker's own -1..1 mesh square (normalized by
// `STICKER_HALF_EXTENT`, defined below in the Fire section), so without a
// per-instance offset every sticker sharing a face normal would sample the
// identical pattern - `water_wave_height`'s `instance_seed` is what makes
// each one distinct.

const SEA_HEIGHT: f32 = 0.12;
const SEA_CHOPPY: f32 = 3.5;
const SEA_SPEED: f32 = 0.8;
const SEA_FREQ: f32 = 1.8;

const SEA_BASE: vec3<f32> = vec3<f32>(0.005, 0.03, 0.09);
const SEA_WATER_COLOR: vec3<f32> = vec3<f32>(0.06, 0.28, 0.45);

// Half-extent, in the normalized units `water_face_uv`/`water_edge_mask`
// work in, of the local UV domain a sticker's wave field is sampled over.
// Local positions are normalized by `STICKER_HALF_EXTENT` before use, so
// this is always 1.0 regardless of the mesh's real size.
const WATER_BOX_EXTENT: f32 = 1.0;

const WATER_OCTAVE_M: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1.6, -1.2), vec2<f32>(1.2, 1.6));
const WATER_UV_ROT: mat2x2<f32> = mat2x2<f32>(vec2<f32>(0.819, -0.573), vec2<f32>(0.573, 0.819));

// 2D value noise in -1..1, ported from the source shader's `noise()`. Built
// on `hash21` (identical formula to the source's own `hash`) rather than a
// second copy of it.
fn water_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let h00 = hash21(i + vec2<f32>(0.0, 0.0));
    let h10 = hash21(i + vec2<f32>(1.0, 0.0));
    let h01 = hash21(i + vec2<f32>(0.0, 1.0));
    let h11 = hash21(i + vec2<f32>(1.0, 1.0));
    return -1.0 + 2.0 * mix(mix(h00, h10, u.x), mix(h01, h11, u.x), u.y);
}

fn water_sea_octave(uv_in: vec2<f32>, choppy: f32) -> f32 {
    let uv = uv_in + water_noise(uv_in);
    var wv = 1.0 - abs(sin(uv));
    let swv = abs(cos(uv));
    wv = mix(wv, swv, wv);
    return pow(1.0 - pow(wv.x * wv.y, 0.65), choppy);
}

// Wave height at `uv_in` (already in the sticker's own -1..1 local domain)
// for a facet whose local (pre-rotation) mesh axis normal is `face_normal`.
// `instance_index` seeds an offset into the wave field so each of the 27
// water stickers on a face samples an independent-looking patch instead of
// literally the same pattern.
fn water_wave_height(uv_in: vec2<f32>, face_normal: vec3<f32>, instance_index: u32) -> f32 {
    let sea_time = 1.0 + transform.elapsed_seconds * SEA_SPEED;
    let instance_seed = vec2<f32>(
        hash11(f32(instance_index) * 0.173) * 100.0,
        hash11(f32(instance_index) * 0.371) * 100.0,
    );
    let face_seed = vec2<f32>(
        dot(face_normal, vec3<f32>(12.3, 45.6, 78.9)),
        dot(face_normal, vec3<f32>(98.7, 65.4, 32.1)),
    ) + instance_seed;
    var uv = (uv_in + face_seed) * WATER_UV_ROT;
    var freq = SEA_FREQ;
    var amp = SEA_HEIGHT;
    var choppy = SEA_CHOPPY;
    var h: f32 = 0.0;
    for (var i = 0; i < 5; i++) {
        var d = water_sea_octave((uv + vec2<f32>(sea_time)) * freq, choppy);
        d += water_sea_octave((uv - vec2<f32>(sea_time)) * freq, choppy);
        h += d * amp;
        uv = WATER_OCTAVE_M * uv;
        freq *= 1.9;
        amp *= 0.22;
        choppy = mix(choppy, 1.0, 0.2);
    }
    return h;
}

// Extracts the two in-plane local coordinates of `p` for a facet whose
// dominant mesh axis is `n`. Pure local-mesh geometry with no water-specific
// meaning - Lightning reuses this too, to recover its own local face UV.
fn sticker_face_uv(p: vec3<f32>, n: vec3<f32>) -> vec2<f32> {
    let abs_n = abs(n);
    if (abs_n.x > 0.5) {
        return p.zy;
    }
    if (abs_n.y > 0.5) {
        return p.xz;
    }
    return p.xy;
}

// Fades the wave field to 0 near a facet's own border, in the same
// normalized -1..1 domain `sticker_face_uv` returns, so neighboring stickers'
// independently-seeded patches don't show a hard seam at the tile edge.
fn water_edge_mask(uv: vec2<f32>, box_extent: f32) -> f32 {
    let border = vec2<f32>(box_extent) - vec2<f32>(0.06);
    let d = max(abs(uv) - border, vec2<f32>(0.0));
    let dist = length(d);
    return smoothstep(0.06, 0.0, dist);
}

// The mesh-local (pre-rotation) axis normal for whichever of a sticker's 6
// cube faces the current fragment belongs to. Every vertex of one cube face
// shares the same dominant coordinate (pinned to +/-STICKER_HALF_EXTENT by
// `CUBE_VERTICES`, and each face is its own dedicated set of 6 vertices, not
// shared with neighboring faces), so the rasterizer's interpolation of
// `local_position` keeps that coordinate exact regardless of the fragment's
// position within the triangle - no `face_3d` plumbing needed. Pure
// local-mesh geometry with no water-specific meaning - Lightning reuses this
// too, to recover its own local face identity.
fn sticker_local_face_normal(local_position: vec3<f32>) -> vec3<f32> {
    let a = abs(local_position);
    if (a.x >= a.y && a.x >= a.z) {
        return vec3<f32>(sign(local_position.x), 0.0, 0.0);
    } else if (a.y >= a.x && a.y >= a.z) {
        return vec3<f32>(0.0, sign(local_position.y), 0.0);
    }
    return vec3<f32>(0.0, 0.0, sign(local_position.z));
}

// Bump-mapped normal in the sticker's own local (pre-rotation) frame, from
// finite-differencing `water_wave_height` across the facet's UV.
fn water_face_normal(
    uv: vec2<f32>,
    face_normal: vec3<f32>,
    box_extent: f32,
    instance_index: u32,
) -> vec3<f32> {
    let eps = vec2<f32>(0.005, 0.0);
    let mask = water_edge_mask(uv, box_extent);
    let h0 = water_wave_height(uv, face_normal, instance_index) * mask;
    let hx = water_wave_height(uv + eps.xy, face_normal, instance_index)
        * water_edge_mask(uv + eps.xy, box_extent) - h0;
    let hy = water_wave_height(uv + eps.yx, face_normal, instance_index)
        * water_edge_mask(uv + eps.yx, box_extent) - h0;
    let abs_n = abs(face_normal);
    var wave_n: vec3<f32>;
    if (abs_n.x > 0.5) {
        wave_n = vec3<f32>(sign(face_normal.x), -hy / eps.x, -hx / eps.x);
    } else if (abs_n.y > 0.5) {
        wave_n = vec3<f32>(-hx / eps.x, sign(face_normal.y), -hy / eps.x);
    } else {
        wave_n = vec3<f32>(-hx / eps.x, -hy / eps.x, sign(face_normal.z));
    }
    return normalize(wave_n);
}

// Shades the water surface at a perturbed world-space normal `n`. `eye` is
// the *incident* view direction (camera -> surface, i.e.
// `world_position - camera.eye_position.xyz`, normalized) - the opposite
// sign from this file's usual `view_dir` (surface -> camera) - because both
// the fresnel term and `reflect` below are written for that convention;
// getting the sign backwards silently inverts the specular highlight and
// the fresnel rim. `light_dir` is surface -> light, this file's usual
// convention.
fn water_shade(n: vec3<f32>, eye: vec3<f32>, light_dir: vec3<f32>) -> vec3<f32> {
    var fresnel_term = clamp(1.0 - max(dot(n, -eye), 0.0), 0.0, 1.0);
    fresnel_term = pow(fresnel_term, 3.0) * 0.5;
    let diff = max(dot(n, light_dir), 0.0);
    let sky_color = vec3<f32>(0.35, 0.55, 0.75);
    let water_base = mix(SEA_BASE, SEA_WATER_COLOR, diff * 0.7 + 0.3);
    var color = mix(water_base, sky_color, fresnel_term);
    let refl = reflect(eye, n);
    let spec_broad = pow(max(dot(refl, light_dir), 0.0), 32.0) * 0.6;
    let spec_tight = pow(max(dot(refl, light_dir), 0.0), 64.0) * 0.8;
    color += vec3<f32>(spec_broad + spec_tight);
    return color;
}

fn water_color(
    instance_index: u32,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    local_position: vec3<f32>,
) -> vec3<f32> {
    let local_face_normal = sticker_local_face_normal(local_position);
    let normalized_local = local_position / STICKER_HALF_EXTENT;
    let local_uv = sticker_face_uv(normalized_local, local_face_normal);

    // Map the local bump-mapped normal into world space through the
    // sticker's own projected frame - the same technique `fs_fire` uses for
    // `to_local`, just applied to a direction rather than a ray. A near-
    // degenerate frame (a facet viewed edge-on under a 4D rotation) falls
    // back to the flat `world_normal` instead of normalizing a near-zero
    // vector: unlike Fire's blended pass, this opaque pass can't discard the
    // fragment without poking a hole in the surface behind it.
    let anchor = compute_sticker_anchor(instance_index, STICKER_HALF_EXTENT * transform.sticker_scale);
    let frame_volume = dot(anchor.edge_x, cross(anchor.edge_y, anchor.edge_z));
    let edge_volume = length(anchor.edge_x) * length(anchor.edge_y) * length(anchor.edge_z);

    var perturbed_world_normal = normalize(world_normal);
    if (anchor.visible && abs(frame_volume) >= edge_volume * STICKER_MIN_FRAME_VOLUME) {
        let to_world = mat3x3<f32>(anchor.edge_x, anchor.edge_y, anchor.edge_z);
        let local_normal = water_face_normal(local_uv, local_face_normal, WATER_BOX_EXTENT, instance_index);
        perturbed_world_normal = normalize(to_world * local_normal);
    }

    let eye = normalize(world_position - camera.eye_position.xyz);
    let light_dir = normalize(-light.direction);

    let color = water_shade(perturbed_world_normal, eye, light_dir);
    return pow(color, vec3<f32>(0.75));
}

// Dirt: a raymarched bumpy rock/earth surface filling each sticker's own
// local cube. Unlike Water/Moss, which bump-map a flat quad, and unlike
// Ice's own opaque raymarch, Dirt's rock box (extent ~0.93) doesn't reach a
// sticker's full local extent (1.0), so it has real transparent regions at
// a facet's own edges/corners - not just under a degenerate 4D frame. That
// makes it a fourth member of Fire/Ice/Light's back-to-front batched,
// blended pass (`fs_dirt`, drawn by `dirt_pipeline` - see `renderer.rs` and
// `shader_widget::DepthLayer`) rather than a material shaded inline in
// `fs_main`: an opaque pass has no way to let a farther sticker's own
// material show correctly through those gaps under 4D perspective.
// Raymarched in the sticker's own local frame exactly like Fire
// (`compute_sticker_anchor`/`inverse3`), but with a hard SDF hit/miss
// instead of a density accumulation, seeded by `piece_slot` like Fire so a
// piece's own rock pattern travels with it through a move rather than
// jumping to a new one. Ported from a standalone Shadertoy-style source
// whose own domain was already normalized to roughly a unit box - the same
// convention `compute_sticker_anchor`'s local frame uses (1.0 == the
// sticker mesh cube's own half-extent, see `STICKER_HALF_EXTENT`) - so the
// source's box and bump literals below carry over unchanged.

// Bounding test for `ray_box`, not the surface itself: the box (extent 0.7)
// plus its bump displacement (up to ~0.18+0.05) reaches at most ~0.93 from
// center, so 0.95 gives a small safety margin - mirrors Fire's
// `FIRE_BASE_EXTENT + FIRE_SURFACE_DISPLACEMENT` bound for `ray_box`.
const DIRT_BOUND_EXTENT: f32 = 0.95;

// Sphere-trace budget. The source used 96 steps from the open camera ray;
// `ray_box` above already starts this march right at the box's own
// boundary instead, eliminating most of that travel for free. What's left
// is sized between Fire's 16 steps and Light's 28 (`LIGHT_CLOUD_STEPS`),
// both of which also cover scenes that can show many simultaneous facets.
const DIRT_MARCH_STEPS: i32 = 36;
const DIRT_MARCH_EPSILON: f32 = 0.003;
const DIRT_STEP_SCALE: f32 = 0.75;

// Octaves for `dirt_map`'s rock bump, evaluated on every march/normal/AO
// sample - kept low for the same reason Fire's `ridged_warp`/`ridged_detail`
// use 2 octaves apiece rather than the source's full 5: a facet covers too
// few pixels for the finest octaves to resolve into anything legible.
const DIRT_BUMP_OCTAVES: i32 = 3;

// Octaves for the one-time macro/micro color noise sampled once per hit
// (not per march step) - kept at the source's own 5, since this call is
// cheap relative to the march.
const DIRT_COLOR_OCTAVES: i32 = 5;

// Reduced from the source's 5-sample AO loop for the same per-step cost
// reason as `DIRT_BUMP_OCTAVES`.
const DIRT_AO_STEPS: i32 = 3;

fn dirt_sd_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// The project's own `value_noise3` (trilinear value noise) stands in for
// the source's own hand-rolled `hash`/`noise` pair - both are the same
// construction, and `value_noise3` is deliberately the non-`sin`-based
// version this file already prefers for heavily-per-pixel volume sampling
// (see `elemental_common.wgsl`). `octaves` is a parameter, not the source's
// fixed 5, so the same function serves both `dirt_map`'s cheap per-step
// bump (`DIRT_BUMP_OCTAVES`) and the hit point's richer one-shot color
// noise (`DIRT_COLOR_OCTAVES`).
fn dirt_fbm(p_in: vec3<f32>, octaves: i32) -> f32 {
    var p = p_in;
    var val = 0.0;
    var amp = 0.5;
    for (var i = 0; i < octaves; i++) {
        val += amp * value_noise3(p);
        p *= 2.07;
        amp *= 0.5;
    }
    return val;
}

// Scaled box (size 0.7) displaced outwards by up to ~0.23 units - ported
// verbatim from the source's `map`, whose own domain already matched this
// file's local-frame convention (see the section comment above).
// `seed_offset` shifts which patch of the (otherwise infinite) noise field
// this instance samples, so neighboring Dirt facets - and a piece's own
// facet before and after a move - don't all show the identical pattern.
fn dirt_map(p: vec3<f32>, seed_offset: vec3<f32>) -> f32 {
    let sp = p + seed_offset;
    let rock_bump = dirt_fbm(sp * 4.0, DIRT_BUMP_OCTAVES) * 0.18
        + dirt_fbm(sp * 12.0, DIRT_BUMP_OCTAVES) * 0.05;
    return dirt_sd_box(p, vec3<f32>(0.7)) - rock_bump;
}

// Ported verbatim from the source's `calcNormal`: a 3-tap gradient that
// reuses the hit distance itself as a 4th sample instead of a full 6-tap
// central difference.
fn dirt_normal(p: vec3<f32>, seed_offset: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(0.002, 0.0);
    let d = dirt_map(p, seed_offset);
    let n = d - vec3<f32>(
        dirt_map(p - e.xyy, seed_offset),
        dirt_map(p - e.yxy, seed_offset),
        dirt_map(p - e.yyx, seed_offset),
    );
    return normalize(n);
}

// Ported from the source's `calcAO`, generalized from its fixed 5 samples
// to `DIRT_AO_STEPS`.
fn dirt_ao(p: vec3<f32>, n: vec3<f32>, seed_offset: vec3<f32>) -> f32 {
    var occ = 0.0;
    var sca = 1.0;
    for (var i = 0; i < DIRT_AO_STEPS; i++) {
        let h = 0.01 + 0.12 * f32(i) / f32(DIRT_AO_STEPS - 1);
        let d = dirt_map(p + h * n, seed_offset);
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
}

// Dirt's own entry point, drawn per sticker in back-to-front order over
// `dirt_pipeline` - premultiplied blending, no depth write, alongside Fire,
// Ice and Light (see `shader_widget::DepthLayer`). Unlike Fire's volume,
// Dirt's rock is a hard surface: coverage is always 0.0 (fully transparent,
// blends through to whatever is behind) or 1.0 (opaque rock), never a
// density in between.
@fragment
fn fs_dirt(in: VertexOutput) -> @location(0) vec4<f32> {
    let view_dir = normalize(camera.eye_position.xyz - in.world_position);
    if (dot(normalize(in.world_normal), view_dir) <= 0.0) {
        discard;
        return vec4<f32>(0.0);
    }

    let anchor = compute_sticker_anchor(in.instance_index, STICKER_HALF_EXTENT * transform.sticker_scale);
    let to_world = mat3x3<f32>(anchor.edge_x, anchor.edge_y, anchor.edge_z);

    let frame_volume = dot(anchor.edge_x, cross(anchor.edge_y, anchor.edge_z));
    let edge_volume = length(anchor.edge_x) * length(anchor.edge_y) * length(anchor.edge_z);
    if (abs(frame_volume) < edge_volume * STICKER_MIN_FRAME_VOLUME) {
        discard;
        return vec4<f32>(0.0);
    }

    let to_local = inverse3(to_world);
    let ray_origin = to_local * (in.world_position - anchor.world_center);
    let ray_direction = normalize(to_local * (in.world_position - camera.eye_position.xyz));

    let span = ray_box(ray_origin, ray_direction, DIRT_BOUND_EXTENT);
    if (span.x < 0.0) {
        discard;
        return vec4<f32>(0.0);
    }

    // Seeded by `piece_slot` rather than instance index, so a piece keeps
    // its own rock pattern when a move relocates it to a different slot -
    // mirrors Fire's own `seed_offset`.
    let seed = hash11(f32(in.piece_slot) * 0.073) * 4.0;
    let seed_offset = vec3<f32>(seed, seed * 1.3, seed * 0.7);

    var t = span.x;
    var hit = false;
    for (var i = 0; i < DIRT_MARCH_STEPS; i++) {
        let p = ray_origin + ray_direction * t;
        let d = dirt_map(p, seed_offset);
        if (d < DIRT_MARCH_EPSILON) {
            hit = true;
            break;
        }
        t += d * DIRT_STEP_SCALE;
        if (t > span.y) {
            break;
        }
    }

    if (!hit) {
        discard;
        return vec4<f32>(0.0);
    }

    let p = ray_origin + ray_direction * t;
    let n_local = dirt_normal(p, seed_offset);
    let ao = dirt_ao(p, n_local, seed_offset);
    let n_world = normalize(to_world * n_local);

    let light_dir = normalize(-light.direction);
    let diff = max(dot(n_world, light_dir), 0.0);
    let amb = clamp(0.5 + 0.5 * n_world.y, 0.0, 1.0);

    let sp = p + seed_offset;
    let macro_noise = dirt_fbm(sp * 3.5, DIRT_COLOR_OCTAVES);
    let micro_noise = dirt_fbm(sp * 20.0, DIRT_COLOR_OCTAVES);

    let dark_earth = vec3<f32>(0.12, 0.08, 0.05);
    let warm_brown = vec3<f32>(0.32, 0.21, 0.14);
    let sand_highlight = vec3<f32>(0.55, 0.43, 0.32);
    var base_color = mix(dark_earth, warm_brown, macro_noise);
    base_color = mix(base_color, sand_highlight, micro_noise * macro_noise);

    let incident = normalize(in.world_position - camera.eye_position.xyz);
    let reflect_dir = reflect(incident, n_world);
    let spec = pow(max(dot(reflect_dir, light_dir), 0.0), 8.0) * micro_noise;
    base_color += vec3<f32>(0.15, 0.12, 0.10) * spec;

    let lin = diff * light.color + amb * light.ambient;
    var col = base_color * lin * ao;
    col = pow(col, vec3<f32>(0.4545));

    return vec4<f32>(apply_highlight(col, 1.0, in.instance_index, in.piece_slot), 1.0);
}

// Moss: a static (non-animated) procedural surface pattern, bump-mapped from
// its own height field. `moss_fbm`/`moss_pattern`/`evaluate_moss` are a
// direct port of a domain-warped fbm moss texture, adapted to sample the
// sticker's own local face UV (`sticker_face_uv`) instead of a mesh UV
// attribute this project's stickers don't have.
struct MossEval {
    color: vec3<f32>,
    height: f32,
};

fn moss_fbm(p_in: vec2<f32>) -> f32 {
    var p = p_in;
    var total = 0.0;
    var amplitude = 0.5;
    let rot = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);

    for (var i = 0; i < 6; i++) {
        total += amplitude * value_noise2(p);
        p = rot * p * 2.02;
        amplitude *= 0.5;
    }
    return total;
}

// Domain warping for organic moss clustering: two nested layers of fbm
// distort the sample point before a final fbm reads the pattern there.
fn moss_pattern(p: vec2<f32>) -> f32 {
    let q = vec2<f32>(moss_fbm(p), moss_fbm(p + vec2<f32>(5.2, 1.3)));
    let r = vec2<f32>(
        moss_fbm(p + 4.0 * q + vec2<f32>(1.7, 9.2)),
        moss_fbm(p + 4.0 * q + vec2<f32>(8.3, 2.8))
    );
    return moss_fbm(p + 4.0 * r);
}

fn evaluate_moss(uv: vec2<f32>, seed_offset: vec2<f32>) -> MossEval {
    let scale = 6.0;
    let st = uv * scale + seed_offset;

    let base_noise = moss_pattern(st);
    let micro_grit = value_noise2(st * 18.0);

    let height = mix(base_noise, micro_grit, 0.25);

    let deep_soil = vec3<f32>(0.08, 0.07, 0.03);
    let dark_moss = vec3<f32>(0.12, 0.28, 0.04);
    let bright_moss = vec3<f32>(0.38, 0.62, 0.08);
    let dry_moss = vec3<f32>(0.55, 0.58, 0.15);

    var col = mix(deep_soil, dark_moss, smoothstep(0.15, 0.40, height));
    col = mix(col, bright_moss, smoothstep(0.40, 0.70, height));
    col = mix(col, dry_moss, smoothstep(0.72, 0.90, height));

    return MossEval(col, height);
}

// Bump-mapped normal in the sticker's own local (pre-rotation) frame, from
// finite-differencing `evaluate_moss`'s height across the facet's UV. Mirrors
// `water_face_normal`'s recipe (including reuse of `water_edge_mask`, pure
// local-mesh geometry despite the name) rather than the reference's arbitrary
// TBN-from-world-normal construction, so it stays correct under the
// sticker's own warped 4D-projected frame.
fn moss_face_normal(uv: vec2<f32>, seed_offset: vec2<f32>, face_normal: vec3<f32>, box_extent: f32) -> vec3<f32> {
    let eps = vec2<f32>(0.005, 0.0);
    let mask = water_edge_mask(uv, box_extent);
    let h0 = evaluate_moss(uv, seed_offset).height * mask;
    let hx = evaluate_moss(uv + eps.xy, seed_offset).height * water_edge_mask(uv + eps.xy, box_extent) - h0;
    let hy = evaluate_moss(uv + eps.yx, seed_offset).height * water_edge_mask(uv + eps.yx, box_extent) - h0;
    let abs_n = abs(face_normal);
    var bump_n: vec3<f32>;
    if (abs_n.x > 0.5) {
        bump_n = vec3<f32>(sign(face_normal.x), -hy / eps.x, -hx / eps.x);
    } else if (abs_n.y > 0.5) {
        bump_n = vec3<f32>(-hx / eps.x, sign(face_normal.y), -hy / eps.x);
    } else {
        bump_n = vec3<f32>(-hx / eps.x, -hy / eps.x, sign(face_normal.z));
    }
    return normalize(bump_n);
}

fn moss_color(
    instance_index: u32,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    local_position: vec3<f32>,
) -> vec3<f32> {
    let local_face_normal = sticker_local_face_normal(local_position);
    let normalized_local = local_position / STICKER_HALF_EXTENT;
    let local_uv = sticker_face_uv(normalized_local, local_face_normal);

    // Offsets which patch of the (otherwise infinite) procedural field this
    // sticker samples, so neighboring Moss-kind stickers don't repeat the
    // same pattern - the same idea as Water's/Sand's per-instance seeding.
    let seed_offset = vec2<f32>(
        hash11(f32(instance_index) * 1.7) * 100.0,
        hash11(f32(instance_index) * 3.1 + 50.0) * 100.0,
    );

    let eval = evaluate_moss(local_uv, seed_offset);

    // Map the local bump-mapped normal into world space through the
    // sticker's own projected frame - see `water_color`'s identical
    // technique and its comment on the degenerate-frame fallback.
    let anchor = compute_sticker_anchor(instance_index, STICKER_HALF_EXTENT * transform.sticker_scale);
    let frame_volume = dot(anchor.edge_x, cross(anchor.edge_y, anchor.edge_z));
    let edge_volume = length(anchor.edge_x) * length(anchor.edge_y) * length(anchor.edge_z);

    var perturbed_world_normal = normalize(world_normal);
    if (anchor.visible && abs(frame_volume) >= edge_volume * STICKER_MIN_FRAME_VOLUME) {
        let to_world = mat3x3<f32>(anchor.edge_x, anchor.edge_y, anchor.edge_z);
        let local_normal = moss_face_normal(local_uv, seed_offset, local_face_normal, 1.0);
        perturbed_world_normal = normalize(to_world * local_normal);
    }

    let normal = perturbed_world_normal;
    let light_dir = normalize(-light.direction);
    let view_dir = normalize(-world_position);

    let ambient = light.ambient * eval.color;
    let diffuse_strength = max(dot(normal, light_dir), 0.0);
    let diffuse = diffuse_strength * light.color * eval.color;

    let half_dir = normalize(light_dir + view_dir);
    let specular_strength = pow(max(dot(normal, half_dir), 0.0), 16.0);
    let specular = specular_strength * light.color * 0.2;

    var final_color = ambient + diffuse + specular;
    // Ambient-occlusion-style darkening inside deep crevasses.
    final_color *= smoothstep(0.0, 0.5, eval.height * 0.8 + 0.2);

    return final_color;
}

// Dark: a raymarched window into one shared toxic-void "portal world" -
// unlike every other material in this file, it has no per-instance
// uniqueness (no instance_index, no local UV): the raymarch already varies
// continuously per-fragment from world_position alone, and every
// Dark-kind facet samples the *same* continuous world-space scene, so
// different facets glimpse different slices of one shared void rather than
// each rendering an independent copy - a deliberate "shared portal" look,
// not a bug. Also unlike every other material here, it ignores the `light`
// uniform entirely: the source shader is fully self-illuminated (its only
// light source is the lightning flash itself), so folding in a directional
// light would fight the portal's own lighting rather than complement it.
// Ported near-verbatim from a standalone Shadertoy-style source; only the
// noise primitives and the ray entry point were adapted to this project's
// conventions. The reddish flash/highlight tones are intentional contrast
// against the shader's own dark purple/violet void and cloud base tones -
// kept verbatim, not retinted.

fn dark_fbm(p_in: vec3<f32>) -> f32 {
    var p = p_in;
    var f = 0.0;
    f += 0.5000 * value_noise3(p); p *= 2.02;
    f += 0.2500 * value_noise3(p); p *= 2.03;
    f += 0.1250 * value_noise3(p); p *= 2.01;
    f += 0.0625 * value_noise3(p);
    return f;
}

fn dark_smooth_lightning(cloud_p: vec3<f32>, dist_to_cloud: f32) -> f32 {
    // Widened from the source's smoothstep(80.0, 10.0, ...) so flashes stay
    // visible/fading in from much farther away.
    let dist_fade = smoothstep(280.0, 10.0, dist_to_cloud);
    if (dist_fade <= 0.0) {
        return 0.0;
    }
    var total_glow = 0.0;
    for (var i = 0; i < 2; i++) {
        let fi = f32(i);
        let time_scale = transform.elapsed_seconds * 1.2 + fi * 15.3;
        let strike_id = floor(time_scale);
        let pulse = fract(time_scale);
        let trigger = hash11(strike_id + fi * 37.81);
        if (trigger > 0.70) {
            let flash = pow(1.0 - pulse, 3.5) * hash11(strike_id * 12.3) * 5.0;
            let noise_pos = cloud_p * 0.25 + vec3<f32>(strike_id * 2.5, 0.0, fi * 4.2);
            let patch_pattern = dark_fbm(noise_pos);
            let patch_mask = smoothstep(0.48, 0.72, patch_pattern);
            total_glow += flash * patch_mask;
        }
    }
    return total_glow * dist_fade;
}

fn dark_terrain(p: vec3<f32>) -> f32 {
    let height = dark_fbm(p * 0.1) * 6.0 - 3.0;
    let floor_y = -1.5;
    return p.y - (floor_y + height);
}

fn dark_render_world(ro: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    var t = 0.1;
    let tmax = 50.0;
    var p = vec3<f32>(0.0);
    var hit = false;
    for (var i = 0; i < 90; i++) {
        p = ro + rd * t;
        let d = dark_terrain(p);
        if (d < 0.01) {
            hit = true;
            break;
        }
        t += d * 0.5;
        if (t > tmax) {
            break;
        }
    }
    var color = vec3<f32>(0.015, 0.008, 0.02);
    let cloud_height = 15.0;
    let dist_to_cloud = (cloud_height - ro.y) / rd.y;
    var cloud_density = 0.0;
    var lightning_flash = 0.0;
    if (rd.y > 0.0 && (!hit || t > dist_to_cloud)) {
        let cloud_plane_p = ro + rd * dist_to_cloud;
        let cloud_p = cloud_plane_p * 0.08
            + vec3<f32>(transform.elapsed_seconds * 0.1, 0.0, transform.elapsed_seconds * 0.05);
        cloud_density = dark_fbm(cloud_p);
        let cloud_color = vec3<f32>(0.03, 0.02, 0.04) * cloud_density;
        lightning_flash = dark_smooth_lightning(cloud_p, dist_to_cloud);
        let lightning_color = vec3<f32>(2.5, 0.1, 0.04) * lightning_flash
            * smoothstep(0.25, 0.65, cloud_density);
        color += cloud_color + lightning_color;
    }
    if (hit) {
        let eps = vec2<f32>(0.02, 0.0);
        let norm = normalize(vec3<f32>(
            dark_terrain(p + eps.xyy) - dark_terrain(p - eps.xyy),
            dark_terrain(p + eps.yxy) - dark_terrain(p - eps.yxy),
            dark_terrain(p + eps.yyx) - dark_terrain(p - eps.yyx),
        ));
        let ground_color = vec3<f32>(0.03, 0.025, 0.03) * (dark_fbm(p * 0.4) * 0.6 + 0.4);
        let ambient = 0.15;
        let flash_diff = max(0.0, dot(norm, normalize(vec3<f32>(0.1, 1.0, 0.1))));
        let flash_light = vec3<f32>(1.2, 0.15, 0.1) * lightning_flash * flash_diff * 0.5;
        let scene_col = ground_color * (ambient + flash_light);
        let fog = 1.0 - exp(-t * 0.04);
        color = mix(scene_col, color, fog);
    }
    return color;
}

// The source's fixed `worldPos + vec3(0,-1,0)` assumed a literal world
// "down"; this puzzle has no fixed "up/down" once 4D-rotated. Entering
// along -normalize(world_normal) instead keeps the portal always opening
// "into" the surface regardless of which of the 8 facet orientations is
// showing.
const DARK_ENTRY_DEPTH: f32 = 1.0;

fn dark_color(world_position: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    let portal_ro = world_position - normalize(world_normal) * DARK_ENTRY_DEPTH;
    let portal_rd = normalize(world_position - camera.eye_position.xyz);
    var final_color = dark_render_world(portal_ro, portal_rd);
    final_color = pow(final_color, vec3<f32>(0.4545));
    return final_color;
}

// Lightning: dark metallic cube surface carrying continuous 3D domain-warped
// arcs plus separate random per-face flash strikes. Each of a face's 27
// Lightning stickers gets its own per-instance seed folded into
// both the arcs and the flashes, so neighboring stickers read as independent
// rather than a single synchronized field repeated 27 times - mirroring
// `water_wave_height`'s `instance_seed` technique in the Water material above.

fn lightning_rotate2d(theta: f32) -> mat2x2<f32> {
    let c = cos(theta);
    let s = sin(theta);
    return mat2x2<f32>(c, s, -s, c);
}

// 2D value noise in 0..1, built on the already-imported `hash21` rather than
// a second near-identical hash.
fn lightning_noise2d(p: vec2<f32>) -> f32 {
    let ip = floor(p);
    let fp = fract(p);
    let a = hash21(ip);
    let b = hash21(ip + vec2<f32>(1.0, 0.0));
    let c = hash21(ip + vec2<f32>(0.0, 1.0));
    let d = hash21(ip + vec2<f32>(1.0, 1.0));
    let t = smoothstep(vec2<f32>(0.0), vec2<f32>(1.0), fp);
    return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
}

fn lightning_fbm2d(p_in: vec2<f32>, octave_count: i32) -> f32 {
    var p = p_in;
    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    let rot = lightning_rotate2d(0.45);
    for (var i: i32 = 0; i < 10; i++) {
        if (i >= octave_count) { break; }
        value += amplitude * lightning_noise2d(p);
        p = rot * p;
        p *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

// 3D domain-warp fbm, ported from the source's `fbm3D`. Built on the
// already-imported `value_noise3` (algorithmically identical to the source's
// own `noise3D`: trilinear value noise over the 8 hashed corners of the unit
// cell) rather than a duplicate.
fn lightning_fbm3d(p_in: vec3<f32>, octaves: i32) -> f32 {
    var p = p_in;
    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    let rot = lightning_rotate2d(0.45);
    for (var i: i32 = 0; i < 6; i++) {
        if (i >= octaves) { break; }
        value += amplitude * value_noise3(p);
        let xy = rot * p.xy;
        let yz = rot * p.yz;
        p = vec3<f32>(xy.x, yz.x, yz.y);
        p *= 2.02;
        amplitude *= 0.5;
    }
    return value;
}

// Continuous, wrapping-glow lightning arcs over `p3d` (the sticker's own
// normalized -1..1 local position). `instance_index` seeds an offset folded
// into every per-arc `seed` below, so each of the 27 Lightning stickers on a
// face shows its own independent-looking arcs instead of literally the same
// field, mirroring `water_wave_height`'s `instance_seed`.
fn lightning_surface_arcs(p3d: vec3<f32>, time_val: f32, instance_index: u32) -> vec3<f32> {
    var col_acc = vec3<f32>(0.0);

    let blue = vec3<f32>(0.2, 0.45, 1.0);
    let purple = vec3<f32>(0.7, 0.2, 0.95);
    let yellow = vec3<f32>(1.0, 0.85, 0.3);

    let slow_time = time_val * 0.45;
    let instance_seed = hash11(f32(instance_index) * 0.6180339887) * 100.0;

    for (var i: i32 = 0; i < 3; i++) {
        let seed = f32(i) * 14.3 + instance_seed;

        let warp_offset = vec3<f32>(
            lightning_fbm3d(p3d * 1.8 + vec3<f32>(slow_time, seed, 0.0), 4),
            lightning_fbm3d(p3d * 1.8 + vec3<f32>(0.0, slow_time, seed), 4),
            lightning_fbm3d(p3d * 1.8 + vec3<f32>(seed, 0.0, slow_time), 4)
        );

        let warped_p = p3d + (warp_offset - vec3<f32>(0.5)) * 0.8;

        let val1 = abs(lightning_fbm3d(warped_p * 2.5 + vec3<f32>(seed), 5) - 0.5) * 2.0;
        let val2 = abs(lightning_fbm3d(warped_p * 2.5 + vec3<f32>(seed + 40.0), 5) - 0.5) * 2.0;

        let bolt_dist = length(vec2<f32>(val1, val2));

        let color_select = fract(lightning_fbm3d(p3d * 1.2 + vec3<f32>(seed), 3) + seed * 0.1);
        var base_col = mix(blue, purple, smoothstep(0.0, 0.55, color_select));
        base_col = mix(base_col, yellow, smoothstep(0.68, 1.0, color_select));

        let core = 0.003 / (bolt_dist + 0.0015);
        let glow = 0.006 / (bolt_dist + 0.025);

        col_acc += base_col * (pow(core, 1.3) + glow);
    }

    return col_acc;
}

// Which local cube face `face_normal` (from `sticker_local_face_normal`)
// points along, signed - matching the source shader's `faceID` numbering,
// used only to seed `lightning_face_flashes` per face.
fn lightning_face_id(face_normal: vec3<f32>) -> f32 {
    let abs_n = abs(face_normal);
    if (abs_n.x > 0.5) {
        return 1.0 * sign(face_normal.x);
    }
    if (abs_n.y > 0.5) {
        return 2.0 * sign(face_normal.y);
    }
    return 3.0 * sign(face_normal.z);
}

// Sparse, sharp directional flash strikes, ported from the source's
// `calculateReferenceFlashes`. `instance_index` seeds an offset folded into
// each strike's `seed`, decorrelating which stickers flash on a given strike
// window from one another - a different salt than `lightning_surface_arcs`'s
// `instance_seed` so the two effects don't always spike on the same stickers.
// Each sticker draws its own flash-window period from this range (not one
// shared clock), so different stickers' windows land at different cadences
// as well as different phases.
const LIGHTNING_FLASH_MIN_PERIOD: f32 = 0.5;
const LIGHTNING_FLASH_MAX_PERIOD: f32 = 1.5;
// A fired strike's own visible duration is drawn from this range per strike,
// kept comfortably below `LIGHTNING_FLASH_MIN_PERIOD` so a strike is always
// off again before its own next window begins. Hard on/off, no fade.
const LIGHTNING_FLASH_MIN_DURATION: f32 = 0.2;
const LIGHTNING_FLASH_MAX_DURATION: f32 = 0.4;

fn lightning_face_flashes(face_uv: vec2<f32>, face_id: f32, time_val: f32, instance_index: u32) -> vec3<f32> {
    var flash_col = vec3<f32>(0.0);

    let blue = vec3<f32>(0.2, 0.5, 1.0);
    let purple = vec3<f32>(0.7, 0.2, 1.0);
    let yellow = vec3<f32>(1.0, 0.85, 0.3);

    let instance_seed = hash11(f32(instance_index) * 0.3141592653) * 100.0;

    // Per-sticker period and phase offset so different stickers' flash
    // windows don't share a cadence or start/end at the same wall-clock
    // instant.
    let period = mix(LIGHTNING_FLASH_MIN_PERIOD, LIGHTNING_FLASH_MAX_PERIOD, hash11(instance_seed * 3.71 + 4.0));
    let phase_offset = hash11(instance_seed * 7.77 + 11.0) * period;
    let shifted_time = time_val + phase_offset;
    let window_index = floor(shifted_time / period);
    let time_in_window = shifted_time - window_index * period;

    for (var i: i32 = 0; i < 2; i++) {
        let seed = face_id * 19.3 + f32(i) * 11.7 + window_index * 7.1 + instance_seed;

        if (hash11(seed) > 0.90) {
            let duration = mix(LIGHTNING_FLASH_MIN_DURATION, LIGHTNING_FLASH_MAX_DURATION, hash11(seed + 5.0));
            let max_start = max(period - duration, 0.0);
            let start = hash11(seed + 9.0) * max_start;
            let local_t = time_in_window - start;

            if (local_t >= 0.0 && local_t <= duration) {
                let angle = hash11(seed + 1.0) * 6.28318530718;
                let rotated_uv = lightning_rotate2d(angle) * face_uv;

                var warped_uv = rotated_uv;
                warped_uv += vec2<f32>(2.0 * lightning_fbm2d(warped_uv + vec2<f32>(0.8 * (time_val + seed)), 8) - 1.0);

                let dist = abs(warped_uv.x);

                let col_pick = hash11(seed + 3.0);
                var c = mix(blue, purple, smoothstep(0.0, 0.5, col_pick));
                c = mix(c, yellow, smoothstep(0.65, 1.0, col_pick));

                let intensity = mix(0.01, 0.05, hash11(window_index + seed)) / dist;
                flash_col += c * pow(intensity, 1.1);
            }
        }
    }

    return flash_col;
}

fn lightning_color(
    instance_index: u32,
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    local_position: vec3<f32>,
) -> vec3<f32> {
    let normal = normalize(world_normal);
    let normalized_local = local_position / STICKER_HALF_EXTENT;

    let face_normal = sticker_local_face_normal(local_position);
    let face_uv = sticker_face_uv(normalized_local, face_normal);
    let face_id = lightning_face_id(face_normal);

    let light_dir = normalize(-light.direction);
    let diff = max(dot(normal, light_dir), 0.0);
    let base_cube_col = vec3<f32>(0.02, 0.025, 0.04) + vec3<f32>(0.03, 0.035, 0.05) * diff;

    let abs_p = abs(normalized_local);
    let edge_dist = max(max(abs_p.x, abs_p.y), abs_p.z);
    let frame_glow = 0.0012 / (abs(edge_dist - 0.75) + 0.0015);
    let frame_col = vec3<f32>(0.6, 0.2, 0.9) * pow(frame_glow, 1.2) * 0.25;

    let continuous_arcs = lightning_surface_arcs(normalized_local, transform.elapsed_seconds, instance_index);
    let reference_flashes = lightning_face_flashes(face_uv, face_id, transform.elapsed_seconds, instance_index);

    var lightning_col = continuous_arcs + reference_flashes;
    // Soft tone-compression curve, ported unchanged from the source, to
    // avoid harsh white clipping where arcs and flashes overlap.
    lightning_col = lightning_col / (1.0 + lightning_col * 0.2);

    return base_cube_col + frame_col + lightning_col;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var final_color: vec3<f32>;

    switch (in.kind) {
        // Ice is drawn by `fs_ice` in its own per-depth-layer passes, so this
        // one skips it rather than shading it twice.
        case 0u: {
            discard;
            return vec4<f32>(0.0);
        }
        case 1u: {
            final_color = moss_color(in.instance_index, in.world_position, in.world_normal, in.local_position);
        }
        case 2u: {
            final_color = lightning_color(in.instance_index, in.world_position, in.world_normal, in.local_position);
        }
        // Fire is drawn by `fs_fire` in its own blended pass, so this one
        // skips it rather than shading it twice.
        case 3u: {
            discard;
            return vec4<f32>(0.0);
        }
        // Dirt is drawn by `fs_dirt` in its own blended pass, so this one
        // skips it rather than shading it twice.
        case 4u: {
            discard;
            return vec4<f32>(0.0);
        }
        // Light is drawn by `fs_light` in its own blended pass, so this one
        // skips it rather than shading it twice.
        case 5u: {
            discard;
            return vec4<f32>(0.0);
        }
        case 6u: {
            final_color = water_color(in.instance_index, in.world_position, in.world_normal, in.local_position);
        }
        case 7u: {
            final_color = dark_color(in.world_position, in.world_normal);
        }
        default: {
            final_color = vec3<f32>(0.5, 0.5, 0.5);
        }
    }

    return vec4<f32>(apply_highlight(final_color, 1.0, in.instance_index, in.piece_slot), 1.0);
}

// Fire: a small cube of flame inset in each sticker cube, raymarched in the
// sticker's own local frame.

// Half-extent of a sticker's mesh cube before `transform.sticker_scale`.
// Must match `BASE_STICKER_SIZE` in math.rs, which renderer.rs premultiplies
// into the cube vertices. Shared by Fire and Water, both of which map
// between a sticker's local mesh frame and its world-space projected frame
// via `compute_sticker_anchor`.
const STICKER_HALF_EXTENT: f32 = 0.33333334;

// The flame cube's half-extent in local units, where 1.0 is the sticker
// mesh cube's own half-extent. `FIRE_BASE_EXTENT + FIRE_SURFACE_DISPLACEMENT`
// sits close to, and can run past, that 1.0: the displaced surface is
// pushed nearly to the facet's own edge rather than fading out well short of
// it. This never bleeds past the facet on screen regardless, since the only
// fragments shaded are the ones the sticker's own front face rasterizes to.
const FIRE_BASE_EXTENT: f32 = 0.9;
const FIRE_SURFACE_DISPLACEMENT: f32 = 0.22;

// Samples taken across the flame cube. The dominant cost of the whole theme:
// each one evaluates four octaves of trilinear value noise, and a scrambled
// cube can show 27 fire facets at once.
const FIRE_STEPS: i32 = 16;

// Optical depth accumulated per flame-extent travelled at unit density.
const FIRE_ABSORPTION: f32 = 3.75;

// Scales local positions into the noise domain. A facet covers few enough
// pixels that features much finer than this stop resolving into anything
// legible and only cost octaves to produce.
const FIRE_NOISE_SCALE: f32 = 0.8;

// How near to degenerate a sticker's projected frame may get before an
// effect built on `compute_sticker_anchor` gives up on it, as a fraction of
// the volume three edges of the same lengths would span if they were
// perpendicular. A 4D rotation can squash a facet flat, collapsing the frame
// toward a plane and sending its inverse - and with it Fire's flame shape,
// or Water's perturbed-normal mapping - to infinity; such a facet is
// edge-on and covers almost no pixels anyway. Shared by Fire and Water.
const STICKER_MIN_FRAME_VOLUME: f32 = 0.05;

// Multiplier on `elapsed_seconds` for the plasma's churn, and the wrap
// applied first. Convection translates the noise domain without bound, so
// the input has to be kept small enough that the hash still resolves detail;
// the churn hides the periodic reset.
const FIRE_CHURN_SPEED: f32 = 1.5;
const FIRE_TIME_WRAP: f32 = 60.0;

// Rotates and rescales the sample position between octaves, so the octaves
// stack at unrelated orientations instead of reinforcing each other's axis
// alignment into a visible grid.
const m3 = mat3x3<f32>(
    vec3<f32>(0.00, 0.80, 0.60),
    vec3<f32>(-0.80, 0.36, -0.48),
    vec3<f32>(-0.60, -0.48, 0.64),
);

// Ridged noise: `1 - |2n - 1|` creases the noise along its own mid-level, so
// the octaves stack into filaments and sheets rather than the soft blobs
// plain FBM gives. The two-octave form warps the domain the three-octave
// form is then sampled in.
fn ridged_warp(p_in: vec3<f32>) -> f32 {
    var p = p_in;
    var f = 0.0;
    var amplitude = 0.5;
    for (var i = 0; i < 2; i++) {
        f += amplitude * (1.0 - abs(value_noise3(p) * 2.0 - 1.0));
        p = m3 * p * 2.05;
        amplitude *= 0.5;
    }
    return f;
}

fn ridged_detail(p_in: vec3<f32>) -> f32 {
    var p = p_in;
    var f = 0.0;
    var amplitude = 0.5;
    for (var i = 0; i < 2; i++) {
        f += amplitude * (1.0 - abs(value_noise3(p) * 2.0 - 1.0));
        p = m3 * p * 2.05;
        amplitude *= 0.5;
    }
    return f;
}

// The plasma field at one point of the flame cube. The volume turns slowly about
// two axes while the noise domain drifts along -y, which reads as convection
// rising through it. Both are expressed in the sticker's own local frame, so
// "up" turns with the puzzle rather than pointing at a world direction a 4D
// puzzle has no fixed version of.
fn solar_plasma(p_in: vec3<f32>, time: f32) -> f32 {
    var p = p_in;
    let c = cos(time * 0.15);
    let s = sin(time * 0.15);

    let rotated_x = p.x * c - p.z * s;
    let rotated_z = p.x * s + p.z * c;
    p.x = rotated_x;
    p.z = rotated_z;

    let tilted_x = p.x * c - p.y * s;
    let tilted_y = p.x * s + p.y * c;
    p.x = tilted_x;
    p.y = tilted_y;

    let convected = p - vec3<f32>(0.0, time * 0.8, 0.0);
    let warp = ridged_warp(convected * 1.5);

    let detailed = p + vec3<f32>(warp) * 0.4 - vec3<f32>(0.0, time * 1.2, 0.0);
    return ridged_detail(detailed * 1.8);
}

// Exposure, folded into the palette rather than applied to the composite -
// which would dim every other element by the same factor.
const FIRE_EXPOSURE: f32 = 0.18;

// Blackbody-ish ramp from a dim red shell to a white fusion core. The stops
// run far above 1.0 on purpose: that range is what the Elemental composite's
// tonemap turns into visible interior shape instead of one flat disc.
fn sun_palette(t: f32) -> vec3<f32> {
    let dark_red = vec3<f32>(1.2, 0.05, 0.0);
    let bright_orange = vec3<f32>(12.0, 3.5, 0.1);
    let yellow_core = vec3<f32>(45.0, 25.0, 5.0);
    let white_fusion = vec3<f32>(90.0, 85.0, 95.0);

    var color = mix(dark_red, bright_orange, smoothstep(0.0, 0.35, t));
    color = mix(color, yellow_core, smoothstep(0.35, 0.65, t));
    color = mix(color, white_fusion, smoothstep(0.65, 1.0, t));
    return color * FIRE_EXPOSURE;
}

// Entry and exit distances along a normalized `rd` from `ro` for the
// axis-aligned cube of `half_extent` centered on the origin, clamped so the
// span starts no earlier than the ray does. Returns a negative entry when
// the ray misses it, or leaves it entirely behind.
fn ray_box(ro: vec3<f32>, rd: vec3<f32>, half_extent: f32) -> vec2<f32> {
    let inv_rd = 1.0 / rd;
    let t0 = (vec3<f32>(-half_extent) - ro) * inv_rd;
    let t1 = (vec3<f32>(half_extent) - ro) * inv_rd;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let near = max(max(tmin.x, tmin.y), tmin.z);
    let far = min(min(tmax.x, tmax.y), tmax.z);
    if (far < 0.0 || near > far) {
        return vec2<f32>(-1.0);
    }
    return vec2<f32>(max(near, 0.0), far);
}

// Fire's own entry point, drawn per sticker in back-to-front order over a
// pipeline with premultiplied blending and no depth write. Emission is
// accumulated against a transmittance along a ray through the flame cube, so
// the result is a premultiplied color and the coverage it was multiplied by.
//
// Every fragment of the cube on one pixel yields the same ray, so shading
// both its faces would march the flame twice and blend the result over
// itself; back faces are discarded here by their current normal rather than
// by `cull_mode` (which stays `None` everywhere), since `world_normal` is
// derived fresh from the instance's own basis and so never goes stale
// mid-move the way the index winding `cull_mode` relies on can.
@fragment
fn fs_fire(in: VertexOutput) -> @location(0) vec4<f32> {
    let view_dir = normalize(camera.eye_position.xyz - in.world_position);
    if (dot(normalize(in.world_normal), view_dir) <= 0.0) {
        discard;
        return vec4<f32>(0.0);
    }

    let anchor = compute_sticker_anchor(in.instance_index, STICKER_HALF_EXTENT * transform.sticker_scale);

    // The sticker's own frame. Its three edges are unequal in length and no
    // longer mutually perpendicular once the 4D perspective divide has
    // warped the cube, so a cube marched in the local space `to_local` maps
    // into comes out stretched in world space exactly as its facet is.
    // Mapping through the edges is a linearization of a projection that
    // isn't linear, but across one sticker's own span the error is far below
    // what a volume of noise resolves.
    let to_world = mat3x3<f32>(anchor.edge_x, anchor.edge_y, anchor.edge_z);

    let frame_volume = dot(anchor.edge_x, cross(anchor.edge_y, anchor.edge_z));
    let edge_volume = length(anchor.edge_x) * length(anchor.edge_y) * length(anchor.edge_z);
    if (abs(frame_volume) < edge_volume * STICKER_MIN_FRAME_VOLUME) {
        discard;
        return vec4<f32>(0.0);
    }

    let to_local = inverse3(to_world);

    // The ray starts at the fragment, on the cube's own front face, rather
    // than at the eye: `to_local` scales by the inverse of a sticker's size,
    // so an eye-relative origin lands hundreds of flame extents out at small
    // sticker scales and the box test loses the flame entirely to
    // cancellation. Both vectors below are world-space differences of
    // comparable magnitude, and the flame sits within its extent of the
    // origin.
    let ray_origin = to_local * (in.world_position - anchor.world_center);
    let ray_direction = normalize(to_local * (in.world_position - camera.eye_position.xyz));

    let span = ray_box(ray_origin, ray_direction, FIRE_BASE_EXTENT + FIRE_SURFACE_DISPLACEMENT);
    if (span.x < 0.0) {
        discard;
        return vec4<f32>(0.0);
    }

    let time = transform.elapsed_seconds % FIRE_TIME_WRAP;
    // Seeded by `piece_slot` rather than instance index, so a piece keeps its
    // own flame pattern when a move relocates it to a different slot. The
    // hash bounds the offset instead of scaling the slot directly, keeping
    // every sticker's noise domain in the same well-resolved range.
    let seed = hash11(f32(in.piece_slot) * 0.073) * 4.0;
    let seed_offset = vec3<f32>(seed, seed * 1.3, seed * 0.7);

    let step_size = (span.y - span.x) / f32(FIRE_STEPS);
    // Densities below are expressed per flame extent, so the integration
    // length is too - which keeps them independent of the flame's own size.
    let step_radii = step_size / FIRE_BASE_EXTENT;
    // Offsetting each ray's first sample by a per-pixel fraction of a step
    // trades this step count's banding for noise. Hashed on the pixel alone,
    // so the pattern is fixed in screen space instead of crawling frame to
    // frame.
    let jitter = hash31(vec3<f32>(in.clip_position.xy, 0.0));
    var travelled = span.x + step_size * jitter;

    var accumulated = vec3<f32>(0.0);
    var transmittance = 1.0;

    for (var i = 0; i < FIRE_STEPS; i++) {
        if (transmittance < 0.01) {
            break;
        }

        let p = ray_origin + ray_direction * travelled;
        travelled += step_size;

        let center_extent = max(max(abs(p.x), abs(p.y)), abs(p.z));
        let plasma = solar_plasma(p * FIRE_NOISE_SCALE + seed_offset, time * FIRE_CHURN_SPEED);
        let surface_extent = FIRE_BASE_EXTENT + plasma * FIRE_SURFACE_DISPLACEMENT;
        if (center_extent > surface_extent) {
            continue;
        }

        // A dense core falling off exponentially, plus the plasma's own
        // filaments faded in over the outer half, all tapered to nothing at
        // the displaced surface so the flame cube has no hard edge.
        let normalized_extent = center_extent / FIRE_BASE_EXTENT;
        let core_density = exp(-normalized_extent * 3.5) * 14.0;
        let surface_density = plasma * 4.5;
        var density = core_density + surface_density * smoothstep(1.3, 0.4, normalized_extent);
        density *= smoothstep(surface_extent, surface_extent - 0.15, center_extent);
        if (density <= 0.01) {
            continue;
        }

        transmittance *= exp(-density * FIRE_ABSORPTION * step_radii);

        let temperature = (1.0 - normalized_extent) * 1.8 + plasma * 0.8;
        accumulated += sun_palette(temperature) * density * transmittance * step_radii;
    }

    let coverage = 1.0 - transmittance;
    return vec4<f32>(
        apply_highlight(accumulated, coverage, in.instance_index, in.piece_slot),
        coverage,
    );
}

// Light: a raymarched volumetric cloud filling each sticker's own local
// cube, shaded by density-based self-shadowing toward the scene's real
// directional light plus a fixed warm/ambient palette. Drawn the same way
// as Fire - its own blended, no-depth-write pass, sharing Fire's
// local-frame technique (`compute_sticker_anchor`/`inverse3`) rather than
// a flat model-matrix inverse, since a 4D rotation warps a sticker's
// projected cube into a non-orthonormal frame. Unlike Dark, which is fully
// self-illuminated and ignores `light` on purpose, Light is meant to read
// as lit, so it shades against the same directional light every other
// material does.
const LIGHT_CLOUD_HALF_EXTENT: f32 = 1.0;
const LIGHT_CLOUD_STEPS: u32 = 28u;
const LIGHT_CLOUD_SHADOW_STEPS: u32 = 5u;
const LIGHT_CLOUD_SHADOW_STEP_SIZE: f32 = 0.12;

// Soft fade to zero near the cube's own boundary, so the cloud has no hard
// edge at the sticker's face.
fn light_cloud_box_mask(pos: vec3<f32>) -> f32 {
    let box_dist = abs(pos) / LIGHT_CLOUD_HALF_EXTENT;
    let max_dist = max(box_dist.x, max(box_dist.y, box_dist.z));
    return smoothstep(1.0, 0.85, max_dist);
}

// 3-octave FBM for the cloud's primary density field.
fn light_cloud_fbm(p_in: vec3<f32>) -> f32 {
    var p = p_in;
    var density = 0.5 * value_noise3(p);
    p *= 2.02;
    density += 0.25 * value_noise3(p);
    p *= 2.03;
    density += 0.125 * value_noise3(p);
    return density;
}

// `seed_offset` (see `fs_light`) shifts which region of the shared,
// scrolling noise field a sticker samples, so same-face stickers - who'd
// otherwise all march the identical local `[-1,1]` box against the same
// `transform.elapsed_seconds` - read as independent clouds instead of one
// pattern repeated on every facet.
fn light_cloud_density(pos: vec3<f32>, seed_offset: vec3<f32>) -> f32 {
    let animated_pos = pos * 2.0 + seed_offset
        + vec3<f32>(transform.elapsed_seconds * 0.15, transform.elapsed_seconds * 0.08, 0.0);
    let d = light_cloud_fbm(animated_pos);
    return clamp((d - 0.25) * 2.0, 0.0, 1.0) * light_cloud_box_mask(pos);
}

// Single-octave density for the cheaper shadow sub-march.
fn light_cloud_fast_density(pos: vec3<f32>, seed_offset: vec3<f32>) -> f32 {
    let animated_pos = pos * 2.0 + seed_offset
        + vec3<f32>(transform.elapsed_seconds * 0.15, transform.elapsed_seconds * 0.08, 0.0);
    let d = value_noise3(animated_pos);
    return clamp((d - 0.3) * 1.8, 0.0, 1.0) * light_cloud_box_mask(pos);
}

// Light's own entry point, drawn per sticker in back-to-front order over a
// pipeline with premultiplied blending and no depth write - see `fs_fire`'s
// own doc comment for why back faces are discarded here rather than by
// `cull_mode`.
@fragment
fn fs_light(in: VertexOutput) -> @location(0) vec4<f32> {
    let view_dir = normalize(camera.eye_position.xyz - in.world_position);
    if (dot(normalize(in.world_normal), view_dir) <= 0.0) {
        discard;
        return vec4<f32>(0.0);
    }

    let anchor = compute_sticker_anchor(in.instance_index, STICKER_HALF_EXTENT * transform.sticker_scale);

    let to_world = mat3x3<f32>(anchor.edge_x, anchor.edge_y, anchor.edge_z);

    let frame_volume = dot(anchor.edge_x, cross(anchor.edge_y, anchor.edge_z));
    let edge_volume = length(anchor.edge_x) * length(anchor.edge_y) * length(anchor.edge_z);
    if (abs(frame_volume) < edge_volume * STICKER_MIN_FRAME_VOLUME) {
        discard;
        return vec4<f32>(0.0);
    }

    let to_local = inverse3(to_world);

    let ray_origin = to_local * (in.world_position - anchor.world_center);
    let ray_direction = normalize(to_local * (in.world_position - camera.eye_position.xyz));
    // `-light.direction`, matching every other material's "toward the
    // light" convention (see e.g. `moss_color`), rather than the raw
    // direction the light travels.
    let local_light_dir = normalize(to_local * (-light.direction));

    let span = ray_box(ray_origin, ray_direction, LIGHT_CLOUD_HALF_EXTENT);
    if (span.x < 0.0) {
        discard;
        return vec4<f32>(0.0);
    }

    let cos_theta = dot(ray_direction, local_light_dir);
    let phase = 0.5 + 0.5 * cos_theta * cos_theta;

    let step_size = (span.y - span.x) / f32(LIGHT_CLOUD_STEPS);
    // Screen-space-stable jitter (fixed pattern, not crawling frame to
    // frame) - see `fs_fire`'s own use of this.
    let jitter = hash31(vec3<f32>(in.clip_position.xy, 0.0));
    var travelled = span.x + step_size * jitter;

    // Seeded by `piece_slot` rather than instance index, so a piece keeps
    // its own cloud pattern when a move relocates it to a different slot -
    // see `fs_fire`'s identical technique.
    let seed = hash11(f32(in.piece_slot) * 0.073) * 4.0;
    let seed_offset = vec3<f32>(seed, seed * 1.3, seed * 0.7);

    var accumulated = vec3<f32>(0.0);
    var transmittance = 1.0;

    for (var i = 0u; i < LIGHT_CLOUD_STEPS; i++) {
        if (travelled >= span.y || transmittance < 0.02) {
            break;
        }

        let p = ray_origin + ray_direction * travelled;
        travelled += step_size;

        if (any(abs(p) > vec3<f32>(LIGHT_CLOUD_HALF_EXTENT))) {
            break;
        }

        let density = light_cloud_density(p, seed_offset);
        if (density <= 0.01) {
            continue;
        }

        var shadow_density = 0.0;
        for (var j = 0u; j < LIGHT_CLOUD_SHADOW_STEPS; j++) {
            let shadow_pos = p + local_light_dir * (f32(j) * LIGHT_CLOUD_SHADOW_STEP_SIZE);
            if (any(abs(shadow_pos) > vec3<f32>(LIGHT_CLOUD_HALF_EXTENT))) {
                break;
            }
            shadow_density += light_cloud_fast_density(shadow_pos, seed_offset);
        }

        let light_attenuation = exp(-shadow_density * 2.0);
        let light_color = vec3<f32>(1.0, 0.9, 0.7) * light_attenuation * phase * 3.0;
        let ambient_color = vec3<f32>(0.2, 0.3, 0.5) * (p.y * 0.5 + 0.5);
        let scatter = light_color + ambient_color;

        let absorption = density * step_size * 4.5;
        let step_transmittance = exp(-absorption);

        accumulated += transmittance * (1.0 - step_transmittance) * scatter;
        transmittance *= step_transmittance;
    }

    let coverage = 1.0 - transmittance;
    return vec4<f32>(
        apply_highlight(accumulated, coverage, in.instance_index, in.piece_slot),
        coverage,
    );
}

// Ice: a raymarched glass cube with real refraction and reflection, ported
// from a self-contained source scene - its own orbiting camera, its own
// floor + box SDF, reflection and refraction sampling a previously-rendered
// frame - so porting it means
// replacing pieces of it with what this app already has, rather than a
// straight per-fragment reshade like Water/Lightning:
//
// - Rasterizing the sticker's own cube mesh already gives an exact entry
//   point and normal (`in.world_position`/`in.world_normal`), so unlike the
//   source there's no need to sphere-trace to *find* the box surface first.
// - Past that entry point there is no floor, only the ice and then whatever
//   the real scene shows behind it: `ice_sample_background` replaces the
//   source's synthetic floor/sky raycast with a screen-space sample of
//   `ice_background_texture` ("iChannel1"), the snapshot `render()` copies
//   the scene into before each Ice depth layer's pass (see `render()`'s doc
//   comment). This is what makes Ice's refraction/reflection show the
//   actual rendered scene.
// - Finding where the refracted ray exits the ice again - which drives both
//   the internal color tint and the exit-ray direction sampled from
//   `iChannel1` - only needs an exact box intersection here, since the ice
//   fills the sticker's whole cube: `ray_box` (defined above, in Fire's
//   section) does that directly, in the same local frame
//   (`compute_sticker_anchor`/`to_world`/`to_local`) Fire already uses to
//   raymarch inside a facet, rather than the source's own sphere-traced SDF.
// - `iChannel0`'s triplanar-sampled gray noise bump map
//   (`ice_smooth_sample`/`ice_triplanar_sample`/`ice_triplanar_noise`) is
//   ported over unchanged - it doesn't touch scene geometry.
// - Dropped entirely: the source's own camera/orbit controls and its
//   standalone vignette+gamma tail, both whole-frame effects this app's own
//   camera and post-process/composite/bloom already cover.

// Hardcoded in the source (its own UI sliders collapsed to fixed values);
// kept as named, independently tunable constants here.
const ICE_ROUGHNESS: f32 = 1.0;
const ICE_REFRACTION_IDX: f32 = 1.9;
// Always 0.0 in the source (an unused slider) - every `smoothstep` in
// `ice_inner_color`'s ramp then reads as 0, so it always evaluates to the
// first branch, `ICE_WHITE`. Kept as a named tunable rather than collapsed,
// matching the source's own structure.
const ICE_COLOR: f32 = 0.0;

// Scales local position into the noise domain the triplanar bump map is
// sampled in, matching the source's `BUMP_MAP_UV_SCALE` folded into `ROUGHNESS`.
const ICE_BUMP_UV_SCALE: f32 = 0.2;
// Notional resolution `ice_smooth_sample` reconstructs bicubic-ish smoothing
// for, matching `ice_noise_64.png`'s actual size (see
// `src/bin/generate_ice_noise.rs`) and the source's own `T_RES`.
const ICE_NOISE_RESOLUTION: f32 = 64.0;
// How far past the ice's exit surface a reflected/refracted ray is stepped
// before projecting it to screen space to sample `iChannel1` - enough to
// clear the ice's own depth without meaningfully displacing the sample.
const ICE_BACKGROUND_SAMPLE_DISTANCE: f32 = 0.05;

// Reconstructs smooth (bicubic-ish) filtering from `ice_noise_texture`'s
// unfiltered texel reads, exactly as the source's `smoothSampling` does for
// its own noise texture - ported unchanged past the binding names.
fn ice_smooth_sample(uv: vec2<f32>) -> f32 {
    let x = fract(uv * ICE_NOISE_RESOLUTION + 0.5);
    let texel_corner = uv - x / ICE_NOISE_RESOLUTION;
    let t = (6.0 * x * x - 15.0 * x + 10.0) * x * x * x;
    return textureSampleLevel(ice_noise_texture, ice_noise_sampler, texel_corner + t / ICE_NOISE_RESOLUTION, 0.0).r;
}

fn ice_triplanar_sample(p: vec3<f32>, n: vec3<f32>) -> f32 {
    let total = abs(n.x) + abs(n.y) + abs(n.z);
    return (abs(n.x) * ice_smooth_sample(p.yz)
          + abs(n.y) * ice_smooth_sample(p.xz)
          + abs(n.z) * ice_smooth_sample(p.xy)) / total;
}

const ICE_BUMP_ROTATE: mat2x2<f32> = mat2x2<f32>(0.90, 0.44, -0.44, 0.90);

fn ice_triplanar_noise(p_in: vec3<f32>, n: vec3<f32>) -> f32 {
    var p = p_in;
    let f1 = ice_triplanar_sample(p * ICE_BUMP_UV_SCALE, n);

    p = vec3<f32>(ICE_BUMP_ROTATE * p.xy, p.z);
    p = vec3<f32>(p.x, ICE_BUMP_ROTATE * p.xz);
    p *= 2.1;
    let f2 = ice_triplanar_sample(p * ICE_BUMP_UV_SCALE, n);

    p = vec3<f32>(ICE_BUMP_ROTATE * p.yx, p.z);
    p = vec3<f32>(p.x, ICE_BUMP_ROTATE * p.yz);
    p *= 2.3;
    let f3 = ice_triplanar_sample(p * ICE_BUMP_UV_SCALE, n);

    return f1 + 0.5 * f2 + 0.25 * f3;
}

fn ice_normal_map(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let d = 0.005;
    let po = ice_triplanar_noise(p, n);
    let px = ice_triplanar_noise(p + vec3<f32>(d, 0.0, 0.0), n);
    let py = ice_triplanar_noise(p + vec3<f32>(0.0, d, 0.0), n);
    let pz = ice_triplanar_noise(p + vec3<f32>(0.0, 0.0, d), n);
    let gradient = vec3<f32>((px - po) / d, (py - po) / d, (pz - po) / d);

    // A perfectly (or near-) flat sample has no defined bump direction, and
    // `normalize` of a near-zero vector is NaN - which then poisons every
    // later pass that blends or blurs across this pixel (Fire's blended
    // pass, bloom), not just this one fragment. No bump reads as a flat,
    // unperturbed surface, a safe fallback for a case that should be rare
    // but isn't provably impossible.
    let gradient_length = length(gradient);
    if (gradient_length < 1e-6) {
        return vec3<f32>(0.0);
    }
    return gradient / gradient_length;
}

// The source's fixed ICE_INNER color ramp. Always evaluates to `c_white`
// with `ICE_COLOR` pinned at 0.0 (see its doc comment above).
fn ice_inner_color() -> vec3<f32> {
    let c_red = vec3<f32>(0.70, -0.5, -0.60);
    let c_green = vec3<f32>(-0.50, 0.0, -0.5);
    let c_blue = vec3<f32>(-0.50, -0.5, 0.30);
    let c_grey = vec3<f32>(-0.3);
    let c_white = vec3<f32>(1.0);

    var col = mix(c_white, c_grey, smoothstep(0.00, 0.20, ICE_COLOR));
    col = mix(col, c_blue, smoothstep(0.20, 0.40, ICE_COLOR));
    col = mix(col, c_green, smoothstep(0.40, 0.60, ICE_COLOR));
    col = mix(col, c_red, smoothstep(0.60, 0.80, ICE_COLOR));
    return col;
}

// Replaces the source's synthetic floor/sky raycast: steps a small distance
// along `dir_world` from `origin_world`, projects that point through the
// real camera to screen space, and samples the actual rendered scene
// (`ice_background_texture`, "iChannel1") there. This is what makes Ice's
// reflection/refraction show the app's real scene rather than a procedural
// backdrop. `ice_background_sampler` clamps to the texture edge, so a sample
// that lands outside the viewport (a grazing reflection near screen edges)
// degrades to the edge color instead of wrapping or reading garbage.
fn ice_sample_background(origin_world: vec3<f32>, dir_world: vec3<f32>) -> vec3<f32> {
    let sample_point = origin_world + dir_world * ICE_BACKGROUND_SAMPLE_DISTANCE;
    let clip = camera.view_proj * vec4<f32>(sample_point, 1.0);
    // `clip.w` is astronomically unlikely to land at exactly 0 for a point
    // this close to real scene geometry, but a 0/0 NaN here would corrupt
    // every later pass that blends or blurs across this pixel - guarding it
    // is cheap insurance against a case that isn't provably impossible.
    let safe_w = select(clip.w, 1e-4, abs(clip.w) < 1e-4);
    let ndc = clip.xyz / safe_w;
    let bounds_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    // `bounds_uv` spans the viewport `render()` actually drew into (`[0,1]`
    // across `Renderer::bounds`), not the whole `ice_background_texture` -
    // that texture is sized to the full window, most of which (menu bar,
    // any letterboxing) `render()` never touches. Without this remap, a
    // fragment reads scaled-and-offset into the wrong part of the texture -
    // visibly, content misaligned with (and bleeding in from) the
    // never-rendered area outside the viewport.
    let uv = ice_background_transform.offset + bounds_uv * ice_background_transform.scale;
    return textureSampleLevel(ice_background_texture, ice_background_sampler, uv, 0.0).rgb;
}

// Ice's own entry point, drawn one instance and one depth layer at a time
// (see `render()`'s doc comment) so each layer's `ice_background_texture`
// snapshot already contains every farther Ice sticker's own result.
//
// Uses the same local frame Fire's `fs_fire` raymarches in
// (`compute_sticker_anchor`/`to_world`/`to_local`), since that frame already
// corrects for the non-uniform stretch a 4D perspective divide can leave a
// facet with - a plain axis-aligned box in that frame is exactly the
// sticker's own cube, so `ray_box` finds the ice's exit point exactly, with
// no sphere-tracing needed.
@fragment
fn fs_ice(in: VertexOutput) -> @location(0) vec4<f32> {
    let anchor = compute_sticker_anchor(in.instance_index, STICKER_HALF_EXTENT * transform.sticker_scale);
    let to_world = mat3x3<f32>(anchor.edge_x, anchor.edge_y, anchor.edge_z);

    let frame_volume = dot(anchor.edge_x, cross(anchor.edge_y, anchor.edge_z));
    let edge_volume = length(anchor.edge_x) * length(anchor.edge_y) * length(anchor.edge_z);
    if (abs(frame_volume) < edge_volume * STICKER_MIN_FRAME_VOLUME) {
        // Near-degenerate edge-on facet, covering almost no pixels anyway -
        // discard rather than divide by the ill-conditioned `to_local`
        // below, same as Fire/Water do for the same case.
        discard;
        return vec4<f32>(0.0);
    }

    let to_local = inverse3(to_world);
    let entry_local = to_local * (in.world_position - anchor.world_center);

    let world_normal = normalize(in.world_normal);
    // Small on purpose: `ice_normal_map` takes a *numerical derivative* of
    // `ice_triplanar_noise` at this offset position, and that noise's own
    // octaves multiply the position by up to ~4.8x internally. A large
    // offset pushes the finite-difference evaluation far enough out that
    // float32 precision can't resolve the `d = 0.005` step `ice_normal_map`
    // diffs by, so it starts returning near-zero gradients and NaNs (see
    // `ice_normal_map`'s guard). This magnitude still shifts each of a
    // face's 27 stickers by roughly a full noise cell (`ICE_BUMP_UV_SCALE` =
    // 0.2, so `5.0 * 0.2 = 1.0`), decorrelating their patterns while keeping
    // the derivative well-conditioned.
    let instance_seed = hash11(f32(in.instance_index) * 0.4127) * 5.0;
    let bump_normal_delta = ice_normal_map(
        entry_local * ICE_ROUGHNESS + vec3<f32>(instance_seed),
        world_normal,
    ) * ICE_ROUGHNESS * 0.1;
    let bump_normal = normalize(world_normal + bump_normal_delta);

    let incident = normalize(in.world_position - camera.eye_position.xyz);
    let refract_dir = refract(incident, bump_normal, 1.0 / ICE_REFRACTION_IDX);
    let reflect_dir = reflect(incident, bump_normal);
    let reflect_alpha = 0.5 * (1.0 - abs(dot(incident, bump_normal)));

    // `1.0 / ICE_REFRACTION_IDX` is comfortably below 1.0, so total internal
    // reflection - `refract` returning the zero vector - is not physically
    // reachable entering the medium the way it is on exit below. Guarded
    // anyway: normalizing a zero vector is NaN, and unlike `exit_dir`'s
    // already-handled TIR case, nothing downstream expects this one to ever
    // be zero. Falls back to the (always well-defined) reflection direction,
    // i.e. treats the entry surface as fully reflective in the case that
    // shouldn't arise.
    let entered = length(refract_dir) > 1e-4;
    let refract_dir_local = normalize(to_local * select(reflect_dir, refract_dir, entered));
    let exit_span = ray_box(entry_local, refract_dir_local, 1.0);
    let travel = max(exit_span.y, 0.0);
    let exit_local = entry_local + refract_dir_local * travel;
    // `to_world`/`to_local` are only a *linear* map, not a similarity
    // transform - a warped facet's three edges are unequal in length and no
    // longer mutually perpendicular (see `StickerAnchor`'s doc comment), so
    // mapping a normalized local direction back through `to_world` does not,
    // in general, point the same way as the world-space direction that was
    // normalized into `refract_dir_local` in the first place. `exit_local`
    // was reached by actually traveling along `refract_dir_local` in local
    // space, so the true world-space travel direction - the one Snell's law
    // at the exit surface needs as its incident ray - is this remapped
    // direction, not the original unwarped `refract_dir`: substituting the
    // pre-warp direction would silently diverge from the ray that actually
    // reached this exit point, an error that grows with how much `to_world`
    // skews near this facet - worst right at a sticker's own edges/corners,
    // where adjacent local faces (and their skew) change fastest.
    let travel_dir_world = normalize(to_world * refract_dir_local);
    // The source tunes its inner-color mix/tint against travel distance
    // through its own box, half-extent 0.25 in the same units as its ray
    // origins - so a straight-through ray travels at most ~0.5. `ray_box`
    // here runs in the sticker's *local* frame, a unit cube of half-extent
    // 1.0, where the same ray travels up to ~2.0 (and up to the space
    // diagonal, ~3.46, corner to corner) - four times the source's scale.
    // Left unscaled, `travel` blows the source's mix weight
    // (`0.3 * travel + ...`) past 1.0, over-extrapolating `mix` beyond
    // `ice_inner_color()` instead of blending toward it. Rescaling by the
    // source's box-to-diameter ratio (0.25 / 1.0) reproduces its tuning
    // regardless of this sticker's actual local-frame box size.
    let travel_tint = travel * 0.25;
    let exit_normal_world = normalize(to_world * sticker_local_face_normal(exit_local));
    let exit_point_world = anchor.world_center + to_world * exit_local;

    // Refracting back out of the ice at the exit surface can hit total
    // internal reflection - `refract` returns a zero vector then - in which
    // case the source falls back to reflecting off that same surface
    // instead, same as it does here.
    var exit_dir = refract(travel_dir_world, -exit_normal_world, ICE_REFRACTION_IDX);
    if (length(exit_dir) <= 0.95) {
        exit_dir = reflect(travel_dir_world, -exit_normal_world);
    }

    let refracted_background = ice_sample_background(exit_point_world, exit_dir);
    let reflected_background = ice_sample_background(in.world_position, reflect_dir);

    var refract_color = refracted_background;
    refract_color = mix(refract_color, ice_inner_color(), 0.3 * travel_tint + 0.2 * sqrt(travel_tint * 3.0));
    refract_color += vec3<f32>(travel_tint * 0.3);
    let final_color = mix(refract_color, reflected_background, reflect_alpha);

    // Unlike Fire/Water/Lightning, whose colors are entirely procedural,
    // `refracted_background`/`reflected_background` above are read straight
    // from the actual rendered HDR scene - which can already carry values
    // brighter than anything this shader itself generates (e.g. a nearby
    // Fire sticker's emissive rim). This app composites through a bloom
    // pass keyed off `BLOOM_THRESHOLD = 2.0` (`post_process.wgsl`),
    // documented there as a no-op for every current material - an ice
    // sticker that mixes/adds on top of an already-bright scene sample can
    // cross that threshold and bloom out its own fine bump detail into a
    // soft glow, which reads as "washed out"/less textured right where the
    // background behind it is brightest. Clamping here keeps Ice inside the
    // same no-bloom invariant every other material already holds itself to.
    let bloom_safe_color = min(final_color, vec3<f32>(1.9));

    return vec4<f32>(
        apply_highlight(bloom_safe_color, 1.0, in.instance_index, in.piece_slot),
        1.0,
    );
}
