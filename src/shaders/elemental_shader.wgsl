#import math4d::{camera, compute_sticker_anchor, compute_vertex_geometry, instances, inverse3, transform}
#import sticker_common::{HighlightingUniform, LightUniform, light, highlighting, piece_slots}
#import elemental_common::{hash11, hash21, hash31, value_noise1, value_noise3, fresnel}

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

@vertex
fn vs_main(
    @location(0) vertex_position: vec3<f32>,
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
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
// vertex-displaced. Each of a face's 27 water stickers
// renders its own independent-looking wave patch: the local UV domain is
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

fn sand_color(instance_index: u32, world_position: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    let normal = normalize(world_normal);
    let light_dir = normalize(-light.direction);
    let view_dir = normalize(-world_position);

    let grain = value_noise1(f32(instance_index) * 5.3 + transform.elapsed_seconds * 0.3);
    let albedo = mix(vec3<f32>(0.55, 0.4, 0.2), vec3<f32>(0.75, 0.6, 0.35), grain);

    let ambient = light.ambient * albedo;
    let diffuse_strength = max(dot(normal, light_dir), 0.0);
    let diffuse = diffuse_strength * light.color * albedo;

    let half_dir = normalize(light_dir + view_dir);
    let specular_strength = pow(max(dot(normal, half_dir), 0.0), 8.0);
    let specular = specular_strength * light.color * 0.15;

    return ambient + diffuse + specular;
}

fn leaves_color(instance_index: u32, world_position: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    let normal = normalize(world_normal);
    let light_dir = normalize(-light.direction);
    let view_dir = normalize(-world_position);

    let sway = value_noise1(f32(instance_index) * 1.7 + transform.elapsed_seconds * 0.8);
    let dapple = value_noise1(f32(instance_index) * 9.3 + transform.elapsed_seconds * 2.5);
    let albedo = mix(vec3<f32>(0.1, 0.35, 0.05), vec3<f32>(0.35, 0.6, 0.15), sway) * mix(0.7, 1.0, dapple);

    let ambient = light.ambient * albedo;
    let diffuse_strength = max(dot(normal, light_dir), 0.0);
    let diffuse = diffuse_strength * light.color * albedo;

    let half_dir = normalize(light_dir + view_dir);
    let specular_strength = pow(max(dot(normal, half_dir), 0.0), 16.0);
    let specular = specular_strength * light.color * 0.2;

    return ambient + diffuse + specular;
}

fn crystal_color(instance_index: u32, world_position: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    let normal = normalize(world_normal);
    let light_dir = normalize(-light.direction);
    let view_dir = normalize(-world_position);

    let facet_seed = hash11(f32(instance_index) * 4.0 + floor(transform.elapsed_seconds * 0.5));
    let albedo = mix(vec3<f32>(0.3, 0.05, 0.5), vec3<f32>(0.55, 0.2, 0.8), facet_seed);

    let ambient = light.ambient * albedo;
    let diffuse_strength = max(dot(normal, light_dir), 0.0);
    let banded_diffuse = floor(diffuse_strength * 4.0) / 4.0;
    let diffuse = banded_diffuse * light.color * albedo;

    let half_dir = normalize(light_dir + view_dir);
    let specular_strength = pow(max(dot(normal, half_dir), 0.0), 96.0);
    let specular = specular_strength * light.color;

    let rim = fresnel(normal, view_dir, 2.5) * 0.6;

    return ambient + diffuse + specular + rim * vec3<f32>(0.7, 0.3, 1.0);
}

fn glowing_light_color(instance_index: u32, world_position: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    let seed = f32(instance_index);
    let pulse = 0.5 + 0.5 * sin(transform.elapsed_seconds * 2.0 + seed * 6.28318);
    let base = mix(vec3<f32>(0.85, 0.8, 0.6), vec3<f32>(1.0, 1.0, 0.9), pulse);
    let view_dir = normalize(-world_position);
    let halo = fresnel(world_normal, view_dir, 1.5);
    return base + halo * vec3<f32>(1.0, 1.0, 0.9) * 0.6;
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
            final_color = leaves_color(in.instance_index, in.world_position, in.world_normal);
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
        case 4u: {
            final_color = sand_color(in.instance_index, in.world_position, in.world_normal);
        }
        case 5u: {
            final_color = glowing_light_color(in.instance_index, in.world_position, in.world_normal);
        }
        case 6u: {
            final_color = water_color(in.instance_index, in.world_position, in.world_normal, in.local_position);
        }
        case 7u: {
            final_color = crystal_color(in.instance_index, in.world_position, in.world_normal);
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
    // offset (this used to be `* 100.0`, matching the much larger shifts
    // Water/Lightning use for effects that only ever *sample* noise, never
    // differentiate it) pushes the finite-difference evaluation far enough
    // out that float32 precision can't resolve the `d = 0.005` step
    // `ice_normal_map` diffs by, so it starts returning near-zero gradients
    // and NaNs (see `ice_normal_map`'s guard) - a bug the user found by
    // spotting black dots where Fire's blended pass composited over them.
    // This range still shifts each of a face's 27 stickers by roughly a
    // full noise cell (`ICE_BUMP_UV_SCALE` = 0.2, so `5.0 * 0.2 = 1.0`),
    // decorrelating their patterns same as before, just at a magnitude the
    // derivative stays well-conditioned at.
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
    // direction, not the original unwarped `refract_dir`. Using `refract_dir`
    // there was self-consistent as a world-space computation but physically
    // wrong: it silently substitutes the pre-warp direction for the ray that
    // actually reached this exit point, an error that grows with how much
    // `to_world` skews near this facet - worst right at a sticker's own
    // edges/corners, where adjacent local faces (and their skew) change
    // fastest, showing up as a visibly wrong refracted sample right at those
    // edges.
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
    // Fire sticker's emissive rim). The source assumed its own background
    // samples (a bounded floor/sky) stayed under ~1.7 and relied on its
    // final `pow(c, 0.4545)` hitting an LDR framebuffer to hard-clip
    // anything over 1.0 with no visible side effect. This app instead
    // composites through a bloom pass keyed off `BLOOM_THRESHOLD = 2.0`
    // (`post_process.wgsl`), documented there as a no-op for every current
    // material - an ice sticker that mixes/adds on top of an
    // already-bright scene sample can cross that threshold and bloom out
    // its own fine bump detail into a soft glow, which reads as "washed
    // out"/less textured right where the background behind it is brightest.
    // Clamping here keeps Ice inside the same no-bloom invariant every
    // other material already holds itself to, rather than laundering
    // unbounded scene brightness through unclipped.
    let bloom_safe_color = min(final_color, vec3<f32>(1.9));

    return vec4<f32>(
        apply_highlight(bloom_safe_color, 1.0, in.instance_index, in.piece_slot),
        1.0,
    );
}
