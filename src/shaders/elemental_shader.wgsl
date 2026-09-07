#import math4d::{camera, compute_sticker_anchor, compute_vertex_geometry, instances, inverse3, transform}
#import sticker_common::{HighlightingUniform, LightUniform, light, highlighting, piece_slots}
#import elemental_common::{ICE_TWINKLE_HZ, LIGHTNING_STROBE_HZ, hash11, hash21, hash31, value_noise1, value_noise3, fresnel, strobe}

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
// dominant mesh axis is `n`.
fn water_face_uv(p: vec3<f32>, n: vec3<f32>) -> vec2<f32> {
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
// normalized -1..1 domain `water_face_uv` returns, so neighboring stickers'
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
// position within the triangle - no `face_3d` plumbing needed.
fn water_local_face_normal(local_position: vec3<f32>) -> vec3<f32> {
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
    let local_face_normal = water_local_face_normal(local_position);
    let normalized_local = local_position / STICKER_HALF_EXTENT;
    let local_uv = water_face_uv(normalized_local, local_face_normal);

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

fn ice_color(instance_index: u32, world_position: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    let normal = normalize(world_normal);
    let light_dir = normalize(-light.direction);
    let view_dir = normalize(-world_position);

    let shimmer = value_noise1(f32(instance_index) * 2.1 + transform.elapsed_seconds * 0.6);
    let albedo = mix(vec3<f32>(0.75, 0.9, 0.95), vec3<f32>(0.9, 0.98, 1.0), shimmer);

    let ambient = light.ambient * albedo;
    let diffuse_strength = max(dot(normal, light_dir), 0.0);
    let diffuse = diffuse_strength * light.color * albedo;

    let half_dir = normalize(light_dir + view_dir);
    let specular_strength = pow(max(dot(normal, half_dir), 0.0), 128.0);
    let specular = specular_strength * light.color * 1.2;

    let sparkle_time_bucket = floor(transform.elapsed_seconds * ICE_TWINKLE_HZ);
    let sparkle_phase = hash21(vec2<f32>(f32(instance_index), sparkle_time_bucket));
    let sparkle = step(0.97, sparkle_phase) * fresnel(normal, view_dir, 1.0);

    return ambient + diffuse + specular + sparkle * vec3<f32>(1.0, 1.0, 1.0);
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

fn lightning_color(instance_index: u32, world_position: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    let flash = strobe(instance_index, transform.elapsed_seconds, LIGHTNING_STROBE_HZ);
    let brightness = step(0.6, flash);
    let base = mix(vec3<f32>(0.2, 0.18, 0.05), vec3<f32>(1.0, 0.95, 0.5), brightness);
    let view_dir = normalize(-world_position);
    let rim = fresnel(world_normal, view_dir, 1.0) * brightness;
    return base + rim * vec3<f32>(1.0, 0.9, 0.3);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var final_color: vec3<f32>;

    switch (in.kind) {
        case 0u: {
            final_color = ice_color(in.instance_index, in.world_position, in.world_normal);
        }
        case 1u: {
            final_color = leaves_color(in.instance_index, in.world_position, in.world_normal);
        }
        case 2u: {
            final_color = lightning_color(in.instance_index, in.world_position, in.world_normal);
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
