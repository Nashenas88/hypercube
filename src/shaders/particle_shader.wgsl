#import math4d::{StickerAnchor, compute_sticker_anchor, instances, transform, camera}
#import sticker_common::{sticker_order}
#import elemental_common::{ICE_TWINKLE_HZ, LIGHTNING_STROBE_HZ, hash11, hash21, strobe}

const TAU: f32 = 6.28318530718;

// Half-extent of a sticker's mesh cube before `transform.sticker_scale`. Must
// match `BASE_STICKER_SIZE` in math.rs, which renderer.rs premultiplies into
// the cube vertices.
const STICKER_HALF_EXTENT: f32 = 0.33333334;

// Radius, in sticker radii, a particle is born at. Past the cube's longest
// diagonal (sqrt 3) so a new particle isn't swallowed by the depth test
// against the sticker it came from.
const BIRTH_RADIUS: f32 = 1.8;

// One corner of a unit quad, as two triangles over 6 vertices. Returned from
// a switch rather than a runtime-indexed array, which most drivers spill to
// scratch memory.
fn quad_corner(vertex_index: u32) -> vec2<f32> {
    switch (vertex_index) {
        case 0u: {
            return vec2<f32>(-1.0, -1.0);
        }
        case 1u: {
            return vec2<f32>(1.0, -1.0);
        }
        case 2u: {
            return vec2<f32>(1.0, 1.0);
        }
        case 3u: {
            return vec2<f32>(-1.0, -1.0);
        }
        case 4u: {
            return vec2<f32>(1.0, 1.0);
        }
        default: {
            return vec2<f32>(-1.0, 1.0);
        }
    }
}

fn unproject(clip_xy: vec2<f32>) -> vec3<f32> {
    let point = camera.view_proj_inv * vec4<f32>(clip_xy, 0.0, 1.0);
    return point.xyz / point.w;
}

// Recovers the camera's world-space right/up axes from the inverse
// view-projection by unprojecting two opposite near-plane points and taking
// the direction between them, so billboards need no extra uniform data.
// `camera.view_proj_inv` is translation-free (it inverts a rotation-only view
// matrix, for the skybox), which costs nothing here: the missing eye
// translation is the same constant on both points and cancels in the
// difference.
fn camera_right_world() -> vec3<f32> {
    return normalize(unproject(vec2<f32>(1.0, 0.0)) - unproject(vec2<f32>(-1.0, 0.0)));
}

fn camera_up_world() -> vec3<f32> {
    return normalize(unproject(vec2<f32>(0.0, 1.0)) - unproject(vec2<f32>(0.0, -1.0)));
}

// A direction distributed evenly over the whole unit sphere. `height` selects
// the z band and `angle_seed` the rotation within it; sampling z uniformly
// (rather than an inclination angle) is what keeps the poles from bunching.
fn direction_on_sphere(height_seed: f32, angle_seed: f32) -> vec3<f32> {
    let height = height_seed * 2.0 - 1.0;
    let ring_radius = sqrt(max(0.0, 1.0 - height * height));
    let angle = angle_seed * TAU;
    return vec3<f32>(ring_radius * cos(angle), ring_radius * sin(angle), height);
}

// Maps a direction expressed in sticker radii onto world space through the
// sticker's own projected cube, so a particle travels the same distance
// relative to the sticker along every local axis - which is an unequal
// world-space distance wherever the 4D projection has warped the cube.
fn sticker_to_world(anchor: StickerAnchor, local: vec3<f32>) -> vec3<f32> {
    return local.x * anchor.edge_x + local.y * anchor.edge_y + local.z * anchor.edge_z;
}

// The sticker's mean projected radius, for sizing a billboard that should
// track its apparent size.
fn apparent_radius(anchor: StickerAnchor) -> f32 {
    return (length(anchor.edge_x) + length(anchor.edge_y) + length(anchor.edge_z)) / 3.0;
}

// Per-element emitter tuning consumed by `styled_particle`, the analytic
// path every element but Fire still emits through. Distances are all in
// sticker radii, mapped through the anchor's warped local frame at use, so
// an element's tuning doesn't change with a sticker's projected size.
// `emission` of zero suppresses the element's particles entirely.
struct ParticleStyle {
    // Seconds one particle takes to travel its arc and fade out. Chosen from
    // values that divide 3600 evenly so the modulo-3600 wrap of
    // `transform.elapsed_seconds` doesn't restart every particle mid-flight.
    lifetime: f32,
    // Distance travelled along the launch direction over a full lifetime.
    speed: f32,
    // Pull back toward the sticker over the square of age, in the same units
    // as `speed`.
    gravity: f32,
    // Billboard half-extent at birth, before the age curve shrinks it.
    size: f32,
    color_hot: vec3<f32>,
    color_cool: vec3<f32>,
    emission: f32,
    // Rate of the per-sticker gate that decides whether a particle born at a
    // given moment emits at all, so an element can fire in bursts rather than
    // as a steady stream. Zero emits continuously.
    burst_hz: f32,
    // Fraction of those gate windows that are open.
    burst_chance: f32,
    // Rate at which a live particle's brightness is re-hashed, so it winks
    // while it flies rather than shining steadily. Seeded per particle, so an
    // element twinkles as scattered points instead of a whole sticker
    // pulsing. Zero holds brightness constant.
    twinkle_hz: f32,
}

// Zero-valued starting point for a `ParticleStyle`: no travel, no burst
// gating, no twinkle. Each element's function below overrides only the
// fields its look actually needs.
fn default_style() -> ParticleStyle {
    var style: ParticleStyle;
    style.lifetime = 1.0;
    style.speed = 0.0;
    style.gravity = 0.0;
    style.size = 0.0;
    style.color_hot = vec3<f32>(0.0, 0.0, 0.0);
    style.color_cool = vec3<f32>(0.0, 0.0, 0.0);
    style.emission = 0.0;
    style.burst_hz = 0.0;
    style.burst_chance = 1.0;
    style.twinkle_hz = 0.0;
    return style;
}

// Whether a particle born at `birth_time` falls inside one of its sticker's
// open burst windows. Seeded per sticker rather than per particle, so a whole
// sticker's worth of particles fires together. Sharing `strobe` with
// `lightning_color` in elemental_shader.wgsl is what puts Lightning's sparks
// on the same beat as its flashes: at a matching rate, a `burst_chance` at or
// below that material's own 0.4 makes these windows a subset of the ones it
// lights the sticker for, so sparks only ever fly on a flash.
fn burst_open(style: ParticleStyle, sticker_index: u32, birth_time: f32) -> bool {
    if (style.burst_hz <= 0.0) {
        return true;
    }
    return strobe(sticker_index, birth_time, style.burst_hz) >= 1.0 - style.burst_chance;
}

// Brightness multiplier for one particle at `time`. Sampled at the current
// moment rather than at birth, which is what makes a particle wink partway
// through its flight; it only dims rather than cutting to nothing, so a mote
// fades between bright and faint instead of blinking out of existence.
fn twinkle_scale(style: ParticleStyle, instance_index: u32, time: f32) -> f32 {
    if (style.twinkle_hz <= 0.0) {
        return 1.0;
    }
    return mix(0.25, 1.0, strobe(instance_index, time, style.twinkle_hz));
}

// One particle's resolved world position, billboard half-extent, color and
// alpha for this frame. `visible` is false wherever the particle shouldn't
// draw at all (sticker culled, element emits nothing, or a burst window
// currently shut), so `vs_main` can bail out without a billboard to place.
struct ParticleSample {
    visible: bool,
    world_position: vec3<f32>,
    size: f32,
    color: vec3<f32>,
    alpha: f32,
}

// The analytic emitter path every element but Fire still uses: launch on a
// random sphere direction, travel outward under `style.gravity`, cool from
// `color_hot` to `color_cool`, and fade in and out across the lifetime.
fn styled_particle(
    style: ParticleStyle,
    sticker_index: u32,
    instance_index: u32,
    anchor: StickerAnchor,
) -> ParticleSample {
    var sample: ParticleSample;
    sample.visible = false;

    if (!anchor.visible || style.emission <= 0.0) {
        return sample;
    }

    // Stagger the particles of one sticker across the lifetime so they emit
    // as a stream rather than all at once.
    let phase = hash11(f32(instance_index) * 0.37) * style.lifetime;
    let cycles = (transform.elapsed_seconds + phase) / style.lifetime;
    let cycle = floor(cycles);
    let age = fract(cycles);

    // Gated on when the particle was born rather than on now, so one that
    // launched during an open window keeps flying after that window shuts.
    let birth_time = transform.elapsed_seconds - age * style.lifetime;
    if (!burst_open(style, sticker_index, birth_time)) {
        return sample;
    }

    // Reseeded every cycle, so each life gets a fresh direction instead of
    // the particle retracing one fixed path forever.
    let angle_seed = hash21(vec2<f32>(f32(instance_index), cycle));
    let height_seed = hash21(vec2<f32>(f32(instance_index) + 37.0, cycle));
    let direction = direction_on_sphere(height_seed, angle_seed);

    let distance = BIRTH_RADIUS + style.speed * age - style.gravity * age * age;

    sample.visible = true;
    sample.world_position = anchor.world_center + sticker_to_world(anchor, direction * distance);
    sample.size = style.size * apparent_radius(anchor) * mix(1.0, 0.45, age);
    sample.color = mix(style.color_hot, style.color_cool, age);
    // Fade in over the first tenth of the life so a particle appears rather
    // than pops, then fade out across the rest.
    sample.alpha = style.emission
        * smoothstep(0.0, 0.1, age)
        * (1.0 - age)
        * twinkle_scale(style, instance_index, transform.elapsed_seconds);
    return sample;
}

// Ice: frost motes barely drifting off the surface, winking on the same beat
// the material buckets its own sparkle at.
fn ice_particle(sticker_index: u32, instance_index: u32, anchor: StickerAnchor) -> ParticleSample {
    var style = default_style();
    style.lifetime = 3.0;
    style.speed = 0.5;
    style.size = 0.14;
    style.color_hot = vec3<f32>(0.85, 0.97, 1.0);
    style.color_cool = vec3<f32>(0.6, 0.85, 1.0);
    style.emission = 0.5;
    style.twinkle_hz = ICE_TWINKLE_HZ;
    return styled_particle(style, sticker_index, instance_index, anchor);
}

// Leaves: broad flecks tumbling slowly outward, sagging back a little as
// they go, and darkening from new growth to leaf litter over their life.
fn leaves_particle(sticker_index: u32, instance_index: u32, anchor: StickerAnchor) -> ParticleSample {
    var style = default_style();
    style.lifetime = 3.0;
    style.speed = 1.0;
    style.gravity = 0.4;
    style.size = 0.18;
    style.color_hot = vec3<f32>(0.45, 0.7, 0.2);
    style.color_cool = vec3<f32>(0.2, 0.4, 0.08);
    style.emission = 0.45;
    return styled_particle(style, sticker_index, instance_index, anchor);
}

// Lightning: small hard sparks flicked out fast and straight, gated onto the
// same strobe the material flashes its sticker on so they read as thrown by
// the flash itself.
fn lightning_particle(sticker_index: u32, instance_index: u32, anchor: StickerAnchor) -> ParticleSample {
    var style = default_style();
    style.lifetime = 0.5;
    style.speed = 3.0;
    style.size = 0.10;
    style.color_hot = vec3<f32>(1.0, 0.95, 0.55);
    style.color_cool = vec3<f32>(1.0, 0.55, 0.05);
    style.emission = 1.4;
    style.burst_hz = LIGHTNING_STROBE_HZ;
    style.burst_chance = 0.15;
    return styled_particle(style, sticker_index, instance_index, anchor);
}

// Fire: embers thrown off the sticker in every direction, cooling from
// yellow to deep red as they slow.
fn fire_particle(sticker_index: u32, instance_index: u32, anchor: StickerAnchor) -> ParticleSample {
    var style = default_style();
    style.lifetime = 1.2;
    style.speed = 1.2;
    style.gravity = 0.9;
    style.size = 0.22;
    style.color_hot = vec3<f32>(1.0, 0.75, 0.25);
    style.color_cool = vec3<f32>(0.85, 0.12, 0.0);
    style.emission = 1.0;
    return styled_particle(style, sticker_index, instance_index, anchor);
}

// Sand: fine dim grains hanging close to the surface, small and numerous
// enough to read as a haze rather than as separate particles.
fn sand_particle(sticker_index: u32, instance_index: u32, anchor: StickerAnchor) -> ParticleSample {
    var style = default_style();
    style.lifetime = 2.0;
    style.speed = 0.8;
    style.gravity = 0.3;
    style.size = 0.09;
    style.color_hot = vec3<f32>(0.8, 0.68, 0.42);
    style.color_cool = vec3<f32>(0.5, 0.38, 0.2);
    style.emission = 0.4;
    return styled_particle(style, sticker_index, instance_index, anchor);
}

// Glowing Light: large soft motes drifting slowly outward and dimming, warm
// white throughout.
fn glowing_light_particle(
    sticker_index: u32,
    instance_index: u32,
    anchor: StickerAnchor,
) -> ParticleSample {
    var style = default_style();
    style.lifetime = 3.0;
    style.speed = 0.9;
    style.size = 0.30;
    style.color_hot = vec3<f32>(1.0, 0.98, 0.85);
    style.color_cool = vec3<f32>(0.85, 0.78, 0.5);
    style.emission = 0.55;
    return styled_particle(style, sticker_index, instance_index, anchor);
}

// Water: droplets flung clear of the sticker and pulled back again, with
// gravity matching speed so a droplet lands back at the radius it launched
// from just as it fades.
fn water_particle(sticker_index: u32, instance_index: u32, anchor: StickerAnchor) -> ParticleSample {
    var style = default_style();
    style.lifetime = 1.5;
    style.speed = 2.4;
    style.gravity = 2.4;
    style.size = 0.16;
    style.color_hot = vec3<f32>(0.55, 0.8, 1.0);
    style.color_cool = vec3<f32>(0.1, 0.35, 0.8);
    style.emission = 0.7;
    return styled_particle(style, sticker_index, instance_index, anchor);
}

// Dark: kept only for the one-function-per-kind symmetry every other
// zero-budget element (Fire, Water, Lightning, Ice) maintains -
// `Theme::particles_per_kind` zeroes this kind's budget, so the `case 7u`
// dispatch below is never actually invoked at runtime. Values otherwise
// unchanged from the Crystal material this replaced.
fn dark_particle(sticker_index: u32, instance_index: u32, anchor: StickerAnchor) -> ParticleSample {
    var style = default_style();
    style.lifetime = 1.5;
    style.speed = 0.2;
    style.size = 0.13;
    style.color_hot = vec3<f32>(0.9, 0.7, 1.0);
    style.color_cool = vec3<f32>(0.5, 0.15, 0.85);
    style.emission = 1.2;
    style.burst_hz = 1.0;
    style.burst_chance = 0.2;
    return styled_particle(style, sticker_index, instance_index, anchor);
}

struct ParticleVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Position within the billboard, -1 to 1 on each axis, for the radial
    // falloff that rounds the quad off into a soft dot.
    @location(0) corner: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) alpha: f32,
}

// Places `vertex_index`'s corner of one billboarded particle. The particle
// carries no stored state: its sample comes entirely from the per-`kind`
// function `sticker_order[instance_index]`'s kind dispatches to below, each
// hashing `instance_index` against the current lifetime cycle, so the whole
// system is a function of `transform.elapsed_seconds`.
@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> ParticleVertexOutput {
    var out: ParticleVertexOutput;
    out.corner = vec2<f32>(0.0, 0.0);
    out.color = vec3<f32>(0.0, 0.0, 0.0);
    out.alpha = 0.0;

    let sticker_index = sticker_order[instance_index];
    let anchor = compute_sticker_anchor(
        sticker_index,
        STICKER_HALF_EXTENT * transform.sticker_scale,
    );

    var sample: ParticleSample;
    switch (instances[sticker_index].kind) {
        case 0u: {
            sample = ice_particle(sticker_index, instance_index, anchor);
        }
        case 1u: {
            sample = leaves_particle(sticker_index, instance_index, anchor);
        }
        case 2u: {
            sample = lightning_particle(sticker_index, instance_index, anchor);
        }
        case 3u: {
            sample = fire_particle(sticker_index, instance_index, anchor);
        }
        case 4u: {
            sample = sand_particle(sticker_index, instance_index, anchor);
        }
        case 5u: {
            sample = glowing_light_particle(sticker_index, instance_index, anchor);
        }
        case 6u: {
            sample = water_particle(sticker_index, instance_index, anchor);
        }
        case 7u: {
            sample = dark_particle(sticker_index, instance_index, anchor);
        }
        default: {
            sample.visible = false;
        }
    }

    if (!sample.visible) {
        out.clip_position = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        return out;
    }

    let corner = quad_corner(vertex_index);
    let offset = camera_right_world() * corner.x * sample.size
        + camera_up_world() * corner.y * sample.size;

    out.clip_position = camera.view_proj * vec4<f32>(sample.world_position + offset, 1.0);
    out.corner = corner;
    out.color = sample.color;
    out.alpha = sample.alpha;

    return out;
}

@fragment
fn fs_main(in: ParticleVertexOutput) -> @location(0) vec4<f32> {
    let falloff = smoothstep(1.0, 0.0, length(in.corner));
    return vec4<f32>(in.color * falloff, in.alpha * falloff);
}
