// Small numeric helpers shared by the Elemental theme's per-element materials
// and its particle emitters.
#define_import_path elemental_common

// Windows per second of Lightning's strobe.
const LIGHTNING_STROBE_HZ: f32 = 5.0;

// Windows per second of Ice's twinkle. Shared so the material's sparkle and
// the frost motes wink at one rate; the two pick their windows from different
// hashes, so they share the beat without firing together.
const ICE_TWINKLE_HZ: f32 = 3.0;

fn hash11(x: f32) -> f32 {
    return fract(sin(x * 127.1) * 43758.5453123);
}

fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn value_noise1(x: f32) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let a = hash11(i);
    let b = hash11(i + 1.0);
    return mix(a, b, smoothstep(0.0, 1.0, f));
}

// Bilinearly-interpolated value noise over the 4 hashed corners of the unit
// cell containing `p`, smoothed by the cubic `3t^2 - 2t^3` - the 2D
// counterpart of `value_noise3`, built on `hash21` instead of `hash31`.
fn value_noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Deliberately not the `sin`-based construction `hash11`/`hash21` use: a
// volume sampled at thousands of points per pixel drives its argument far
// enough out that `sin`'s periodicity starts aliasing against the multiply,
// collapsing what should be organic variation into regular geometric bands.
// Folding the input into a unit cell first keeps the argument small no
// matter how far the sample position travels.
fn hash31(p_in: vec3<f32>) -> f32 {
    var p = fract(p_in * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

// Trilinearly-interpolated value noise over the 8 hashed corners of the unit
// cell containing `x`, smoothed by the cubic `3t^2 - 2t^3`.
fn value_noise3(x: vec3<f32>) -> f32 {
    let i = floor(x);
    let t = fract(x);
    let f = t * t * (3.0 - 2.0 * t);

    let x00 = mix(hash31(i), hash31(i + vec3<f32>(1.0, 0.0, 0.0)), f.x);
    let x10 = mix(hash31(i + vec3<f32>(0.0, 1.0, 0.0)), hash31(i + vec3<f32>(1.0, 1.0, 0.0)), f.x);
    let x01 = mix(hash31(i + vec3<f32>(0.0, 0.0, 1.0)), hash31(i + vec3<f32>(1.0, 0.0, 1.0)), f.x);
    let x11 = mix(hash31(i + vec3<f32>(0.0, 1.0, 1.0)), hash31(i + vec3<f32>(1.0, 1.0, 1.0)), f.x);

    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

// Hashed on/off window for `seed` at `time`, `hz` windows per second. Returns
// the window's raw hash rather than a decision, so callers pick their own
// threshold and a stricter one selects a subset of a looser one's windows -
// which is what keeps two effects sharing a seed firing on the same beat
// instead of on unrelated ones.
fn strobe(seed: u32, time: f32, hz: f32) -> f32 {
    let window = floor(time * hz);
    return hash11(f32(seed) * 5.0 + window);
}
