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

// Hashed on/off window for `seed` at `time`, `hz` windows per second. Returns
// the window's raw hash rather than a decision, so callers pick their own
// threshold and a stricter one selects a subset of a looser one's windows -
// which is what keeps two effects sharing a seed firing on the same beat
// instead of on unrelated ones.
fn strobe(seed: u32, time: f32, hz: f32) -> f32 {
    let window = floor(time * hz);
    return hash11(f32(seed) * 5.0 + window);
}

fn fresnel(normal: vec3<f32>, view_dir: vec3<f32>, power: f32) -> f32 {
    return pow(1.0 - clamp(dot(normalize(normal), normalize(view_dir)), 0.0, 1.0), power);
}
