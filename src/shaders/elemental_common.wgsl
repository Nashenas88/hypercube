// Small numeric helpers shared by the Elemental theme's per-element materials.
#define_import_path elemental_common

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

fn fresnel(normal: vec3<f32>, view_dir: vec3<f32>, power: f32) -> f32 {
    return pow(1.0 - clamp(dot(normalize(normal), normalize(view_dir)), 0.0, 1.0), power);
}
