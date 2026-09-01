// HDR post-processing chain: bright-pass, a two-pass separable blur, and the
// final composite that blits the scene (plus any bloom) into iced's surface.
// Every pass shares one fullscreen-triangle vertex shader. Bright-pass and
// the two blur passes use its `uv` output directly: their own render target
// is never viewport-restricted, so `uv` already spans that target's whole
// 0..1 range regardless of its resolution, which is exactly what lets
// bright-pass address a full-res input while writing a half-res output.
// Composite instead derives its own UV from `@builtin(position)`, since it
// alone is viewport-restricted (to `Renderer::bounds`, a sub-rectangle of a
// larger `scene_texture`) and needs the absolute pixel address that gives,
// not a UV renormalized to just the visible sub-rectangle.

struct FullscreenOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Unused by `fs_composite`, which derives its own UV from `clip_position`
    // instead (see file header).
    @location(0) uv: vec2<f32>,
}

// A single triangle covering the whole clip-space square, generated from the
// vertex index alone - no vertex buffer, and no seam down a quad's diagonal.
@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> FullscreenOutput {
    var out: FullscreenOutput;
    let x = f32((vertex_index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vertex_index & 2u) * 2.0 - 1.0;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    // NDC y points up; texture v points down.
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var input_sampler: sampler;

// Colors at or below this never bloom. Set well above what any current
// material can write (the highest today is Fire's unclamped rim add, up to
// ~1.5 on its red channel) so this pass stays a no-op until a material is
// deliberately reauthored to write brighter linear light on purpose.
const BLOOM_THRESHOLD: f32 = 2.0;

@fragment
fn fs_bright_pass(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let color = textureSample(input_texture, input_sampler, in.uv).rgb;
    let bright = max(color - vec3<f32>(BLOOM_THRESHOLD), vec3<f32>(0.0));
    return vec4<f32>(bright, 1.0);
}

// Normalized 9-tap Gaussian half-kernel (center + 4 taps), reused for both
// blur directions.
const BLUR_WEIGHT_0: f32 = 0.227027;
const BLUR_WEIGHT_1: f32 = 0.1945946;
const BLUR_WEIGHT_2: f32 = 0.1216216;
const BLUR_WEIGHT_3: f32 = 0.054054;
const BLUR_WEIGHT_4: f32 = 0.016216;

fn blur(uv: vec2<f32>, texel: vec2<f32>) -> vec3<f32> {
    var result = textureSample(input_texture, input_sampler, uv).rgb * BLUR_WEIGHT_0;
    result += textureSample(input_texture, input_sampler, uv + texel * 1.0).rgb * BLUR_WEIGHT_1;
    result += textureSample(input_texture, input_sampler, uv - texel * 1.0).rgb * BLUR_WEIGHT_1;
    result += textureSample(input_texture, input_sampler, uv + texel * 2.0).rgb * BLUR_WEIGHT_2;
    result += textureSample(input_texture, input_sampler, uv - texel * 2.0).rgb * BLUR_WEIGHT_2;
    result += textureSample(input_texture, input_sampler, uv + texel * 3.0).rgb * BLUR_WEIGHT_3;
    result += textureSample(input_texture, input_sampler, uv - texel * 3.0).rgb * BLUR_WEIGHT_3;
    result += textureSample(input_texture, input_sampler, uv + texel * 4.0).rgb * BLUR_WEIGHT_4;
    result += textureSample(input_texture, input_sampler, uv - texel * 4.0).rgb * BLUR_WEIGHT_4;
    return result;
}

@fragment
fn fs_blur_h(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(input_texture));
    return vec4<f32>(blur(in.uv, vec2<f32>(1.0 / dims.x, 0.0)), 1.0);
}

@fragment
fn fs_blur_v(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(input_texture));
    return vec4<f32>(blur(in.uv, vec2<f32>(0.0, 1.0 / dims.y)), 1.0);
}

@group(0) @binding(0)
var scene_texture: texture_2d<f32>;

@group(0) @binding(1)
var bloom_texture: texture_2d<f32>;

@group(0) @binding(2)
var composite_sampler: sampler;

@fragment
fn fs_composite(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(scene_texture));
    let uv = in.clip_position.xy / dims;
    let scene = textureSample(scene_texture, composite_sampler, uv).rgb;
    // Bilinear-sampled from the half-res bloom texture, upsampling it back
    // to scene resolution for free.
    let bloom = textureSample(bloom_texture, composite_sampler, uv).rgb;
    return vec4<f32>(scene + bloom, 1.0);
}
