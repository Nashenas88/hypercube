#import math4d::CameraUniform
#import sticker_common::LightUniform

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
// Declared locally (rather than importing `sticker_common::light`, which is
// fixed at binding 5 there for the main sticker bind group) since the gizmo
// has its own small bind group with the light at binding 1.
@group(0) @binding(1) var<uniform> light: LightUniform;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    out.color = color;
    out.world_position = position;
    out.world_normal = normal;
    return out;
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    // The ring/marker tubes are double-sided (`cull_mode: None`, since the
    // camera can orbit to see either face of a thin tube), so the flat
    // per-triangle normal `recompute_flat_normals` computed on the CPU may
    // point away from the camera for a back-facing triangle - flip it back
    // toward the viewer using the rasterizer's own front/back determination,
    // the same two-sided lighting fix classic/elemental shading doesn't need
    // since the puzzle's own stickers are single-sided.
    let normal = normalize(select(-in.world_normal, in.world_normal, is_front));
    let light_dir = normalize(-light.direction);
    let view_dir = normalize(-in.world_position);

    let ambient = light.ambient * in.color.rgb;

    let diffuse_strength = max(dot(normal, light_dir), 0.0);
    let diffuse = diffuse_strength * light.color * in.color.rgb;

    let half_dir = normalize(light_dir + view_dir);
    let specular_strength = pow(max(dot(normal, half_dir), 0.0), 32.0);
    let specular = specular_strength * light.color * 0.3;

    let final_color = ambient + diffuse + specular;
    return vec4<f32>(final_color, in.color.a);
}
