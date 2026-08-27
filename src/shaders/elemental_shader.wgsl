#import math4d::{compute_vertex_geometry, instances, transform}
#import sticker_common::{HighlightingUniform, LightUniform, light, highlighting}
#import elemental_common::{value_noise1, fresnel}

@group(0) @binding(5)
var<storage, read> piece_slots: array<u32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) instance_index: u32,
    @location(3) piece_slot: u32,
    @location(4) kind: u32,
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

fn fire_color(instance_index: u32, world_position: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    let seed = f32(instance_index);
    let flicker = value_noise1(seed * 3.7 + transform.elapsed_seconds * 4.0);
    let base = mix(vec3<f32>(0.8, 0.1, 0.0), vec3<f32>(1.0, 0.75, 0.15), flicker);
    let view_dir = normalize(-world_position);
    let rim = fresnel(world_normal, view_dir, 2.0);
    return base + rim * vec3<f32>(1.0, 0.6, 0.2) * 0.5;
}

fn water_color(instance_index: u32, world_position: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    let normal = normalize(world_normal);
    let light_dir = normalize(-light.direction);
    let view_dir = normalize(-world_position);

    let wave = value_noise1(f32(instance_index) * 1.3 + transform.elapsed_seconds * 1.5);
    let albedo = mix(vec3<f32>(0.05, 0.2, 0.5), vec3<f32>(0.2, 0.5, 0.8), wave);

    let ambient = light.ambient * albedo;
    let diffuse_strength = max(dot(normal, light_dir), 0.0);
    let diffuse = diffuse_strength * light.color * albedo;

    let half_dir = normalize(light_dir + view_dir);
    let specular_strength = pow(max(dot(normal, half_dir), 0.0), 64.0);
    let specular = specular_strength * light.color;

    let sheen = fresnel(normal, view_dir, 3.0) * 0.4;

    return ambient + diffuse + specular + sheen * vec3<f32>(0.6, 0.8, 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var final_color: vec3<f32>;

    switch (in.kind) {
        case 0u: {
            final_color = fire_color(in.instance_index, in.world_position, in.world_normal);
        }
        case 2u: {
            final_color = water_color(in.instance_index, in.world_position, in.world_normal);
        }
        default: {
            final_color = vec3<f32>(0.5, 0.5, 0.5);
        }
    }

    if (in.instance_index == highlighting.hovered_sticker_index) {
        final_color = mix(final_color, highlighting.highlight_color.rgb, highlighting.highlight_color.a);
    } else if (in.piece_slot == highlighting.hovered_piece_slot) {
        final_color = mix(final_color, highlighting.piece_highlight_color.rgb, highlighting.piece_highlight_color.a);
    }

    return vec4<f32>(final_color, 1.0);
}
