#import math4d::{compute_vertex_geometry, instances, transform}
#import sticker_common::{HighlightingUniform, LightUniform, light, highlighting}
#import elemental_common::{LIGHTNING_STROBE_HZ, hash11, hash21, value_noise1, fresnel, strobe}

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

    let sparkle_time_bucket = floor(transform.elapsed_seconds * 3.0);
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
        case 3u: {
            final_color = fire_color(in.instance_index, in.world_position, in.world_normal);
        }
        case 4u: {
            final_color = sand_color(in.instance_index, in.world_position, in.world_normal);
        }
        case 5u: {
            final_color = glowing_light_color(in.instance_index, in.world_position, in.world_normal);
        }
        case 6u: {
            final_color = water_color(in.instance_index, in.world_position, in.world_normal);
        }
        case 7u: {
            final_color = crystal_color(in.instance_index, in.world_position, in.world_normal);
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
