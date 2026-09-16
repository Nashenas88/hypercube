#import math4d::{CameraUniform, compute_vertex_geometry, instances, transform}
#import sticker_common::{HighlightingUniform, LightUniform, light, highlighting, piece_slots, tutorial_flash_pulse}

struct KindColors {
    colors: array<vec4<f32>, 8>,
};

@group(0) @binding(7)
var<uniform> kind_colors: KindColors;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) instance_index: u32,
    @location(4) piece_slot: u32,
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
        out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return out;
    }

    out.color = kind_colors.colors[instances[instance_index].kind];
    out.instance_index = instance_index;
    out.piece_slot = piece_slots[instance_index];

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let light_dir = normalize(-light.direction);
    let view_dir = normalize(-in.world_position);

    let ambient = light.ambient * in.color.rgb;

    let diffuse_strength = max(dot(normal, light_dir), 0.0);
    let diffuse = diffuse_strength * light.color * in.color.rgb;

    let half_dir = normalize(light_dir + view_dir);
    let specular_strength = pow(max(dot(normal, half_dir), 0.0), 32.0);
    let specular = specular_strength * light.color * 0.3;

    var final_color = ambient + diffuse + specular;

    // Apply highlighting: the exact hovered sticker gets its own color, the
    // rest of the hovered piece's stickers get a dimmer shared highlight.
    if (in.instance_index == highlighting.hovered_sticker_index) {
        final_color = mix(final_color, highlighting.highlight_color.rgb, highlighting.highlight_color.a);
    } else if (in.piece_slot == highlighting.hovered_piece_slot) {
        final_color = mix(final_color, highlighting.piece_highlight_color.rgb, highlighting.piece_highlight_color.a);
    }

    let flash = tutorial_flash_pulse(instances[in.instance_index].facet_count, transform.elapsed_seconds);
    if (flash > 0.0) {
        final_color = mix(final_color, highlighting.highlight_color.rgb, flash * highlighting.highlight_color.a);
    }

    return vec4<f32>(final_color, in.color.a);
}

@group(0) @binding(0)
var<uniform> sky_camera: CameraUniform;

@group(0) @binding(1)
var sky_texture: texture_cube<f32>;

@group(0) @binding(2)
var sky_sampler: sampler;

struct SkyboxVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
}

@vertex
fn vs_sky(@location(0) position: vec2<f32>) -> SkyboxVertexOutput {
    var out: SkyboxVertexOutput;

    let x = position.x;
    let y = position.y;
    
    out.clip_position = vec4<f32>(x, y, 1.0, 1.0);

    // Convert screen position back to world direction for cubemap sampling using
    // the translation-free inverse view-projection matrix. Leaving the result
    // un-normalized keeps it affine in (x, y), so linear interpolation across the
    // quad's four corners lands on the exact per-pixel direction.
    let world_pos = sky_camera.view_proj_inv * vec4<f32>(x, y, 1.0, 1.0);
    out.world_position = world_pos.xyz / world_pos.w;

    return out;
}

@fragment
fn fs_sky(in: SkyboxVertexOutput) -> @location(0) vec4<f32> {
    return textureSample(sky_texture, sky_sampler, normalize(in.world_position));
}