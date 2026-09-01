// Per-sticker identity, lighting and hover-highlighting state shared by
// sticker material shaders and the particle pipeline.
#define_import_path sticker_common

struct LightUniform {
    direction: vec3<f32>,
    // Trailing underscore: naga_oil's cross-module identifier check rejects
    // an exported struct field whose name ends in a digit.
    _padding1_: f32,
    color: vec3<f32>,
    _padding2_: f32,
    ambient: vec3<f32>,
    _padding3_: f32,
};

struct HighlightingUniform {
    hovered_sticker_index: u32,
    hovered_piece_slot: u32,
    _padding: vec2<u32>,
    highlight_color: vec4<f32>,       // rgb = color, a = intensity
    piece_highlight_color: vec4<f32>, // rgb = color, a = intensity
};

// Which piece slot each `instances` entry currently occupies, indexed by
// instance index. Bindings are ordered by descending readership across the
// classic/elemental/particle pipelines; this is read by all three.
@group(0) @binding(3)
var<storage, read> piece_slots: array<u32>;

@group(0) @binding(4)
var<uniform> highlighting: HighlightingUniform;

@group(0) @binding(5)
var<uniform> light: LightUniform;
