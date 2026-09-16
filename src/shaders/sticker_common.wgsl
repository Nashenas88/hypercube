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
    // Facet counts (2..=4; 0 = unused) the first-run tutorial wants
    // pulse-highlighted this frame.
    tutorial_flash_facet_count_a: u32,
    tutorial_flash_facet_count_b: u32,
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

// Pulsing intensity (0 = off) for a facet whose piece has `facet_count`
// stickers, driving the first-run tutorial's flash-the-target-pieces
// highlight; `facet_count == 0` (the invisible center) never flashes.
fn tutorial_flash_pulse(facet_count: u32, elapsed_seconds: f32) -> f32 {
    if (facet_count != 0u &&
        (facet_count == highlighting.tutorial_flash_facet_count_a ||
         facet_count == highlighting.tutorial_flash_facet_count_b)) {
        return 0.5 + 0.5 * sin(elapsed_seconds * 4.0);
    }
    return 0.0;
}

// Sticker indices grouped into contiguous `(face_id, kind)` blocks, read only
// by the particle pipeline to look up a particle's owning sticker, since
// each kind has its own per-kind emission budget.
@group(0) @binding(6)
var<storage, read> sticker_order: array<u32>;
