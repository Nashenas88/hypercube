//! GPU rendering system for the 4D hypercube visualization.
//!
//! This module handles all graphics rendering using wgpu, including GPU resource management,
//! render pipeline setup, and per-frame rendering of the hypercube instances.

use core::f32;
use std::borrow::Cow;

use iced::wgpu::{self, CommandEncoder, Device, Queue, TextureFormat, TextureView};
use iced::widget::shader;
use iced::{Rectangle, Size};
use naga_oil::compose::{ComposableModuleDescriptor, Composer, NagaModuleDescriptor};
use wgpu::util::DeviceExt;

use crate::app::RenderMode;
use crate::camera::{Camera, CameraUniform, Projection};
use crate::geometry::{CUBE_VERTICES, VERTEX_NORMAL_INDICES};
use crate::math::BASE_STICKER_SIZE;
use crate::piece::{FACET_TABLE, Hypercube, StickerInstance, generate_sticker_instances};
use crate::shader_widget::UiControls;
use crate::theme::Theme;

/// Vertices in one billboard quad: two triangles, generated in the vertex
/// shader rather than read from a buffer.
const QUAD_VERTICES: u32 = 6;

/// Format of the offscreen scene target every pipeline but the final
/// composite renders into, and the two half-res bloom targets. Half-float
/// so a material can write emission above 1.0 for bloom to threshold
/// against, instead of clipping at the display's 8-bit range like a direct
/// render to `target` would.
const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Side length of the Fire ground-truth debug inset, as a fraction of
/// `bounds`' shorter side.
const GROUND_TRUTH_DEBUG_INSET_FRACTION: f32 = 0.2;

/// Sticker indices grouped into contiguous `(face_id, kind)` blocks, plus the
/// sub-range each block occupies.
///
/// `FACET_TABLE` groups instances into 8 contiguous per-`face_id` blocks of
/// `facets_per_face`, but `kind` is dynamic within a block: a move permutes
/// which piece, and so which kind, sits in a slot. This regroups a block's
/// instances by kind without touching which face they belong to.
struct StickerOrder {
    /// Every sticker index, covering each exactly once, ordered face-major
    /// then kind-major within each face.
    sorted: Vec<u32>,
    /// `ranges[face_id][kind]` is `sorted`'s sub-range for that group.
    ranges: [[std::ops::Range<u32>; 8]; 8],
}

fn build_sticker_order(instances: &[StickerInstance], facets_per_face: u32) -> StickerOrder {
    let mut sorted = Vec::with_capacity(instances.len());
    let mut ranges: [[std::ops::Range<u32>; 8]; 8] =
        std::array::from_fn(|_| std::array::from_fn(|_| 0..0u32));

    for face_id in 0..8u32 {
        let block_start = face_id * facets_per_face;
        for kind in 0..8u32 {
            let range_start = sorted.len() as u32;
            for offset in 0..facets_per_face {
                let index = block_start + offset;
                if instances[index as usize].kind == kind {
                    sorted.push(index);
                }
            }
            ranges[face_id as usize][kind as usize] = range_start..sorted.len() as u32;
        }
    }

    StickerOrder { sorted, ranges }
}

/// Expands a [`StickerOrder`]'s per-`(face_id, kind)` sticker groups into the
/// particle pipeline's actual instance buffer: `counts[kind]` copies of a
/// sticker's index for every sticker of that kind, so the vertex shader
/// recovers a particle's owning sticker with one indexed load instead of
/// dividing by a shared per-kind budget. Returns the expanded buffer contents
/// plus the `(face_id, kind)` sub-range each group occupies within it.
fn build_particle_instances(
    order: &StickerOrder,
    counts: &[u32; 8],
) -> (Vec<u32>, [[std::ops::Range<u32>; 8]; 8]) {
    let mut expanded = Vec::new();
    let mut ranges: [[std::ops::Range<u32>; 8]; 8] =
        std::array::from_fn(|_| std::array::from_fn(|_| 0..0u32));

    for (face_id, kind_ranges) in ranges.iter_mut().enumerate() {
        for (kind, range_slot) in kind_ranges.iter_mut().enumerate() {
            let sticker_range = order.ranges[face_id][kind].clone();
            let count = counts[kind];
            let start = expanded.len() as u32;
            if count > 0 {
                for &sticker_index in
                    &order.sorted[sticker_range.start as usize..sticker_range.end as usize]
                {
                    expanded.extend(std::iter::repeat_n(sticker_index, count as usize));
                }
            }
            *range_slot = start..expanded.len() as u32;
        }
    }

    (expanded, ranges)
}

/// Which pipeline/bind groups a [`DepthBatchGroup`] draws with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DepthBatchKind {
    Fire,
    Ice,
    Light,
    Dirt,
}

/// One instanced `draw_indexed` call within a depth batch's shared pass:
/// every sticker of `batch_remap[remap_offset..remap_offset + count]` shares
/// both `kind` and `face_id`, so they share `fire_pipeline`/`ice_pipeline`/
/// `light_pipeline`/`dirt_pipeline` (whichever `kind` picks) and `face_id`'s own winding-corrected index
/// chunk (see `calculate_indices`) - `compute_vertex_geometry` derives a
/// vertex's local cube face straight from the raw index buffer position, so
/// mixing chunks across a call would corrupt geometry/normals for whichever
/// instances didn't match. Built by `update_depth_batches`, which also
/// uploads the matching `depth_batch_remap_buffer` contents; consumed by
/// `render()`.
#[derive(Debug, Clone, Copy)]
struct DepthBatchGroup {
    kind: DepthBatchKind,
    face_id: u32,
    remap_offset: u32,
    count: u32,
}

/// Groups `depth_batches`' entries into `update_depth_batches`'s draw plan:
/// each batch's `Vec<DepthLayer>` becomes a `Vec<DepthBatchGroup>`,
/// partitioned by `(kind, face_id)` so same-kind, same-face stickers within
/// one batch collapse into a single instanced draw call. Returns that plan
/// alongside the flat sticker-index array `depth_batch_remap_buffer` needs
/// uploaded - each group's `remap_offset..remap_offset + count` is a slice
/// of it. `BTreeMap` only for deterministic (ascending `face_id`) group
/// ordering, not for performance - a batch holds at most a few dozen
/// stickers.
fn group_depth_batches(
    depth_batches: &[Vec<crate::shader_widget::DepthLayer>],
    facets_per_face: u32,
) -> (Vec<Vec<DepthBatchGroup>>, Vec<u32>) {
    use crate::shader_widget::DepthLayer;

    let mut flat_remap: Vec<u32> = Vec::new();
    let mut batches: Vec<Vec<DepthBatchGroup>> = Vec::with_capacity(depth_batches.len());

    for batch in depth_batches {
        let mut fire_by_face: std::collections::BTreeMap<u32, Vec<u32>> =
            std::collections::BTreeMap::new();
        let mut ice_by_face: std::collections::BTreeMap<u32, Vec<u32>> =
            std::collections::BTreeMap::new();
        let mut light_by_face: std::collections::BTreeMap<u32, Vec<u32>> =
            std::collections::BTreeMap::new();
        let mut dirt_by_face: std::collections::BTreeMap<u32, Vec<u32>> =
            std::collections::BTreeMap::new();
        for &layer in batch {
            let (by_face, instance_index) = match layer {
                DepthLayer::Fire(index) => (&mut fire_by_face, index),
                DepthLayer::Ice(index) => (&mut ice_by_face, index),
                DepthLayer::Light(index) => (&mut light_by_face, index),
                DepthLayer::Dirt(index) => (&mut dirt_by_face, index),
            };
            let face_id = instance_index / facets_per_face;
            by_face.entry(face_id).or_default().push(instance_index);
        }

        let mut groups = Vec::new();
        for (kind, by_face) in [
            (DepthBatchKind::Fire, fire_by_face),
            (DepthBatchKind::Ice, ice_by_face),
            (DepthBatchKind::Light, light_by_face),
            (DepthBatchKind::Dirt, dirt_by_face),
        ] {
            for (face_id, indices) in by_face {
                let remap_offset = flat_remap.len() as u32;
                let count = indices.len() as u32;
                flat_remap.extend(indices);
                groups.push(DepthBatchGroup {
                    kind,
                    face_id,
                    remap_offset,
                    count,
                });
            }
        }
        batches.push(groups);
    }

    (batches, flat_remap)
}

/// Copies `texture`'s full extent back to CPU memory as tightly-packed RGBA8
/// rows, converting from `format`'s BGRA channel order if that's what the
/// caller's surface turned out to be (the `image` crate, and every caller
/// here, wants RGBA8).
///
/// wgpu requires a texture-to-buffer copy's `bytes_per_row` to be a multiple
/// of `COPY_BYTES_PER_ROW_ALIGNMENT` (256), which a texture's actual row
/// width usually isn't, so the buffer holds each row padded out to that
/// alignment and this strips the padding back out on the way to `pixels`.
///
/// Blocks the calling thread on `device.poll` until the copy completes -
/// fine for the occasional caller `capture_frame` has, wrong for a hot path.
fn read_texture_rgba8(
    device: &Device,
    queue: &Queue,
    texture: &wgpu::Texture,
    format: wgpu::TextureFormat,
) -> Vec<u8> {
    let size = texture.size();
    let bytes_per_pixel = format
        .block_copy_size(None)
        .expect("readback format must be an uncompressed color format");

    let unpadded_bytes_per_row = size.width * bytes_per_pixel;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Texture Readback Buffer"),
        size: (padded_bytes_per_row * size.height) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(size.height),
            },
        },
        wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = readback_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device.poll failed while reading back a texture");
    rx.recv()
        .expect("map_async callback never fired")
        .expect("failed to map texture readback buffer");

    let mut pixels = Vec::with_capacity((unpadded_bytes_per_row * size.height) as usize);
    {
        let padded = slice.get_mapped_range();
        for row in 0..size.height {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + unpadded_bytes_per_row as usize;
            pixels.extend_from_slice(&padded[start..end]);
        }
    }
    readback_buffer.unmap();

    match format {
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
            for pixel in pixels.as_chunks_mut::<4>().0 {
                pixel.swap(0, 2);
            }
        }
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {}
        // Compositors that support HDR/wide-gamut output (confirmed via a
        // live run: iced_wgpu logged "Selected format: Rgb10a2Unorm") can
        // negotiate this as the window surface format instead of an 8-bit
        // one. It's still 4 bytes per pixel, but packed as one little-endian
        // u32 - Vulkan's `A2B10G10R10_UNORM_PACK32`, per wgpu-hal's format
        // table - rather than four independent one-byte channels, so it
        // needs unpacking rather than a byte reorder.
        wgpu::TextureFormat::Rgb10a2Unorm => {
            let scale10 = |bits: u32| ((bits * 255 + 511) / 1023) as u8;
            for pixel in pixels.as_chunks_mut::<4>().0 {
                let packed = u32::from_le_bytes(*pixel);
                let r = scale10(packed & 0x3ff);
                let g = scale10((packed >> 10) & 0x3ff);
                let b = scale10((packed >> 20) & 0x3ff);
                let a = (((packed >> 30) & 0x3) * 255 / 3) as u8;
                *pixel = [r, g, b, a];
            }
        }
        other => log::warn!(
            "read_texture_rgba8: unrecognized 4-byte-per-pixel format {other:?}; \
             assuming byte order R,G,B,A - colors may come out wrong"
        ),
    }

    pixels
}

/// GPU renderer for the hypercube visualization.
///
/// Manages all graphics resources including buffers, textures, pipelines, and rendering state.
/// Uses instanced rendering to efficiently draw all 216 hypercube stickers.
#[derive(Debug)]
pub(crate) struct Renderer {
    /// Bounds within the viewport to render to.
    bounds: Rectangle<f32>,
    /// Format `composite`'s target must be, since the pipelines it draws
    /// with (`composite_pipeline`/`composite_tonemapped_pipeline`) were
    /// created against it and a wgpu render pipeline's target format is
    /// fixed at creation time. Needed by `capture_frame`, which draws into a
    /// scratch texture of its own rather than iced's surface.
    target_format: TextureFormat,
    /// Vertex buffer for sky quad
    sky_vertex_buffer: wgpu::Buffer,
    /// Index buffer for sky quad
    sky_index_buffer: wgpu::Buffer,
    /// Graphics pipeline for sky rendering
    sky_pipeline: wgpu::RenderPipeline,
    /// Graphics pipeline for standard rendering
    classic_pipeline: wgpu::RenderPipeline,
    /// Graphics pipeline for the Elemental theme's sticker materials
    elemental_pipeline: wgpu::RenderPipeline,
    /// Graphics pipeline for the Elemental theme's Fire stickers, blended
    /// rather than opaque
    fire_pipeline: wgpu::RenderPipeline,
    /// Graphics pipeline for the Elemental theme's Ice stickers. Opaque and
    /// depth-writing, unlike Fire, but drawn one instance and one depth
    /// layer at a time (see `render()`) since each layer reads back a
    /// snapshot of the scene so far as its refraction/reflection background.
    ice_pipeline: wgpu::RenderPipeline,
    /// Graphics pipeline for the Elemental theme's Light stickers, blended
    /// rather than opaque - like Fire, but with its own raymarched
    /// volumetric cloud material instead of a flame.
    light_pipeline: wgpu::RenderPipeline,
    /// Graphics pipeline for the Elemental theme's Dirt stickers, blended
    /// rather than opaque - like Fire and Light, but with its own
    /// raymarched bumpy rock material and a hard SDF hit/miss (coverage
    /// 0.0 or 1.0) rather than a density accumulation.
    dirt_pipeline: wgpu::RenderPipeline,
    /// Depth-only prepass pipelines for Fire/Light/Dirt, drawn front-to-back
    /// before their color pipelines above so hardware early-Z can reject
    /// fragments a nearer sticker already covers - see `render()`'s "Depth
    /// Prepass" pass and `perf_improvements.md` item 1. Ice needs no
    /// equivalent: `ice_pipeline` already writes real depth itself.
    fire_depth_prepass_pipeline: wgpu::RenderPipeline,
    light_depth_prepass_pipeline: wgpu::RenderPipeline,
    dirt_depth_prepass_pipeline: wgpu::RenderPipeline,
    /// Graphics pipeline for the Elemental theme's billboarded particles
    particle_pipeline: wgpu::RenderPipeline,
    /// Graphics pipeline for normal visualization
    normal_pipeline: wgpu::RenderPipeline,
    /// Graphics pipeline for depth visualization
    depth_pipeline: wgpu::RenderPipeline,
    /// Graphics pipeline for debug AABB rendering
    debug_pipeline: wgpu::RenderPipeline,
    /// Graphics pipeline for the 4D-rotation-axis gizmo ring/marker, drawn
    /// into `scene_view` right after the opaque hypercube pass and before
    /// the translucent Fire/Ice/Light/Dirt batches - see `render`.
    gizmo_pipeline: wgpu::RenderPipeline,
    /// Bright-pass, blur H, blur V and composite pipelines, in that order:
    /// the post-processing chain that turns `scene_view` (plus, under
    /// `Theme::Elemental`, a blurred `bloom_view_a`) into the final frame in
    /// `target`. Debug AABBs are drawn after compositing, straight into
    /// `target`, so they're excluded from bloom.
    bright_pass_pipeline: wgpu::RenderPipeline,
    blur_h_pipeline: wgpu::RenderPipeline,
    blur_v_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    /// The composite variant `Theme::Elemental` uses instead, applying an
    /// ACES tonemap on the way out so the emission its materials write above
    /// 1.0 keeps its shape rather than clipping flat at the surface format's
    /// range.
    composite_tonemapped_pipeline: wgpu::RenderPipeline,
    /// Current rendering mode
    current_render_mode: RenderMode,
    /// Currently selected sticker theme
    current_theme: Theme,
    /// When set, draws every cell of this `face_id` - not just Fire ones -
    /// opaque and depth-tested, in a small inset in the corner of the main
    /// scene, using the exact same live camera/instance/transform data the
    /// main pass used. Reusing those buffers rather than recomputing
    /// anything means the inset can never drift from what `depth_draw_order`'s
    /// topological sort actually scored - the same transform decides the
    /// geometry either way. `shader_widget.rs` cycles this through whichever
    /// faces currently hold a Fire sticker.
    current_ground_truth_debug_face: Option<u32>,
    /// Buffer containing cube vertex positions
    vertex_buffer: wgpu::Buffer,
    /// Number of stickers (each generates 36 vertices)
    num_stickers: usize,
    /// GPU buffer containing per-sticker instance data (position, color, face_id)
    instance_buffer: wgpu::Buffer,
    /// Index buffers for each 4D face
    face_index_buffer: wgpu::Buffer,
    /// Generation of the indices last uploaded to `face_index_buffer`, so
    /// `update_indices` can skip re-uploading unchanged data.
    last_indices_generation: Option<u64>,
    /// Generation of the sticker instances last uploaded to
    /// `instance_buffer`, so `update_sticker_instances` can skip
    /// re-uploading unchanged data.
    last_sticker_generation: Option<u64>,
    /// Backs `vs_main_batched`'s `depth_batch_remap` binding: a flat run of
    /// sticker indices per `(kind, face_id)` group across every depth
    /// batch, written fresh each frame by `update_depth_batches`. Sized to
    /// `num_stickers` u32 slots - the most Fire+Ice participants any single
    /// frame can have - so it never needs resizing.
    depth_batch_remap_buffer: wgpu::Buffer,
    /// This frame's Fire/Ice draw plan, one `Vec<DepthBatchGroup>` per
    /// batch in back-to-front order; populated by `update_depth_batches`
    /// and consumed by `render()`. See `DepthBatchGroup`.
    depth_batch_groups: Vec<Vec<DepthBatchGroup>>,
    /// Particle pipeline's indirection buffer: sticker indices grouped into
    /// contiguous `(face_id, kind)` blocks, rebuilt by
    /// `update_sticker_instances` alongside `instance_buffer` whenever
    /// `sticker_generation` changes.
    sticker_order_buffer: wgpu::Buffer,
    /// `particle_ranges[face_id][kind]` is `sticker_order_buffer`'s
    /// instance-index sub-range for that group, used by `render()` to issue
    /// one particle draw per visible face and populated kind.
    particle_ranges: [[std::ops::Range<u32>; 8]; 8],
    /// CPU-side camera uniform data
    camera_uniform: CameraUniform,
    /// GPU buffer containing camera matrices
    camera_buffer: wgpu::Buffer,
    /// CPU-side highlighting uniform data
    highlighting_uniform: HighlightingUniform,
    /// GPU buffer containing highlighting data
    highlighting_buffer: wgpu::Buffer,
    /// CPU-side lighting uniform data
    light_uniform: LightUniform,
    /// GPU buffer containing lighting data
    light_buffer: wgpu::Buffer,
    /// GPU buffer for debug instance data (vertex attributes)
    debug_instance_buffer: wgpu::Buffer,
    /// Reused across frames by `update_debug_instances` to avoid allocating
    /// a fresh `Vec` every frame for what's usually empty.
    debug_scratch: Vec<DebugInstance>,
    /// GPU buffer for the rotation-axis gizmo's per-vertex position+color
    /// data - a plain (non-instanced) vertex buffer, fixed at
    /// `GIZMO_VERTEX_CAPACITY`.
    gizmo_vertex_buffer: wgpu::Buffer,
    /// Reused across frames by `update_gizmo`, mirroring `debug_scratch`.
    gizmo_scratch: Vec<GizmoVertex>,
    /// Vertex count `update_gizmo` last uploaded - `render`'s own gizmo pass
    /// draws this many vertices (0 draws nothing) since it has no other way
    /// to learn how much of the fixed-capacity buffer is live this frame.
    gizmo_vertex_count: u32,
    /// Bind group for main shader (transform, camera, light, normals, instances)
    main_bind_group: wgpu::BindGroup,
    /// Bind group for normal shader (transform, camera, normals, instances)
    normal_bind_group: wgpu::BindGroup,
    /// Bind group for debug shaders (transform, camera, instances)
    debug_bind_group: wgpu::BindGroup,
    /// Bind group for debug AABB rendering (camera, debug_instances)
    debug_aabb_bind_group: wgpu::BindGroup,
    /// Bind group for the gizmo pipeline (camera + light - geometry is
    /// already fully resolved to 3D positions on the CPU, but shading still
    /// needs the scene's directional light).
    gizmo_bind_group: wgpu::BindGroup,
    /// Depth texture for z-buffering
    depth_texture: wgpu::Texture,
    /// Depth texture view for rendering
    depth_view: wgpu::TextureView,
    /// Dedicated depth buffer for the Fire ground-truth debug inset, kept
    /// separate from `depth_texture` so clearing it can't disturb the main
    /// scene's depth values that `render_debug_aabb` reads back afterward.
    debug_inset_depth_texture: wgpu::Texture,
    debug_inset_depth_view: wgpu::TextureView,
    /// Offscreen HDR target every pipeline but the final composite renders
    /// into, resized alongside `depth_texture`.
    scene_texture: wgpu::Texture,
    scene_view: wgpu::TextureView,
    /// Half-res HDR ping-pong pair for the separable bloom blur. The blur
    /// always leaves its result in `bloom_texture_a`: bright-pass writes it,
    /// blur H reads it into `bloom_texture_b`, blur V reads that back.
    bloom_texture_a: wgpu::Texture,
    bloom_view_a: wgpu::TextureView,
    bloom_texture_b: wgpu::Texture,
    bloom_view_b: wgpu::TextureView,
    /// Layouts for the post-process bind groups below, kept to rebuild them
    /// in `resize` whenever the views they reference are replaced.
    post_process_bind_group_layout: wgpu::BindGroupLayout,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    /// Shared by every post-process pipeline's texture input.
    post_process_sampler: wgpu::Sampler,
    bright_pass_bind_group: wgpu::BindGroup,
    blur_h_bind_group: wgpu::BindGroup,
    blur_v_bind_group: wgpu::BindGroup,
    composite_bind_group: wgpu::BindGroup,
    /// Gray noise texture Ice's triplanar bump-mapping samples as
    /// `iChannel0`, loaded once from `src/resources/ice_noise_64.png` (see
    /// `src/bin/generate_ice_noise.rs`). The `Texture` is otherwise unused
    /// after creation, kept alive only because `ice_noise_view` borrows it.
    _ice_noise_texture: wgpu::Texture,
    ice_noise_view: wgpu::TextureView,
    ice_noise_sampler: wgpu::Sampler,
    /// Snapshot of `scene_texture`'s contents ("iChannel1") each Ice depth
    /// layer's pass reads as its refraction/reflection background; see
    /// `render()`. Resized alongside `scene_texture`.
    ice_bg_texture: wgpu::Texture,
    ice_bg_view: wgpu::TextureView,
    /// Layout for `ice_bind_group`, kept to rebuild it in `resize` whenever
    /// `ice_bg_view` is replaced.
    ice_bind_group_layout: wgpu::BindGroupLayout,
    ice_bind_group: wgpu::BindGroup,
    /// Backs `ice_bind_group`'s binding 4, rewritten every frame in
    /// `update_camera`. Never needs recreating in `resize` (only its
    /// contents change), unlike the rest of `ice_bind_group`'s resources.
    ice_background_transform_buffer: wgpu::Buffer,
    /// Transform uniform buffer for vertex shaders
    transform_buffer: wgpu::Buffer,
    /// Skybox bind group
    skybox_bind_group: wgpu::BindGroup,
}

/// Transform data passed to compute shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Transform4D {
    /// 4D rotation matrix
    rotation_matrix: [[f32; 4]; 4],
    /// Distance of viewer from W=0 plane
    viewer_distance: f32,
    /// Scale of individual stickers
    sticker_scale: f32,
    /// 3D distance to push each face outward from the tesseract, applied
    /// after 4D-to-3D projection
    face_gap: f32,
    /// Slider value (1.0 = no push) that `(face_gap_4d - 1.0)` scales into
    /// the magnitude of a depth-preserving push applied to each facet's
    /// rotated face-normal direction, added after rotation but before
    /// projection (see `math::depth_preserving_push`)
    face_gap_4d: f32,
    _padding: [f32; 3],
    /// Wall-clock seconds since the app started, wrapped modulo 3600.
    elapsed_seconds: f32,
}

/// Lighting uniform data
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LightUniform {
    /// Direction of the light (normalized)
    direction: [f32; 3],
    _padding1: f32,
    /// Color of the light
    color: [f32; 3],
    _padding2: f32,
    /// Ambient light color
    ambient: [f32; 3],
    _padding3: f32,
}

/// Maps NDC (from `camera.view_proj`) into `ice_bg_texture`'s texture-space
/// UV, for `elemental_shader.wgsl`'s `ice_sample_background`. `scene_texture`/
/// `ice_bg_texture` are sized to the full window, but the 3D scene only
/// occupies `Renderer::bounds` within it (a sub-rectangle - the shader
/// widget's viewport, e.g. below the menu bar) - treating NDC as spanning the
/// *whole* texture would sample outside that sub-rectangle, into pixels
/// `render()` never draws. Recomputed every frame in `update_camera`, since
/// either `bounds` or the texture's size can change between frames.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct IceBackgroundTransform {
    /// `bounds.{width,height}` as a fraction of the full texture's size.
    scale: [f32; 2],
    /// `bounds.{x,y}` as a fraction of the full texture's size.
    offset: [f32; 2],
}

/// Highlighting uniform data for sticker hover effects
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct HighlightingUniform {
    /// Index of the hovered sticker (u32::MAX if none)
    hovered_sticker_index: u32,
    /// `Hypercube::pieces` slot of the hovered sticker's piece (u32::MAX if none)
    hovered_piece_slot: u32,
    /// Facet counts (2..=4; `0` = unused) the first-run tutorial wants
    /// pulse-highlighted this frame.
    tutorial_flash_facet_count_a: u32,
    tutorial_flash_facet_count_b: u32,
    /// Color and intensity (in `a`) for the exact hovered sticker
    highlight_color: [f32; 4],
    /// Color and intensity (in `a`) for the rest of the hovered piece's stickers
    piece_highlight_color: [f32; 4],
}

/// Classic theme's per-kind RGBA colors, indexed by kind.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct KindColors {
    colors: [[f32; 4]; 8],
}

const DEFAULT_KIND_COLORS: KindColors = KindColors {
    colors: [
        [0.0, 1.0, 1.0, 1.0],  // 0: center, Cyan
        [0.0, 1.0, 0.0, 1.0],  // 1: left, Green
        [1.0, 1.0, 0.0, 1.0],  // 2: bottom, Yellow
        [1.0, 0.0, 0.0, 1.0],  // 3: front, Red
        [1.0, 0.65, 0.0, 1.0], // 4: back, Orange
        [1.0, 1.0, 1.0, 1.0],  // 5: top, White
        [0.1, 0.1, 1.0, 1.0],  // 6: right, Blue
        [0.5, 0.0, 1.0, 1.0],  // 7: void, Purple
    ],
};

/// Debug instance data for GPU vertex attributes (transparent bounding box rendering)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct DebugInstance {
    /// Transform matrix for AABB positioning and scaling (4x4 matrix)
    transform: [[f32; 4]; 4],
    /// RGBA color for this AABB
    color: [f32; 4],
}

/// CPU-side debug instance with distance for sorting
#[derive(Copy, Clone, Debug)]
pub(crate) struct DebugInstanceWithDistance {
    /// GPU data that will be uploaded to vertex buffer
    pub gpu_data: DebugInstance,
    /// Distance from camera (for back-to-front sorting)
    pub distance: f32,
}

/// Fixed capacity for `gizmo_vertex_buffer`, sized comfortably above the
/// worst case `shader_widget::gizmo_torus_vertices` can emit for a single
/// call (a focus animation or a Shift+drag renders exactly one ring, never
/// both at once): a segmented main-ring torus (48 major x 8 minor x 6
/// vertices = 2304) plus four field-loop marker toruses (12 major x 6 minor
/// x 6 vertices x 4 markers = 1728) plus their arrow cones (3 arrows x 4
/// markers x 18 vertices = 216), for a worst case of 4248 - so `update_gizmo`
/// never needs per-frame buffer-growth logic.
const GIZMO_VERTEX_CAPACITY: usize = 8192;

/// Per-vertex data for the 4D-rotation-axis gizmo: CPU-projected 3D
/// positions (see `shader_widget::gizmo_ring_vertices`) with a per-vertex
/// flat face normal (see `shader_widget::recompute_flat_normals`) and
/// color. Unlike `DebugInstance`, no per-instance transform is needed - the
/// geometry is already fully resolved to 3D positions on the CPU each
/// frame - but color must vary *within* one draw call (ring vs. marker,
/// horizontal vs. vertical axis), so it travels as a vertex attribute
/// rather than a per-instance storage entry. The normal is likewise
/// per-vertex rather than derived in the shader, since the geometry has no
/// index buffer for a fragment-shader derivative-based approach to use.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GizmoVertex {
    pub(crate) position: [f32; 3],
    pub(crate) normal: [f32; 3],
    pub(crate) color: [f32; 4],
}

impl DebugInstanceWithDistance {
    /// Create a new debug instance for an AABB
    pub fn new(
        min: [f32; 3],
        max: [f32; 3],
        color: [f32; 4],
        camera_pos: [f32; 3],
        scale: f32,
    ) -> Self {
        // Calculate center and size
        let center = [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ];
        let size = [
            (max[0] - min[0]) * 0.5 * scale,
            (max[1] - min[1]) * 0.5 * scale,
            (max[2] - min[2]) * 0.5 * scale,
        ];

        // Create transform matrix: scale then translate
        let transform = [
            [size[0], 0.0, 0.0, 0.0],
            [0.0, size[1], 0.0, 0.0],
            [0.0, 0.0, size[2], 0.0],
            [center[0], center[1], center[2], 1.0],
        ];

        // Calculate distance from camera for sorting
        let dx = center[0] - camera_pos[0];
        let dy = center[1] - camera_pos[1];
        let dz = center[2] - camera_pos[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

        let gpu_data = DebugInstance { transform, color };

        Self { gpu_data, distance }
    }
}

/// Loads a cross-format cubemap and creates a GPU texture.
///
/// The cross format is arranged as:
/// ```ignore
///     +Y
/// -X  +Z  +X  -Z
///     -Y
/// ```
///
/// # Arguments
/// * `device` - GPU device for texture creation
/// * `queue` - GPU queue for data upload
/// * `image_path` - Path to the cross-format cubemap image
///
/// # Returns
/// A tuple containing (texture, view, sampler, bind_group)
fn load_cross_cubemap(
    device: &Device,
    queue: &Queue,
    image_path: &str,
) -> Result<(wgpu::Texture, wgpu::TextureView, wgpu::Sampler), Box<dyn std::error::Error>> {
    // Load the image
    let image_bytes = std::fs::read(image_path)?;
    let image = image::load_from_memory(&image_bytes)?.to_rgba8();
    let (img_width, img_height) = image.dimensions();

    // Validate dimensions - should be 2:3 aspect ratio for cross format (width:height = 4:3)
    if img_width * 3 != img_height * 4 {
        return Err("Invalid cross cubemap dimensions. Expected 4:3 aspect ratio.".into());
    }

    // Calculate face size (each face should be square)
    let face_size = img_width / 4;
    if face_size * 3 != img_height {
        return Err("Invalid cross cubemap face dimensions.".into());
    }

    // Create the cubemap texture
    let cubemap_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Skybox Cubemap"),
        size: wgpu::Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: 6, // 6 faces
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Extract and upload each face
    // Cross layout mapping: +X, -X, +Y, -Y, +Z, -Z
    let face_positions = [
        (face_size * 2, face_size), // +X (right)
        (0, face_size),             // -X (left)
        (face_size, 0),             // +Y (top)
        (face_size, face_size * 2), // -Y (bottom)
        (face_size, face_size),     // +Z (front)
        (face_size * 3, face_size), // -Z (back)
    ];

    for (face_index, &(x_offset, y_offset)) in face_positions.iter().enumerate() {
        let mut face_data = Vec::new();

        for y in 0..face_size {
            for x in 0..face_size {
                let pixel_x = x_offset + x;
                let pixel_y = y_offset + y;
                let pixel_index = ((pixel_y * img_width + pixel_x) * 4) as usize;

                // Copy RGBA data
                face_data.extend_from_slice(&image.as_raw()[pixel_index..pixel_index + 4]);
            }
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &cubemap_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: face_index as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &face_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(face_size * 4),
                rows_per_image: Some(face_size),
            },
            wgpu::Extent3d {
                width: face_size,
                height: face_size,
                depth_or_array_layers: 1,
            },
        );
    }

    // Create texture view
    let view = cubemap_texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("Skybox View"),
        format: None,
        dimension: Some(wgpu::TextureViewDimension::Cube),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: 0,
        array_layer_count: Some(6),
        usage: None,
    });

    // Create sampler
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Skybox Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    Ok((cubemap_texture, view, sampler))
}

/// Creates one `Depth32Float` depth-stencil attachment, sized to the full
/// viewport regardless of what fraction of it a pass restricts its viewport
/// to - a render pass's attachments must all share one size, even when a
/// `set_viewport` call inside it only touches part of them.
fn create_depth_texture(
    device: &Device,
    label: &str,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Creates one `HDR_FORMAT` render target usable both as a pass's color
/// attachment and as a later pass's sampled input. `extra_usage` adds any
/// further capability a particular target needs beyond that - `COPY_SRC` for
/// `scene_texture`, which `render()` copies from into `ice_bg_texture`
/// before each Ice depth layer's pass.
fn create_hdr_target(
    device: &Device,
    label: &str,
    width: u32,
    height: u32,
    extra_usage: wgpu::TextureUsages,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | extra_usage,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Loads a small grayscale PNG as a `Rgba8Unorm` 2D texture, sampled nearest
/// and repeat-wrapped. Used for the Ice material's `iChannel0` bump-map
/// noise (`src/resources/ice_noise_64.png`, produced offline by
/// `src/bin/generate_ice_noise.rs`) - the shader's own `smoothSampling`
/// reconstructs its bicubic-ish smoothing from these unfiltered texel reads,
/// so the sampler itself stays unfiltered.
fn load_gray_noise_texture(
    device: &Device,
    queue: &Queue,
    image_path: &str,
) -> Result<(wgpu::Texture, wgpu::TextureView, wgpu::Sampler), Box<dyn std::error::Error>> {
    let image_bytes = std::fs::read(image_path)?;
    let image = image::load_from_memory(&image_bytes)?.to_rgba8();
    let (width, height) = image.dimensions();

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Ice Noise Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        image.as_raw(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Ice Noise Sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    Ok((texture, view, sampler))
}

/// Builds Ice's `@group(1)` bind group (iChannel0 noise, iChannel1
/// background, and the NDC-to-background-UV transform), needed both at
/// creation and whenever `resize` replaces `ice_bg_view`.
#[allow(clippy::too_many_arguments)]
fn create_ice_bind_group(
    device: &Device,
    layout: &wgpu::BindGroupLayout,
    noise_view: &wgpu::TextureView,
    noise_sampler: &wgpu::Sampler,
    background_view: &wgpu::TextureView,
    background_sampler: &wgpu::Sampler,
    background_transform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Ice Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(noise_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(noise_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(background_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(background_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: background_transform_buffer.as_entire_binding(),
            },
        ],
    })
}

/// Rebuilds the post-process pipelines' bind groups against the current
/// scene/bloom views, needed both at creation and whenever `resize` replaces
/// those views.
#[allow(clippy::too_many_arguments)]
fn create_post_process_bind_groups(
    device: &Device,
    post_process_bind_group_layout: &wgpu::BindGroupLayout,
    composite_bind_group_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    scene_view: &wgpu::TextureView,
    bloom_view_a: &wgpu::TextureView,
    bloom_view_b: &wgpu::TextureView,
) -> (
    wgpu::BindGroup,
    wgpu::BindGroup,
    wgpu::BindGroup,
    wgpu::BindGroup,
) {
    let single_texture_bind_group = |label: &str, view: &wgpu::TextureView| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: post_process_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    };

    // Bright-pass reads the full-res scene; the two blur passes ping-pong
    // between the half-res bloom textures.
    let bright_pass_bind_group = single_texture_bind_group("Bright Pass Bind Group", scene_view);
    let blur_h_bind_group = single_texture_bind_group("Blur H Bind Group", bloom_view_a);
    let blur_v_bind_group = single_texture_bind_group("Blur V Bind Group", bloom_view_b);

    // The blur ping-pong always leaves its result in `bloom_view_a`: bright-
    // pass writes it, blur H reads it into `bloom_view_b`, blur V reads that
    // back into `bloom_view_a`.
    let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Composite Bind Group"),
        layout: composite_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(scene_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(bloom_view_a),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });

    (
        bright_pass_bind_group,
        blur_h_bind_group,
        blur_v_bind_group,
        composite_bind_group,
    )
}

/// Runs one post-process pipeline over its bind group's input, drawing the
/// fullscreen triangle `vs_fullscreen` generates into `target`. `viewport`
/// restricts the draw to a sub-rectangle of `target` (used only by the
/// composite pass, to `Renderer::bounds`); the other passes always cover
/// their entire (always freshly-sized) target, so need none.
fn draw_fullscreen_pass(
    encoder: &mut CommandEncoder,
    label: &str,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    target: &TextureView,
    viewport: Option<Rectangle<f32>>,
) {
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            ops: wgpu::Operations {
                // The fullscreen triangle always covers every pixel this
                // pass's viewport reaches, so prior contents never matter.
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    if let Some(bounds) = viewport {
        render_pass.set_viewport(bounds.x, bounds.y, bounds.width, bounds.height, 0.0, 1.0);
    }
    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, bind_group, &[]);
    render_pass.draw(0..3, 0..1);
}

/// Clears `view` to black - used in place of the bloom passes when they're
/// skipped, so `composite` adds in nothing rather than a stale bloom result
/// left over from an earlier `Theme::Elemental` frame.
fn clear_texture(encoder: &mut CommandEncoder, label: &str, view: &TextureView) {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
}

impl Renderer {
    /// Creates a new renderer with initialized GPU resources.
    ///
    /// Sets up the complete rendering pipeline including device, surface, buffers,
    /// and render pipeline for hypercube visualization.
    ///
    /// # Arguments
    /// * `window` - Window to render into
    /// * `hypercube` - Initial hypercube data for setting up instance buffer
    ///
    /// # Returns
    /// A fully initialized renderer ready for frame rendering
    pub(crate) fn new(
        device: &Device,
        queue: &Queue,
        format: TextureFormat,
        bounds: Rectangle<f32>,
        viewport_size: Size<u32>,
        ui_controls: UiControls,
    ) -> Self {
        let camera_uniform = CameraUniform::new();

        // Initial seed direction; overwritten every frame by `Renderer::update_light`
        // once a camera-attached light direction is available.
        let light_dir = nalgebra::Vector3::new(0.5, -1.0, 0.3).normalize();
        let light_uniform = LightUniform {
            direction: [light_dir.x, light_dir.y, light_dir.z],
            _padding1: 0.0,
            color: [1.0, 0.95, 0.8], // Warm sunlight color
            _padding2: 0.0,
            ambient: [0.1, 0.1, 0.15], // Cool ambient light
            _padding3: 0.0,
        };

        let (depth_texture, depth_view) = create_depth_texture(
            device,
            "Depth Texture",
            viewport_size.width,
            viewport_size.height,
        );

        // A second depth buffer dedicated to the Fire ground-truth debug
        // inset, so clearing it for that pass can never disturb `depth_view`
        // - which `render_debug_aabb` reads back after `composite()`, long
        // after the inset pass has run.
        let (debug_inset_depth_texture, debug_inset_depth_view) = create_depth_texture(
            device,
            "Fire Ground Truth Debug Inset Depth Texture",
            viewport_size.width,
            viewport_size.height,
        );

        // Offscreen HDR scene target every pipeline but the final composite
        // renders into, plus a half-res ping-pong pair for the separable
        // bloom blur. Sized and resized alongside `depth_texture`.
        let (scene_texture, scene_view) = create_hdr_target(
            device,
            "Scene Texture",
            viewport_size.width,
            viewport_size.height,
            wgpu::TextureUsages::COPY_SRC,
        );
        let bloom_width = (viewport_size.width / 2).max(1);
        let bloom_height = (viewport_size.height / 2).max(1);
        let (bloom_texture_a, bloom_view_a) = create_hdr_target(
            device,
            "Bloom Texture A",
            bloom_width,
            bloom_height,
            wgpu::TextureUsages::empty(),
        );
        let (bloom_texture_b, bloom_view_b) = create_hdr_target(
            device,
            "Bloom Texture B",
            bloom_width,
            bloom_height,
            wgpu::TextureUsages::empty(),
        );
        // Snapshot of `scene_view`'s contents each Ice depth layer's pass
        // reads as its background ("iChannel1"): `render()` copies
        // `scene_texture` into this every layer, right before that layer's
        // own pass, so a nearer layer's snapshot already contains every
        // farther one's own refracted/reflected result.
        let (ice_bg_texture, ice_bg_view) = create_hdr_target(
            device,
            "Ice Background Texture",
            viewport_size.width,
            viewport_size.height,
            wgpu::TextureUsages::COPY_DST,
        );

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Buffer"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create initial highlighting uniform (no sticker highlighted)
        let highlighting_uniform = HighlightingUniform {
            hovered_sticker_index: u32::MAX, // No sticker highlighted
            hovered_piece_slot: u32::MAX,    // No piece highlighted
            tutorial_flash_facet_count_a: 0,
            tutorial_flash_facet_count_b: 0,
            highlight_color: [1.0, 1.0, 0.0, 0.3], // Yellow, 30% intensity
            piece_highlight_color: [0.2, 0.2, 0.2, 0.6], // Gray, 60% intensity
        };

        let highlighting_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Highlighting Buffer"),
            contents: bytemuck::cast_slice(&[highlighting_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sticker_instances = generate_sticker_instances(&Hypercube::solved());
        let num_stickers = sticker_instances.len();

        // Create instance buffer for sticker data
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&sticker_instances),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Backs `vs_main_batched`'s `depth_batch_remap` binding - see
        // `DepthBatchGroup`. Sized to `num_stickers` u32 slots, the most
        // Fire+Ice participants any single frame could ever populate;
        // `update_depth_batches` overwrites only the prefix it uses each
        // frame, so uninitialized content past that is never read.
        let depth_batch_remap_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Depth Batch Remap Buffer"),
            size: (num_stickers * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Static mapping from sticker instance index to owning piece slot,
        // for piece-level hover highlighting. Unlike `instance_buffer`, this
        // never changes, so it's uploaded once with no `COPY_DST` and isn't
        // kept as a `Renderer` field — `main_bind_group` holds the only
        // reference it needs after creation.
        let piece_slots: Vec<u32> = FACET_TABLE.iter().map(|f| f.piece_slot as u32).collect();
        let piece_slot_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Piece Slot Buffer"),
            contents: bytemuck::cast_slice(&piece_slots),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let kind_colors_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Kind Colors Buffer"),
            contents: bytemuck::cast_slice(&[DEFAULT_KIND_COLORS]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Particle pipeline's indirection buffer (see `build_particle_instances`).
        // Only `Theme::Elemental` ever draws particles (`render()`'s gate), so
        // it's sized and built for that theme's budget regardless of
        // `current_theme` — a kind's kept sticker count is invariant to
        // scrambling (moves permute which piece holds a kind, never how many
        // stickers have it), so this length never changes across generations,
        // only the order of indices within it.
        let facets_per_face = (num_stickers / 8) as u32;
        let initial_sticker_order = build_sticker_order(&sticker_instances, facets_per_face);
        let (mut initial_particle_instances, initial_particle_ranges) = build_particle_instances(
            &initial_sticker_order,
            &Theme::Elemental.particles_per_kind(),
        );
        // Every kind's Elemental particle budget is zero now that Dirt has
        // joined Fire/Water/Lightning/Ice/Dark/Light/Moss in carrying its
        // whole look on the surface, so `initial_particle_instances` is
        // always empty - `create_buffer_init` can't produce a bindable
        // buffer from zero-length `contents`, so this pads it the same way
        // `debug_instance_buffer` below does. Never read: every
        // `particle_ranges` entry stays an empty range while every budget
        // is zero.
        if initial_particle_instances.is_empty() {
            initial_particle_instances.push(0);
        }
        let sticker_order_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sticker Order Buffer"),
            contents: bytemuck::cast_slice(&initial_particle_instances),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Create debug instance buffer for transparent AABB rendering
        // Initialize with dummy instances to avoid zero-size buffer
        let dummy_instance = DebugInstance {
            transform: [[0.0; 4]; 4],    // Zero matrix (won't be visible)
            color: [0.0, 0.0, 0.0, 0.0], // Transparent
        };
        let debug_instances = vec![dummy_instance; 50]; // 30 dummy elements
        let debug_instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Debug Instance Buffer"),
            contents: bytemuck::cast_slice(&debug_instances),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Fixed-capacity vertex buffer for the rotation-axis gizmo, sized
        // for the worst case up front so `update_gizmo` never needs to grow
        // it - mirrors `debug_instance_buffer`'s dummy-init pattern above.
        let gizmo_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gizmo Vertex Buffer"),
            size: (GIZMO_VERTEX_CAPACITY * std::mem::size_of::<GizmoVertex>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut vertices = CUBE_VERTICES;
        vertices
            .iter_mut()
            .for_each(|v| v.iter_mut().for_each(|i| *i *= BASE_STICKER_SIZE));
        // Create vertex buffer for cube geometry
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let indices = VERTEX_NORMAL_INDICES
            .into_iter()
            .cycle()
            .take(VERTEX_NORMAL_INDICES.len() * 8)
            .collect::<Vec<_>>();
        let face_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Face Index Buffer"),
            contents: bytemuck::cast_slice(indices.as_slice()),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        // Create skybox bind group layout
        let skybox_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("Skybox Bind Group Layout"),
            });

        // Main shader bind group layout, ordered by descending readership
        // across the classic/elemental/particle pipelines: transform,
        // camera, instances, piece_slots, highlighting, light,
        // sticker_order (particle pipeline only), kind_colors,
        // depth_batch_remap (fire_pipeline/ice_pipeline's `vs_main_batched`
        // only - see `DepthBatchGroup`).
        let main_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("Main Bind Group Layout"),
            });

        // Ice's own bind group (group 1 alongside `main_bind_group_layout`
        // at group 0, which has no free texture slots): iChannel0 (gray
        // noise) and iChannel1 (the scene-so-far snapshot `render()` copies
        // into `ice_bg_texture` before each depth layer).
        let ice_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("Ice Bind Group Layout"),
            });

        // Normal shader bind group layout (transform, camera, instances)
        let normal_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("Normal Bind Group Layout"),
            });

        // Debug shaders bind group layout (transform, camera, instances)
        let debug_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("Debug Bind Group Layout"),
            });

        // Debug AABB bind group layout (camera, debug_instances)
        let debug_aabb_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("Debug AABB Bind Group Layout"),
            });

        // Gizmo bind group layout (camera + light) - the rotation-axis
        // gizmo's ring/arrow geometry is already fully resolved to 3D
        // positions on the CPU, so the shader needs nothing but the
        // camera's view_proj and the same directional light `sticker_common`
        // shades the puzzle's own stickers with (see `gizmo_shader.wgsl`),
        // so the ring reads with real shading/depth rather than as a flat
        // overlay.
        let gizmo_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("Gizmo Bind Group Layout"),
            });

        // Post-process bind group layouts: one texture + sampler for
        // bright-pass and the two blur passes, and scene + bloom textures
        // sharing one sampler for the final composite.
        let post_process_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("Post Process Bind Group Layout"),
            });

        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("Composite Bind Group Layout"),
            });

        let post_process_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Post Process Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let (bright_pass_bind_group, blur_h_bind_group, blur_v_bind_group, composite_bind_group) =
            create_post_process_bind_groups(
                device,
                &post_process_bind_group_layout,
                &composite_bind_group_layout,
                &post_process_sampler,
                &scene_view,
                &bloom_view_a,
                &bloom_view_b,
            );

        // Create transform uniform buffer with initial slider values
        let transform_data = Transform4D {
            rotation_matrix: nalgebra::Matrix4::identity().into(),
            viewer_distance: ui_controls.viewer_distance,
            sticker_scale: ui_controls.sticker_scale,
            face_gap: ui_controls.face_gap,
            face_gap_4d: ui_controls.face_gap_4d,
            _padding: [0.0; 3],
            elapsed_seconds: 0.0,
        };
        let transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transform Buffer"),
            contents: bytemuck::cast_slice(&[transform_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let main_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &main_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: transform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: piece_slot_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: highlighting_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: sticker_order_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: kind_colors_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: depth_batch_remap_buffer.as_entire_binding(),
                },
            ],
            label: Some("Main Bind Group"),
        });

        let normal_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &normal_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: transform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: instance_buffer.as_entire_binding(),
                },
            ],
            label: Some("Normal Bind Group"),
        });

        let debug_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &debug_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: transform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: instance_buffer.as_entire_binding(),
                },
            ],
            label: Some("Debug Bind Group"),
        });

        // Create debug AABB bind group
        let debug_aabb_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &debug_aabb_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: debug_instance_buffer.as_entire_binding(),
                },
            ],
            label: Some("Debug AABB Bind Group"),
        });

        let gizmo_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &gizmo_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_buffer.as_entire_binding(),
                },
            ],
            label: Some("Gizmo Bind Group"),
        });

        let mut composer = Composer::default();
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source: include_str!("shaders/math4d.wgsl"),
                file_path: "shaders/math4d.wgsl",
                ..Default::default()
            })
            .expect("shaders/math4d.wgsl failed to compose");
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source: include_str!("shaders/sticker_common.wgsl"),
                file_path: "shaders/sticker_common.wgsl",
                ..Default::default()
            })
            .expect("shaders/sticker_common.wgsl failed to compose");
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source: include_str!("shaders/elemental_common.wgsl"),
                file_path: "shaders/elemental_common.wgsl",
                ..Default::default()
            })
            .expect("shaders/elemental_common.wgsl failed to compose");
        let mut compose_shader =
            |source: &str, file_path: &str| match composer.make_naga_module(NagaModuleDescriptor {
                source,
                file_path,
                ..Default::default()
            }) {
                Ok(module) => module,
                Err(err) => panic!("{}", err.emit_to_string(&composer)),
            };

        let classic_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Classic Shader"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(compose_shader(
                include_str!("shaders/classic_shader.wgsl"),
                "shaders/classic_shader.wgsl",
            ))),
        });

        let sky_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky Pipeline Layout"),
            bind_group_layouts: &[&skybox_bind_group_layout],
            push_constant_ranges: &[],
        });

        let classic_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Classic Pipeline Layout"),
                bind_group_layouts: &[&main_bind_group_layout],
                push_constant_ranges: &[],
            });

        let ice_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ice Pipeline Layout"),
            bind_group_layouts: &[&main_bind_group_layout, &ice_bind_group_layout],
            push_constant_ranges: &[],
        });

        let normal_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Normal Pipeline Layout"),
                bind_group_layouts: &[&normal_bind_group_layout],
                push_constant_ranges: &[],
            });

        let debug_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Debug Pipeline Layout"),
                bind_group_layouts: &[&debug_bind_group_layout],
                push_constant_ranges: &[],
            });

        let debug_aabb_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Debug AABB Pipeline Layout"),
                bind_group_layouts: &[&debug_aabb_bind_group_layout],
                push_constant_ranges: &[],
            });

        let gizmo_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Gizmo Pipeline Layout"),
                bind_group_layouts: &[&gizmo_bind_group_layout],
                push_constant_ranges: &[],
            });

        let post_process_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Post Process Pipeline Layout"),
                bind_group_layouts: &[&post_process_bind_group_layout],
                push_constant_ranges: &[],
            });

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Composite Pipeline Layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                push_constant_ranges: &[],
            });

        let sky_vertices: &[[f32; 2]] = &[
            [-1.0, -1.0], // bottom-left
            [1.0, -1.0],  // bottom-right
            [1.0, 1.0],   // top-right
            [-1.0, 1.0],  // top-left
        ];
        let sky_indices: &[u16] = &[0, 1, 2, 0, 2, 3];

        let sky_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Clear Vertex Buffer"),
            contents: bytemuck::cast_slice(sky_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let sky_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Clear Index Buffer"),
            contents: bytemuck::cast_slice(sky_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky"),
            layout: Some(&sky_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &classic_shader,
                entry_point: Some("vs_sky"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                }],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: &classic_shader,
                entry_point: Some("fs_sky"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            }),
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let classic_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Classic Pipeline"),
            layout: Some(&classic_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &classic_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: &classic_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // A move animation rotates a facet's basis away from the
                // static per-4D-face winding `calculate_indices` computed,
                // which isn't recomputed mid-move (see `rotation_changed`
                // in shader_widget.rs) - backface culling against that
                // stale winding can hide the correctly-outward triangle of
                // a moving sticker and show its (dark) interior instead.
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let elemental_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Elemental Shader"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(compose_shader(
                include_str!("shaders/elemental_shader.wgsl"),
                "shaders/elemental_shader.wgsl",
            ))),
        });

        let elemental_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Elemental Pipeline"),
            layout: Some(&classic_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &elemental_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: &elemental_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Fire's material accumulates emission against a transmittance, which
        // needs blending; the opaque elemental pipeline above cannot provide
        // one, so Fire gets its own over the same layout and module.
        //
        // Depth is tested but not written, which is what the back-to-front
        // draw order pays for: a farther ball still blends through a nearer
        // one's translucent rim, the ball's edge fades out instead of being
        // cut where it stops claiming depth, and the particles drawn
        // afterward stay visible where they pass behind a ball. Writing
        // depth here would also write the wrong value - the rasterized
        // fragment sits on the cube's front face, up to a good fraction of a
        // half-width nearer than the ball surface it shades.
        let fire_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fire Pipeline"),
            layout: Some(&classic_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &elemental_shader,
                entry_point: Some("vs_main_batched"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: &elemental_shader,
                entry_point: Some("fs_fire"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    // Premultiplied: `fs_fire` returns emission already
                    // scaled by its own coverage, so the source is added
                    // whole and only the destination is attenuated.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Ice's material raymarches refraction/reflection against a
        // snapshot of the scene so far (`ice_bind_group`'s iChannel1), so it
        // needs the extra bind group `fire_pipeline` doesn't. Unlike Fire it
        // writes a fully opaque result and real depth: there's no blending
        // left for the GPU to do (the shader already composited the
        // background itself via texture reads), and later particles should
        // be properly occluded by / occlude solid ice rather than glow
        // through it.
        let ice_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Ice Pipeline"),
            layout: Some(&ice_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &elemental_shader,
                entry_point: Some("vs_main_batched"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: &elemental_shader,
                entry_point: Some("fs_ice"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Light's material raymarches its own volumetric cloud and, like
        // Fire, carries its whole look as blended emission with no depth
        // write - so this pipeline is `fire_pipeline` in every particular
        // except its entry point and label; it needs no extra bind group,
        // unlike Ice, since it reads back no background snapshot.
        let light_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Light Pipeline"),
            layout: Some(&classic_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &elemental_shader,
                entry_point: Some("vs_main_batched"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: &elemental_shader,
                entry_point: Some("fs_light"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    // Premultiplied, like Fire: `fs_light` returns emission
                    // already scaled by its own coverage, so the source is
                    // added whole and only the destination is attenuated.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Dirt's material raymarches its own bumpy rock box and, like Fire
        // and Light, carries its whole look as blended emission with no
        // depth write - so this pipeline is `light_pipeline` in every
        // particular except its entry point and label. Unlike Fire's/
        // Light's density accumulation, `fs_dirt` returns a hard 0.0/1.0
        // coverage, but the same premultiplied blend state still applies:
        // at 1.0 it behaves like a plain opaque write, and at 0.0 it
        // contributes nothing, letting whatever is farther in the batch (or
        // the opaque pass beneath it) show through unchanged.
        let dirt_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Dirt Pipeline"),
            layout: Some(&classic_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &elemental_shader,
                entry_point: Some("vs_main_batched"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: &elemental_shader,
                entry_point: Some("fs_dirt"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    // Premultiplied, like Fire/Light: `fs_dirt` returns
                    // color already scaled by its own (binary) coverage, so
                    // the source is added whole and only the destination is
                    // attenuated.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Fire/Light/Dirt's depth-only prepass pipelines: same vertex stage,
        // bind group layout and depth format as their color counterparts
        // above, but no color target (depth-only) and `depth_write_enabled:
        // true` so the write actually lands. Drawn front-to-back ahead of
        // the existing color passes, into the same `depth_view`, so those
        // passes' own `depth_compare: Less` can reject fragments a nearer
        // sticker already covered before their raymarch shader ever runs -
        // see `perf_improvements.md` item 1 and `render()`'s "Depth
        // Prepass" pass. Ice is excluded: it already writes real depth from
        // its own color pipeline.
        let create_depth_prepass_pipeline = |label: &str, entry_point: &'static str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&classic_pipeline_layout),
                cache: None,
                vertex: wgpu::VertexState {
                    module: &elemental_shader,
                    entry_point: Some("vs_main_batched"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                    }],
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[],
                        zero_initialize_workgroup_memory: false,
                    },
                },
                fragment: Some(wgpu::FragmentState {
                    module: &elemental_shader,
                    entry_point: Some(entry_point),
                    targets: &[],
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[],
                        zero_initialize_workgroup_memory: false,
                    },
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            })
        };
        let fire_depth_prepass_pipeline =
            create_depth_prepass_pipeline("Fire Depth Prepass Pipeline", "fs_fire_depth");
        let light_depth_prepass_pipeline =
            create_depth_prepass_pipeline("Light Depth Prepass Pipeline", "fs_light_depth");
        let dirt_depth_prepass_pipeline =
            create_depth_prepass_pipeline("Dirt Depth Prepass Pipeline", "fs_dirt_depth");

        let particle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(compose_shader(
                include_str!("shaders/particle_shader.wgsl"),
                "shaders/particle_shader.wgsl",
            ))),
        });

        let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Pipeline"),
            layout: Some(&classic_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &particle_shader,
                entry_point: Some("vs_main"),
                // Billboard corners are generated from the vertex index, so
                // there is no per-vertex data to feed in.
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: &particle_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    // Additive: particles glow over whatever is behind them
                    // and need no back-to-front sort to composite correctly.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                // Tested against the stickers already drawn this pass, so a
                // particle behind one is hidden, but never written, so
                // particles don't occlude each other.
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create normal visualization shader and pipeline
        let normal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Normal Shader"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(compose_shader(
                include_str!("shaders/normal_shader.wgsl"),
                "shaders/normal_shader.wgsl",
            ))),
        });

        let normal_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Normal Pipeline"),
            layout: Some(&normal_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &normal_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: &normal_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Create depth visualization shader and pipeline
        let depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Depth Shader"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(compose_shader(
                include_str!("shaders/depth_shader.wgsl"),
                "shaders/depth_shader.wgsl",
            ))),
        });

        let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Depth Pipeline"),
            layout: Some(&debug_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &depth_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: &depth_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Create debug AABB shader and pipeline for transparent rendering
        let debug_aabb_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Debug AABB Shader"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(compose_shader(
                include_str!("shaders/debug_shader.wgsl"),
                "shaders/debug_shader.wgsl",
            ))),
        });

        let debug_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Debug AABB Pipeline"),
            layout: Some(&debug_aabb_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &debug_aabb_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: &debug_aabb_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING), // Enable transparency
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // Don't write depth for transparency
                depth_compare: wgpu::CompareFunction::Less, // Still test depth
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Create the rotation-axis gizmo's shader and pipeline. Geometry
        // (ring ribbon + a field-loop marker ribbon) is already fully
        // resolved to 3D positions on the CPU each frame (see
        // `shader_widget::gizmo_ring_vertices`), so unlike `debug_pipeline`
        // this needs no per-instance transform - just a plain vertex buffer
        // carrying position, normal and color together.
        let gizmo_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gizmo Shader"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(compose_shader(
                include_str!("shaders/gizmo_shader.wgsl"),
                "shaders/gizmo_shader.wgsl",
            ))),
        });

        let gizmo_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Gizmo Pipeline"),
            layout: Some(&gizmo_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &gizmo_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GizmoVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4],
                }],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: &gizmo_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Unlike `debug_pipeline`'s watertight AABB cubes, a thin
                // ring/arrow ribbon must stay visible from either side as
                // the camera orbits around it.
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                // Unlike `debug_pipeline`'s translucent overlay, the gizmo's
                // ring/marker/arrow colors are now fully opaque (see
                // `GIZMO_FOCUS_PALETTE`/`GIZMO_DRAG_PALETTE`/
                // `GIZMO_ARROW_COLOR`), so it writes real depth like any
                // other opaque scene geometry - letting the translucent
                // Fire/Ice/Light/Dirt batches drawn after it in `render`
                // correctly occlude/be occluded relative to it instead of
                // always compositing on top as a flat overlay.
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let post_process_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Post Process Shader"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(compose_shader(
                include_str!("shaders/post_process.wgsl"),
                "shaders/post_process.wgsl",
            ))),
        });

        // Bright-pass and the two blur passes share one vertex shader, pipeline
        // layout and primitive/multisample state, differing only in fragment
        // entry point; all three write a half-res `HDR_FORMAT` target with no
        // depth test.
        let post_process_pipeline = |label: &str, entry_point: &'static str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&post_process_pipeline_layout),
                cache: None,
                vertex: wgpu::VertexState {
                    module: &post_process_shader,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[],
                        zero_initialize_workgroup_memory: false,
                    },
                },
                fragment: Some(wgpu::FragmentState {
                    module: &post_process_shader,
                    entry_point: Some(entry_point),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[],
                        zero_initialize_workgroup_memory: false,
                    },
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            })
        };

        let bright_pass_pipeline = post_process_pipeline("Bright Pass Pipeline", "fs_bright_pass");
        let blur_h_pipeline = post_process_pipeline("Blur H Pipeline", "fs_blur_h");
        let blur_v_pipeline = post_process_pipeline("Blur V Pipeline", "fs_blur_v");

        // Both composite variants differ only in their fragment entry point:
        // one writes the linear sum straight out, the other rolls it through
        // a tonemapping curve first.
        let composite_pipeline_variant = |label: &str, entry_point: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&composite_pipeline_layout),
                cache: None,
                vertex: wgpu::VertexState {
                    module: &post_process_shader,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[],
                        zero_initialize_workgroup_memory: false,
                    },
                },
                fragment: Some(wgpu::FragmentState {
                    module: &post_process_shader,
                    entry_point: Some(entry_point),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[],
                        zero_initialize_workgroup_memory: false,
                    },
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            })
        };

        let composite_pipeline = composite_pipeline_variant("Composite Pipeline", "fs_composite");
        let composite_tonemapped_pipeline =
            composite_pipeline_variant("Tonemapped Composite Pipeline", "fs_composite_tonemapped");

        // Load skybox cubemap texture
        let (_skybox_texture, skybox_view, skybox_sampler) =
            load_cross_cubemap(device, queue, "src/resources/Cubemap_Sky_02-512x512.png")
                .expect("Failed to load skybox texture");

        // Create skybox bind group
        let skybox_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &skybox_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&skybox_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&skybox_sampler),
                },
            ],
            label: Some("Skybox Bind Group"),
        });

        // Ice's gray noise bump-map texture (see `load_gray_noise_texture`)
        // and the bind group pairing it with the scene-snapshot texture
        // created earlier alongside `scene_texture`.
        let (ice_noise_texture, ice_noise_view, ice_noise_sampler) =
            load_gray_noise_texture(device, queue, "src/resources/ice_noise_64.png")
                .expect("Failed to load Ice noise texture");
        let ice_background_transform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Ice Background Transform Buffer"),
                contents: bytemuck::cast_slice(&[IceBackgroundTransform {
                    scale: [
                        bounds.width / viewport_size.width as f32,
                        bounds.height / viewport_size.height as f32,
                    ],
                    offset: [
                        bounds.x / viewport_size.width as f32,
                        bounds.y / viewport_size.height as f32,
                    ],
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let ice_bind_group = create_ice_bind_group(
            device,
            &ice_bind_group_layout,
            &ice_noise_view,
            &ice_noise_sampler,
            &ice_bg_view,
            &post_process_sampler,
            &ice_background_transform_buffer,
        );
        Self {
            bounds,
            target_format: format,
            sky_vertex_buffer,
            sky_index_buffer,
            sky_pipeline,
            classic_pipeline,
            elemental_pipeline,
            fire_pipeline,
            ice_pipeline,
            light_pipeline,
            dirt_pipeline,
            fire_depth_prepass_pipeline,
            light_depth_prepass_pipeline,
            dirt_depth_prepass_pipeline,
            particle_pipeline,
            normal_pipeline,
            depth_pipeline,
            debug_pipeline,
            gizmo_pipeline,
            bright_pass_pipeline,
            blur_h_pipeline,
            blur_v_pipeline,
            composite_pipeline,
            composite_tonemapped_pipeline,
            current_render_mode: ui_controls.render_mode,
            current_theme: ui_controls.theme,
            current_ground_truth_debug_face: None,
            vertex_buffer,
            face_index_buffer,
            last_indices_generation: None,
            last_sticker_generation: None,
            depth_batch_remap_buffer,
            depth_batch_groups: Vec::new(),
            sticker_order_buffer,
            particle_ranges: initial_particle_ranges,
            num_stickers,
            instance_buffer,
            camera_uniform,
            camera_buffer,
            highlighting_uniform,
            highlighting_buffer,
            light_uniform,
            light_buffer,
            debug_instance_buffer,
            debug_scratch: Vec::new(),
            gizmo_vertex_buffer,
            gizmo_scratch: Vec::new(),
            gizmo_vertex_count: 0,
            main_bind_group,
            normal_bind_group,
            debug_bind_group,
            debug_aabb_bind_group,
            gizmo_bind_group,
            depth_texture,
            depth_view,
            debug_inset_depth_texture,
            debug_inset_depth_view,
            scene_texture,
            scene_view,
            bloom_texture_a,
            bloom_view_a,
            bloom_texture_b,
            bloom_view_b,
            post_process_bind_group_layout,
            composite_bind_group_layout,
            post_process_sampler,
            bright_pass_bind_group,
            blur_h_bind_group,
            blur_v_bind_group,
            composite_bind_group,
            _ice_noise_texture: ice_noise_texture,
            ice_noise_view,
            ice_noise_sampler,
            ice_bg_texture,
            ice_bg_view,
            ice_bind_group_layout,
            ice_bind_group,
            ice_background_transform_buffer,
            transform_buffer,
            skybox_bind_group,
        }
    }

    /// Handles window resize events by updating surface and depth buffer.
    ///
    /// Recreates size-dependent resources like the depth texture when the window
    /// size changes.
    ///
    /// # Arguments
    /// * `new_size` - New window dimensions in pixels
    pub(crate) fn resize(
        &mut self,
        device: &Device,
        new_bounds: Rectangle<f32>,
        new_size: Size<u32>,
    ) {
        if new_bounds != self.bounds && new_bounds.width > 0.0 && new_bounds.height > 0.0 {
            self.bounds = new_bounds;
        }

        if new_size.width > 0
            && new_size.height > 0
            && (self.depth_texture.size().width != new_size.width
                || self.depth_texture.size().height != new_size.height)
        {
            (self.depth_texture, self.depth_view) =
                create_depth_texture(device, "Depth Texture", new_size.width, new_size.height);
            (self.debug_inset_depth_texture, self.debug_inset_depth_view) = create_depth_texture(
                device,
                "Fire Ground Truth Debug Inset Depth Texture",
                new_size.width,
                new_size.height,
            );

            (self.scene_texture, self.scene_view) = create_hdr_target(
                device,
                "Scene Texture",
                new_size.width,
                new_size.height,
                wgpu::TextureUsages::COPY_SRC,
            );
            let bloom_width = (new_size.width / 2).max(1);
            let bloom_height = (new_size.height / 2).max(1);
            (self.bloom_texture_a, self.bloom_view_a) = create_hdr_target(
                device,
                "Bloom Texture A",
                bloom_width,
                bloom_height,
                wgpu::TextureUsages::empty(),
            );
            (self.bloom_texture_b, self.bloom_view_b) = create_hdr_target(
                device,
                "Bloom Texture B",
                bloom_width,
                bloom_height,
                wgpu::TextureUsages::empty(),
            );
            (self.ice_bg_texture, self.ice_bg_view) = create_hdr_target(
                device,
                "Ice Background Texture",
                new_size.width,
                new_size.height,
                wgpu::TextureUsages::COPY_DST,
            );

            (
                self.bright_pass_bind_group,
                self.blur_h_bind_group,
                self.blur_v_bind_group,
                self.composite_bind_group,
            ) = create_post_process_bind_groups(
                device,
                &self.post_process_bind_group_layout,
                &self.composite_bind_group_layout,
                &self.post_process_sampler,
                &self.scene_view,
                &self.bloom_view_a,
                &self.bloom_view_b,
            );

            self.ice_bind_group = create_ice_bind_group(
                device,
                &self.ice_bind_group_layout,
                &self.ice_noise_view,
                &self.ice_noise_sampler,
                &self.ice_bg_view,
                &self.post_process_sampler,
                &self.ice_background_transform_buffer,
            );
        }
    }

    pub(crate) fn update_camera(
        &mut self,
        queue: &Queue,
        camera: &Camera,
        projection: &Projection,
    ) {
        self.camera_uniform.update_view_proj(camera, projection);
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // `bounds` (the shader widget's viewport) and `scene_texture`'s size
        // (the full window) can each change independently between frames -
        // recomputed unconditionally here rather than only in `resize`,
        // since `update_camera` already runs every frame regardless.
        let scene_size = self.scene_texture.size();
        let ice_background_transform = IceBackgroundTransform {
            scale: [
                self.bounds.width / scene_size.width as f32,
                self.bounds.height / scene_size.height as f32,
            ],
            offset: [
                self.bounds.x / scene_size.width as f32,
                self.bounds.y / scene_size.height as f32,
            ],
        };
        queue.write_buffer(
            &self.ice_background_transform_buffer,
            0,
            bytemuck::cast_slice(&[ice_background_transform]),
        );
    }

    /// Updates the light direction to track the camera, offset top-right.
    pub(crate) fn update_light(&mut self, queue: &Queue, camera: &Camera) {
        self.light_uniform.direction = camera.top_right_light_direction().into();
        queue.write_buffer(
            &self.light_buffer,
            0,
            bytemuck::cast_slice(&[self.light_uniform]),
        );
    }

    /// Sets the current render mode
    pub(crate) fn set_render_mode(&mut self, mode: RenderMode) {
        self.current_render_mode = mode;
    }

    /// Sets the current sticker theme
    pub(crate) fn set_theme(&mut self, theme: Theme) {
        self.current_theme = theme;
    }

    /// Sets which face_id, if any, the Fire ground-truth debug inset draws
    /// this frame.
    pub(crate) fn set_ground_truth_debug_face(&mut self, face_id: Option<u32>) {
        self.current_ground_truth_debug_face = face_id;
    }

    /// Updates the instance buffer using compute shaders for 4D transformations.
    ///
    /// Runs the 4D transformation compute shader and copies the result to the instance buffer.
    ///
    /// # Arguments
    /// * `queue` - GPU queue for submitting commands
    /// * `rotation_4d` - Current 4D rotation matrix
    /// * `sticker_scale` - Scale factor for individual stickers (from sticker scale slider)
    /// * `face_gap` - 3D distance to push each face outward (from face gap slider)
    /// * `face_gap_4d` - 4D anchor scale for each facet (from 4D face gap slider)
    /// * `viewer_distance` - Distance of the 4D viewer from the W=0 plane
    ///   (from the 4D viewer distance slider)
    /// * `elapsed_seconds` - Wall-clock seconds since the app started,
    ///   wrapped modulo 3600
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update_instances(
        &mut self,
        queue: &Queue,
        rotation_4d: &nalgebra::Matrix4<f32>,
        sticker_scale: f32,
        face_gap: f32,
        face_gap_4d: f32,
        viewer_distance: f32,
        elapsed_seconds: f32,
    ) {
        // Update transform uniform
        let transform_data = Transform4D {
            rotation_matrix: (*rotation_4d).into(),
            viewer_distance,
            sticker_scale,
            face_gap,
            face_gap_4d,
            _padding: [0.0; 3],
            elapsed_seconds,
        };
        queue.write_buffer(
            &self.transform_buffer,
            0,
            bytemuck::cast_slice(&[transform_data]),
        );
    }

    /// Uploads `indices` to the GPU only if `generation` differs from the
    /// last generation uploaded, skipping the `write_buffer` call when the
    /// caller's cached indices haven't actually changed since last frame.
    pub(crate) fn update_indices(&mut self, queue: &Queue, indices: &[u16], generation: u64) {
        if self.last_indices_generation == Some(generation) {
            return;
        }
        queue.write_buffer(&self.face_index_buffer, 0, bytemuck::cast_slice(indices));
        self.last_indices_generation = Some(generation);
    }

    /// Uploads `instances` to the GPU only if `generation` differs from the
    /// last generation uploaded, mirroring `update_indices`. Also rebuilds
    /// and re-uploads the particle pipeline's `(face_id, kind)` indirection
    /// buffer, since a move can change which kind occupies which face.
    pub(crate) fn update_sticker_instances(
        &mut self,
        queue: &Queue,
        instances: &[StickerInstance],
        generation: u64,
    ) {
        if self.last_sticker_generation == Some(generation) {
            return;
        }
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));

        let facets_per_face = (instances.len() / 8) as u32;
        let order = build_sticker_order(instances, facets_per_face);
        let (particle_instances, particle_ranges) =
            build_particle_instances(&order, &Theme::Elemental.particles_per_kind());
        queue.write_buffer(
            &self.sticker_order_buffer,
            0,
            bytemuck::cast_slice(&particle_instances),
        );
        self.particle_ranges = particle_ranges;

        self.last_sticker_generation = Some(generation);
    }

    /// Builds this frame's Fire/Ice draw plan from `depth_batches`
    /// (`shader_widget::depth_draw_order`) and uploads the matching
    /// `depth_batch_remap_buffer` contents, so `render()` can later issue
    /// one instanced `draw_indexed` call per `(kind, face_id)` group within
    /// a batch instead of one per sticker (see `DepthBatchGroup`). Must run
    /// before `render()` or `capture_frame` each frame under
    /// `Theme::Elemental`; a no-op elsewhere, since `depth_batches` is empty
    /// under any other theme.
    pub(crate) fn update_depth_batches(
        &mut self,
        queue: &Queue,
        depth_batches: &[Vec<crate::shader_widget::DepthLayer>],
    ) {
        let facets_per_face = self.num_stickers as u32 / 8;
        let (groups, flat_remap) = group_depth_batches(depth_batches, facets_per_face);
        queue.write_buffer(
            &self.depth_batch_remap_buffer,
            0,
            bytemuck::cast_slice(&flat_remap),
        );
        self.depth_batch_groups = groups;
    }

    /// Updates the highlighting uniform buffer with the currently hovered
    /// sticker and its owning piece (looked up via `FACET_TABLE`).
    ///
    /// # Arguments
    /// * `queue` - GPU command queue for buffer updates
    /// * `hovered_sticker_index` - Index of the sticker being hovered (None if no hover)
    pub(crate) fn update_highlighting(
        &mut self,
        queue: &Queue,
        hovered_sticker_index: Option<usize>,
    ) {
        self.highlighting_uniform.hovered_sticker_index = hovered_sticker_index
            .map(|index| index as u32)
            .unwrap_or(u32::MAX);
        self.highlighting_uniform.hovered_piece_slot = hovered_sticker_index
            .map(|index| FACET_TABLE[index].piece_slot as u32)
            .unwrap_or(u32::MAX);

        queue.write_buffer(
            &self.highlighting_buffer,
            0,
            bytemuck::cast_slice(&[self.highlighting_uniform]),
        );
    }

    /// Sets which facet counts (2..=4; `0` = unused) the first-run
    /// tutorial's current step wants pulse-highlighted, if any.
    pub(crate) fn update_tutorial_flash(&mut self, queue: &Queue, targets: [u8; 2]) {
        self.highlighting_uniform.tutorial_flash_facet_count_a = targets[0] as u32;
        self.highlighting_uniform.tutorial_flash_facet_count_b = targets[1] as u32;

        queue.write_buffer(
            &self.highlighting_buffer,
            0,
            bytemuck::cast_slice(&[self.highlighting_uniform]),
        );
    }

    /// Updates the debug instances buffer for AABB visualization
    ///
    /// # Arguments
    /// * `queue` - GPU command queue for buffer updates
    /// * `debug_instances` - Debug instances to render as transparent AABBs
    pub(crate) fn update_debug_instances(
        &mut self,
        queue: &Queue,
        debug_instances: &[DebugInstanceWithDistance],
    ) {
        // Extract GPU data from debug instances (already sorted back-to-front)
        // into a scratch buffer reused across frames, instead of allocating a
        // fresh Vec every frame for what's usually empty (AABB debug mode is
        // off by default).
        self.debug_scratch.clear();
        self.debug_scratch
            .extend(debug_instances.iter().map(|instance| instance.gpu_data));

        // Write to GPU buffer
        queue.write_buffer(
            &self.debug_instance_buffer,
            0,
            bytemuck::cast_slice(&self.debug_scratch),
        );
    }

    /// Updates the gizmo vertex buffer, mirroring `update_debug_instances`.
    ///
    /// # Arguments
    /// * `queue` - GPU command queue for buffer updates
    /// * `vertices` - This frame's gizmo ring/arrow geometry, already
    ///   projected to 3D (see `shader_widget::gizmo_ring_vertices`); empty
    ///   when no gizmo should be shown this frame.
    pub(crate) fn update_gizmo(&mut self, queue: &Queue, vertices: &[GizmoVertex]) {
        debug_assert!(
            vertices.len() <= GIZMO_VERTEX_CAPACITY,
            "gizmo vertex count {} exceeds fixed capacity {GIZMO_VERTEX_CAPACITY}",
            vertices.len()
        );
        self.gizmo_scratch.clear();
        self.gizmo_scratch.extend_from_slice(vertices);
        self.gizmo_vertex_count = vertices.len() as u32;
        queue.write_buffer(
            &self.gizmo_vertex_buffer,
            0,
            bytemuck::cast_slice(&self.gizmo_scratch),
        );
    }

    /// Renders a single frame of the hypercube visualization into the
    /// offscreen HDR `scene_view`; `composite` blits it (plus, under
    /// `Theme::Elemental`, bloom) into the real surface afterward.
    ///
    /// Under `(RenderMode::Standard, Theme::Elemental)` this is several
    /// passes rather than one, all loading (not clearing) `scene_view`/
    /// `depth_view` so each sees what the last left behind:
    /// 1. Skybox, then the opaque hypercube (every kind but Ice and Fire,
    ///    which both discard - see `elemental_shader.wgsl`'s `fs_main`),
    ///    then the rotation-axis gizmo (ring/marker/arrows), if any - drawn
    ///    in this same pass right after the opaque hypercube so it's shaded
    ///    and writes real depth like any other opaque geometry (see
    ///    `gizmo_pipeline`'s doc comment), rather than as a flat overlay
    ///    added after the frame is otherwise done.
    /// 2. Fire's and Ice's stickers, back-to-front together per
    ///    `depth_batches` (`shader_widget::depth_draw_order`) so either kind
    ///    correctly draws over the other depending on the current 4D
    ///    rotation, not one kind fully before the other. Batches must run in
    ///    order, but a batch's own members share one pass, since nothing
    ///    established a draw-order relation between any two of them -
    ///    `depth_draw_order` only puts stickers in the same batch when
    ///    `FireSticker::is_behind` found no evidence any pair of them
    ///    occludes the other on screen. If a batch contains any Ice, its
    ///    entries need `scene_texture` copied into `ice_bg_texture` once
    ///    before the pass opens (outside any pass, since a texture can't be
    ///    a render target and a sampled input at once) so `ice_pipeline` has
    ///    every earlier batch's result to sample as its refraction/
    ///    reflection background - one copy per batch instead of one per Ice
    ///    sticker. That trade is not free: `is_behind`'s occlusion test says
    ///    nothing about where a reflection ray can land, so two Ice stickers
    ///    batched together (e.g. neighbors on the same face, mutually
    ///    unrelated by occlusion) no longer see each other's own result the
    ///    way they would if drawn one at a time, losing some mutual
    ///    reflection detail between them in exchange for far fewer copies
    ///    and passes on a mostly- or fully-Ice face. Unlike Fire, Ice writes
    ///    real depth, so later particles are correctly hidden behind or
    ///    drawn over it. Within a batch, every same-kind, same-`face_id`
    ///    run of stickers - the most a `draw_indexed` call can cover at
    ///    once, see `DepthBatchGroup` - draws with one instanced call
    ///    instead of one per sticker, via `self.depth_batch_groups`
    ///    (`update_depth_batches`, which must run first each frame).
    /// 3. Particles, now that Ice's depth is in place for them to test
    ///    against.
    ///
    /// # Arguments
    /// * `camera` - Current camera state for view matrix
    /// * `projection` - Current projection parameters
    /// * `visible_faces` - Per-`face_id` visibility (see `math::visible_faces`);
    ///   faces marked invisible are skipped entirely, issuing no draw call
    ///   and no vertex-shader invocations for their 27 instances.
    pub(crate) fn render(&self, encoder: &mut CommandEncoder, visible_faces: &[bool; 8]) {
        let indices_per_face = VERTEX_NORMAL_INDICES.len() as u32;
        let facets_per_face = self.num_stickers as u32 / 8;
        let is_elemental_standard = (self.current_render_mode, self.current_theme)
            == (RenderMode::Standard, Theme::Elemental);

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // No clear needed: the skybox pass right below always
                        // draws an opaque fullscreen quad over the whole
                        // viewport first, so every pixel `composite` will later
                        // read gets fully overwritten regardless of what was in
                        // `scene_view` before this pass.
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_viewport(
                self.bounds.x,
                self.bounds.y,
                self.bounds.width,
                self.bounds.height,
                0.0,
                1.0,
            );

            // First render the skybox
            render_pass.set_pipeline(&self.sky_pipeline);
            render_pass.set_bind_group(0, &self.skybox_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.sky_vertex_buffer.slice(..));
            render_pass
                .set_index_buffer(self.sky_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..6, 0, 0..1);

            // Then render the hypercube
            let (pipeline, bind_group) = match (self.current_render_mode, self.current_theme) {
                (RenderMode::Standard, Theme::Classic) => {
                    (&self.classic_pipeline, &self.main_bind_group)
                }
                (RenderMode::Standard, Theme::Elemental) => {
                    (&self.elemental_pipeline, &self.main_bind_group)
                }
                (RenderMode::Normals, _) => (&self.normal_pipeline, &self.normal_bind_group),
                (RenderMode::Depth, _) => (&self.depth_pipeline, &self.debug_bind_group),
            };
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass
                .set_index_buffer(self.face_index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            // One draw per 4D face: `face_index_buffer` holds 8 winding-corrected
            // 36-index chunks (one per face_id, see `calculate_indices`), and
            // `FACET_TABLE` (piece.rs) is built in matching face-major blocks of
            // 27, so chunk N only ever reaches the instances it was computed
            // for. A single draw over all 288 indices and 216 instances would
            // feed every chunk to every instance, relying on backface culling to
            // silently discard the wrong ones; slicing per face keeps culling
            // meaningful instead.
            // Faces `visible_faces` marks invisible skip the draw call entirely,
            // rather than issuing it and relying on the vertex shader to cull.
            for face_id in 0..8u32 {
                if !visible_faces[face_id as usize] {
                    continue;
                }
                let index_start = face_id * indices_per_face;
                let instance_start = face_id * facets_per_face;
                render_pass.draw_indexed(
                    index_start..index_start + indices_per_face,
                    0,
                    instance_start..instance_start + facets_per_face,
                );
            }

            // Fire's and Ice's stickers draw after the opaque ones, so the
            // depth they test against is complete; both are handled below,
            // batched by `depth_batches` rather than in this pass.

            // The rotation-axis gizmo draws next, in this same pass, so it
            // gets real shading and writes real depth against the puzzle's
            // own opaque geometry (see `gizmo_pipeline`'s doc comment) -
            // before the translucent Fire/Ice/Light/Dirt batches below, so
            // those correctly occlude/be occluded relative to it instead of
            // it always compositing on top as a flat post-process overlay.
            if self.gizmo_vertex_count > 0 {
                render_pass.set_pipeline(&self.gizmo_pipeline);
                render_pass.set_bind_group(0, &self.gizmo_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.gizmo_vertex_buffer.slice(..));
                render_pass.draw(0..self.gizmo_vertex_count, 0..1);
            }
        }

        if is_elemental_standard {
            // Depth-only prepass: the same batches the color loop below
            // draws, walked front-to-back via `.rev()` of the back-to-front
            // partition `group_depth_batches` already computed (see its own
            // doc comment - reversing it is still a valid partition, just
            // walked the other way) so a nearer sticker's solid depth lands
            // in `depth_view` before the color loop's own `depth_compare:
            // Less` tests farther ones against it. Ice is skipped: its own
            // color pass already writes real depth (`ice_pipeline`'s
            // `depth_write_enabled: true`), so it needs no prepass draw.
            // See `perf_improvements.md` item 1.
            for batch in self.depth_batch_groups.iter().rev() {
                let has_prepass_kind = batch.iter().any(|group| group.kind != DepthBatchKind::Ice);
                if !has_prepass_kind {
                    continue;
                }

                let mut prepass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Depth Prepass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                prepass.set_viewport(
                    self.bounds.x,
                    self.bounds.y,
                    self.bounds.width,
                    self.bounds.height,
                    0.0,
                    1.0,
                );

                let mut bound_kind: Option<DepthBatchKind> = None;
                for group in batch {
                    if group.kind == DepthBatchKind::Ice {
                        continue;
                    }
                    if bound_kind != Some(group.kind) {
                        match group.kind {
                            DepthBatchKind::Fire => {
                                prepass.set_pipeline(&self.fire_depth_prepass_pipeline)
                            }
                            DepthBatchKind::Light => {
                                prepass.set_pipeline(&self.light_depth_prepass_pipeline)
                            }
                            DepthBatchKind::Dirt => {
                                prepass.set_pipeline(&self.dirt_depth_prepass_pipeline)
                            }
                            DepthBatchKind::Ice => {
                                unreachable!("Ice groups are filtered out above")
                            }
                        }
                        prepass.set_bind_group(0, &self.main_bind_group, &[]);
                        prepass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                        prepass.set_index_buffer(
                            self.face_index_buffer.slice(..),
                            wgpu::IndexFormat::Uint16,
                        );
                        bound_kind = Some(group.kind);
                    }

                    let index_start = group.face_id * indices_per_face;
                    prepass.draw_indexed(
                        index_start..index_start + indices_per_face,
                        0,
                        group.remap_offset..group.remap_offset + group.count,
                    );
                }
            }
        }

        if is_elemental_standard {
            let scene_size = self.scene_texture.size();

            // Batches must run in order (each one fresh `RenderPass` local,
            // never one carried across iterations, so the borrow checker
            // can see each pass as done before the next begins), but a
            // batch's own groups can draw in any order - `depth_batches`
            // only ever puts stickers in the same batch when nothing
            // establishes an order between any two of them (see `render`'s
            // doc comment) - so groups are simply issued Fire-then-Ice
            // (`group_depth_batches`'s own emission order), purely to
            // minimize pipeline/bind-group switches inside the pass.
            for batch in &self.depth_batch_groups {
                let has_ice = batch.iter().any(|group| group.kind == DepthBatchKind::Ice);

                if has_ice {
                    // A texture can't be a render target and a sampled
                    // input at the same time, so this copy runs with no
                    // pass open - guaranteed here since the previous
                    // batch's pass, a fresh local of its own, was dropped
                    // at the end of the last loop iteration. One copy
                    // covers every Ice group in this batch: none of them
                    // need to see each other's result, only every earlier
                    // batch's, which this snapshot already has.
                    encoder.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &self.scene_texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: &self.ice_bg_texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        scene_size,
                    );
                }

                let mut batch_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Depth Layer Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.scene_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                batch_pass.set_viewport(
                    self.bounds.x,
                    self.bounds.y,
                    self.bounds.width,
                    self.bounds.height,
                    0.0,
                    1.0,
                );

                // Only re-bind pipeline/bind-groups/buffers on a kind
                // change - `group_depth_batches` already emits every Fire
                // group before any Ice group before any Light group before
                // any Dirt group, so this is at most three switches per
                // batch.
                let mut bound_kind: Option<DepthBatchKind> = None;
                for group in batch {
                    if bound_kind != Some(group.kind) {
                        match group.kind {
                            DepthBatchKind::Fire => {
                                batch_pass.set_pipeline(&self.fire_pipeline);
                                batch_pass.set_bind_group(0, &self.main_bind_group, &[]);
                            }
                            DepthBatchKind::Ice => {
                                batch_pass.set_pipeline(&self.ice_pipeline);
                                batch_pass.set_bind_group(0, &self.main_bind_group, &[]);
                                batch_pass.set_bind_group(1, &self.ice_bind_group, &[]);
                            }
                            DepthBatchKind::Light => {
                                batch_pass.set_pipeline(&self.light_pipeline);
                                batch_pass.set_bind_group(0, &self.main_bind_group, &[]);
                            }
                            DepthBatchKind::Dirt => {
                                batch_pass.set_pipeline(&self.dirt_pipeline);
                                batch_pass.set_bind_group(0, &self.main_bind_group, &[]);
                            }
                        }
                        batch_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                        batch_pass.set_index_buffer(
                            self.face_index_buffer.slice(..),
                            wgpu::IndexFormat::Uint16,
                        );
                        bound_kind = Some(group.kind);
                    }

                    let index_start = group.face_id * indices_per_face;
                    batch_pass.draw_indexed(
                        index_start..index_start + indices_per_face,
                        0,
                        group.remap_offset..group.remap_offset + group.count,
                    );
                }
            }

            let mut particle_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Particle Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            particle_pass.set_viewport(
                self.bounds.x,
                self.bounds.y,
                self.bounds.width,
                self.bounds.height,
                0.0,
                1.0,
            );
            particle_pass.set_pipeline(&self.particle_pipeline);
            particle_pass.set_bind_group(0, &self.main_bind_group, &[]);
            for (face_id, visible) in visible_faces.iter().enumerate() {
                if !visible {
                    continue;
                }
                for range in &self.particle_ranges[face_id] {
                    if range.is_empty() {
                        continue;
                    }
                    particle_pass.draw(0..QUAD_VERTICES, range.clone());
                }
            }
        }

        if let Some(face_id) = self.current_ground_truth_debug_face {
            self.render_ground_truth_debug_inset(
                encoder,
                face_id,
                indices_per_face,
                facets_per_face,
            );
        }
    }

    /// Draws every cell of `face_id` - not just Fire ones - opaque and
    /// depth-tested, restricted to a small square viewport in the corner of
    /// `scene_view`, regardless of `visible_faces`. Reuses `classic_pipeline`
    /// and the same live vertex/instance/transform buffers `render` just
    /// drew with, so the inset can never drift from what `depth_draw_order`'s
    /// topological sort actually scored - the exact same transform decides the
    /// geometry either way. Loads `scene_view` rather than clearing it, so
    /// the main scene this pass draws over survives; clears its own
    /// dedicated `debug_inset_depth_view` rather than `depth_view`, so it
    /// can't disturb the depth values `render_debug_aabb` reads back after
    /// `composite()`.
    fn render_ground_truth_debug_inset(
        &self,
        encoder: &mut CommandEncoder,
        face_id: u32,
        indices_per_face: u32,
        facets_per_face: u32,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Fire Ground Truth Debug Inset"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.scene_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.debug_inset_depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        let inset_extent =
            GROUND_TRUTH_DEBUG_INSET_FRACTION * self.bounds.width.min(self.bounds.height);
        render_pass.set_viewport(
            self.bounds.x,
            self.bounds.y,
            inset_extent,
            inset_extent,
            0.0,
            1.0,
        );

        render_pass.set_pipeline(&self.classic_pipeline);
        render_pass.set_bind_group(0, &self.main_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.face_index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        let index_start = face_id * indices_per_face;
        let instance_start = face_id * facets_per_face;
        render_pass.draw_indexed(
            index_start..index_start + indices_per_face,
            0,
            instance_start..instance_start + facets_per_face,
        );
    }

    /// Blits `scene_view` into `target`, within `self.bounds`. Under
    /// `Theme::Elemental` this first runs a bright-pass and a two-pass
    /// separable blur so bloom is added in, and tonemaps on the way out;
    /// otherwise the bloom input is cleared to black rather than sampled
    /// stale and the scene is written through unchanged. Debug AABBs are
    /// drawn afterward, directly onto `target`, so they're excluded from
    /// bloom.
    pub(crate) fn composite(&self, encoder: &mut CommandEncoder, target: &TextureView) {
        let elemental = self.current_theme == Theme::Elemental;
        if elemental {
            draw_fullscreen_pass(
                encoder,
                "Bright Pass",
                &self.bright_pass_pipeline,
                &self.bright_pass_bind_group,
                &self.bloom_view_a,
                None,
            );
            draw_fullscreen_pass(
                encoder,
                "Blur H Pass",
                &self.blur_h_pipeline,
                &self.blur_h_bind_group,
                &self.bloom_view_b,
                None,
            );
            draw_fullscreen_pass(
                encoder,
                "Blur V Pass",
                &self.blur_v_pipeline,
                &self.blur_v_bind_group,
                &self.bloom_view_a,
                None,
            );
        } else {
            clear_texture(encoder, "Clear Bloom", &self.bloom_view_a);
        }

        let composite_pipeline = if elemental {
            &self.composite_tonemapped_pipeline
        } else {
            &self.composite_pipeline
        };
        draw_fullscreen_pass(
            encoder,
            "Composite Pass",
            composite_pipeline,
            &self.composite_bind_group,
            target,
            Some(self.bounds),
        );
    }

    /// Renders one full frame - `render` then `composite`, as
    /// `HypercubePrimitive::render` does - into a scratch texture of its
    /// own, then reads the result back to CPU memory as tightly-packed
    /// RGBA8 rows.
    ///
    /// The scratch texture is sized to the full render target (matching
    /// `scene_texture`/`depth_texture`, not just `self.bounds`), since
    /// `composite`'s own viewport is a sub-rectangle at `self.bounds`'s
    /// offset within that larger target - a texture sized to `bounds` alone
    /// could clip it. wgpu zero-initializes a texture's content on first
    /// use, so pixels outside `bounds` read back as transparent black rather
    /// than garbage.
    ///
    /// Blocks the calling thread on `device.poll` until the copy completes.
    /// Fine for an occasional manual capture or a test; wrong for a hot
    /// path.
    ///
    /// Like `render`, needs `update_depth_batches` called first under
    /// `Theme::Elemental`.
    pub(crate) fn capture_frame(
        &mut self,
        device: &Device,
        queue: &Queue,
        visible_faces: &[bool; 8],
    ) -> (Vec<u8>, u32, u32) {
        let size = self.scene_texture.size();
        log::info!(
            "Capturing a {}x{} frame at surface format {:?}",
            size.width,
            size.height,
            self.target_format
        );

        let capture_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Frame Capture Target"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let capture_view = capture_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.render(&mut encoder, visible_faces);
        self.composite(&mut encoder, &capture_view);
        queue.submit(Some(encoder.finish()));

        let pixels = read_texture_rgba8(device, queue, &capture_texture, self.target_format);
        (pixels, size.width, size.height)
    }

    /// Renders transparent debug AABB visualization
    ///
    /// # Arguments
    /// * `encoder` - Command encoder for GPU commands
    /// * `target` - Target texture view to render to
    /// * `debug_instance_count` - Number of debug instances to render
    pub(crate) fn render_debug_aabb(
        &self,
        encoder: &mut CommandEncoder,
        target: &TextureView,
        debug_instance_count: u32,
    ) {
        if debug_instance_count == 0 {
            return; // Nothing to render
        }

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Debug AABB Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Don't clear - render on top of existing content
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Keep existing depth values
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_viewport(
            self.bounds.x,
            self.bounds.y,
            self.bounds.width,
            self.bounds.height,
            0.0,
            1.0,
        );

        // Render transparent debug AABBs
        render_pass.set_pipeline(&self.debug_pipeline);
        render_pass.set_bind_group(0, &self.debug_aabb_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..)); // Use same cube vertices

        // Draw debug instances (36 vertices per cube, debug_instance_count instances)
        render_pass.draw(0..36, 0..debug_instance_count);
    }
}

impl shader::Pipeline for Renderer {
    /// Creates the renderer's GPU resources.
    ///
    /// The real hypercube data, viewport bounds, and UI controls aren't known yet at this
    /// point; they're supplied on the very first `prepare` call via `resize` and the other
    /// `update_*` methods, so placeholder values are fine here.
    fn new(device: &Device, queue: &Queue, format: TextureFormat) -> Self {
        Renderer::new(
            device,
            queue,
            format,
            Rectangle::default(),
            Size::new(1, 1),
            UiControls {
                sticker_scale: 0.0,
                face_gap: 0.0,
                face_gap_4d: 1.0,
                viewer_distance: crate::math::VIEWER_DISTANCE,
                render_mode: RenderMode::Standard,
                theme: Theme::Classic,
                ground_truth_debug_face: None,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::CameraController;
    use crate::shader_widget::{
        SECONDARY_FACE_GAP, SECONDARY_FACE_GAP_4D, SECONDARY_STICKER_SCALE,
    };

    #[test]
    fn transform4d_size_is_16_byte_aligned() {
        assert_eq!(std::mem::size_of::<Transform4D>() % 16, 0);
    }

    fn sticker_instance_with_kind(kind: u32) -> StickerInstance {
        StickerInstance {
            position_4d: [0.0; 4],
            basis: [[0.0; 4]; 3],
            face_normal_4d: [0.0; 4],
            kind,
            facet_count: 0,
            _padding: [0; 2],
        }
    }

    /// 8 face-major blocks of 27, kind cycling 0..8 within each block so
    /// every `(face_id, kind)` group is non-empty.
    fn test_instances(facets_per_face: u32) -> Vec<StickerInstance> {
        (0..8 * facets_per_face)
            .map(|index| sticker_instance_with_kind(index % 8))
            .collect()
    }

    #[test]
    fn sticker_order_covers_every_sticker_exactly_once() {
        let facets_per_face = 27;
        let instances = test_instances(facets_per_face);

        let order = build_sticker_order(&instances, facets_per_face);

        let mut seen = vec![false; instances.len()];
        for &index in &order.sorted {
            assert!(!seen[index as usize], "sticker {index} appears twice");
            seen[index as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "every sticker must appear");
    }

    #[test]
    fn sticker_order_ranges_are_contiguous_and_non_overlapping() {
        let facets_per_face = 27;
        let instances = test_instances(facets_per_face);

        let order = build_sticker_order(&instances, facets_per_face);

        let mut expected_start = 0u32;
        for face_id in 0..8usize {
            for kind in 0..8usize {
                let range = order.ranges[face_id][kind].clone();
                assert_eq!(range.start, expected_start);
                expected_start = range.end;
            }
        }
        assert_eq!(expected_start, order.sorted.len() as u32);
    }

    #[test]
    fn sticker_order_groups_belong_to_their_face_and_kind() {
        let facets_per_face = 27;
        let instances = test_instances(facets_per_face);

        let order = build_sticker_order(&instances, facets_per_face);

        for face_id in 0..8usize {
            for kind in 0..8usize {
                let range = order.ranges[face_id][kind].clone();
                for &index in &order.sorted[range.start as usize..range.end as usize] {
                    assert_eq!(index / facets_per_face, face_id as u32);
                    assert_eq!(instances[index as usize].kind, kind as u32);
                }
            }
        }
    }

    #[test]
    fn particle_instances_repeat_each_sticker_by_its_kind_count() {
        let facets_per_face = 27;
        let instances = test_instances(facets_per_face);
        let order = build_sticker_order(&instances, facets_per_face);

        let mut counts = [0u32; 8];
        for (kind, count) in counts.iter_mut().enumerate() {
            *count = kind as u32 + 1;
        }

        let (expanded, ranges) = build_particle_instances(&order, &counts);

        for (face_id, kind_ranges) in ranges.iter().enumerate() {
            for (kind, range) in kind_ranges.iter().enumerate() {
                let sticker_range = order.ranges[face_id][kind].clone();
                let expected_len = (sticker_range.end - sticker_range.start) * counts[kind];
                assert_eq!(range.end - range.start, expected_len);
                for &sticker_index in &expanded[range.start as usize..range.end as usize] {
                    assert_eq!(instances[sticker_index as usize].kind, kind as u32);
                }
            }
        }
        assert_eq!(expanded.len(), ranges[7][7].end as usize);
    }

    #[test]
    fn render_with_elemental_theme_does_not_panic() {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("no GPU adapter available to run this smoke test");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("failed to request a device for this smoke test");

        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut renderer = Renderer::new(
            &device,
            &queue,
            format,
            Rectangle {
                x: 0.0,
                y: 0.0,
                width: 64.0,
                height: 64.0,
            },
            Size::new(64, 64),
            UiControls {
                sticker_scale: 0.5,
                face_gap: 0.1,
                face_gap_4d: 1.0,
                viewer_distance: crate::math::VIEWER_DISTANCE,
                render_mode: RenderMode::Standard,
                theme: Theme::Elemental,
                ground_truth_debug_face: None,
            },
        );

        renderer.update_depth_batches(
            &queue,
            &[
                vec![
                    crate::shader_widget::DepthLayer::Fire(0),
                    crate::shader_widget::DepthLayer::Ice(1),
                ],
                vec![crate::shader_widget::DepthLayer::Fire(2)],
            ],
        );
        let (pixels, width, height) = renderer.capture_frame(&device, &queue, &[true; 8]);

        assert_eq!((width, height), (64, 64));
        assert_eq!(pixels.len(), (width * height * 4) as usize);
        let first_pixel: [u8; 4] = pixels[0..4].try_into().unwrap();
        assert!(
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .any(|pixel| pixel != &first_pixel),
            "expected more than one distinct color across the frame (skybox alone should vary)"
        );
    }

    #[test]
    fn read_texture_rgba8_reconstructs_a_non_aligned_width_pattern() {
        // 64x64 (the smoke test above) has 64 * 4 = 256 bytes per row, which
        // is already wgpu's copy alignment - it can't exercise row
        // unpadding at all. This width's row (100 * 4 = 400 bytes) needs
        // padding out to 512, so a stride bug in `read_texture_rgba8` would
        // show up here as a mismatch against the known input pattern.
        let (width, height) = (100u32, 37u32);
        let format = wgpu::TextureFormat::Rgba8Unorm;

        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("no GPU adapter available to run this smoke test");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("failed to request a device for this smoke test");

        let pattern: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| [(x % 256) as u8, (y % 256) as u8, 128, 255])
            })
            .collect();

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Pattern Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pattern,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let pixels = read_texture_rgba8(&device, &queue, &texture, format);

        assert_eq!(pixels, pattern);
    }

    #[test]
    fn read_texture_rgba8_unpacks_rgb10a2() {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("no GPU adapter available to run this smoke test");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("failed to request a device for this smoke test");

        let format = wgpu::TextureFormat::Rgb10a2Unorm;
        // Packed as Vulkan's A2B10G10R10_UNORM_PACK32: R in bits 0-9, G in
        // 10-19, B in 20-29, A in 30-31 of a little-endian u32.
        let packed: [u32; 6] = [
            0,                                           // black, transparent
            0x3ff,                                       // full red
            0x3ff << 10,                                 // full green
            0x3ff << 20,                                 // full blue
            0x3 << 30,                                   // opaque black
            (2 << 30) | (512 << 20) | (512 << 10) | 512, // mid gray, a=2/3
        ];
        let expected: [[u8; 4]; 6] = [
            [0, 0, 0, 0],
            [255, 0, 0, 0],
            [0, 255, 0, 0],
            [0, 0, 255, 0],
            [0, 0, 0, 255],
            [128, 128, 128, 170],
        ];
        let bytes: Vec<u8> = packed.iter().flat_map(|p| p.to_le_bytes()).collect();

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Rgb10a2 Pattern Texture"),
            size: wgpu::Extent3d {
                width: packed.len() as u32,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(packed.len() as u32 * 4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: packed.len() as u32,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let pixels = read_texture_rgba8(&device, &queue, &texture, format);

        assert_eq!(pixels, expected.concat());
    }

    /// Renders a fully deterministic scene at 1920x1080 - a solved cube,
    /// the app's default camera angle, the "revealed" sticker scale/face gap
    /// (spreading the 8 cells apart so all of them are visible instead of
    /// the collapsed default), and `elapsed_seconds` pinned to `0.0`
    /// (Elemental's materials, Fire's raymarched noise especially, are
    /// driven by it) - for golden-image comparison.
    ///
    /// The 4D rotation is a 90-degree turn in the XW plane rather than
    /// identity, so Fire (kind 3, the X=-1 face) lands in the near,
    /// unoccluded "center" slot (W=-1) instead of one of the six
    /// cross-arranged side cells, which - at this camera angle - it would
    /// otherwise render mostly hidden behind its neighbor.
    fn render_fixed_scene(theme: Theme) -> (Vec<u8>, u32, u32) {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("no GPU adapter available to run this test");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("failed to request a device for this test");

        let format = wgpu::TextureFormat::Rgba8Unorm;
        let (width, height) = (1920u32, 1080u32);
        let mut renderer = Renderer::new(
            &device,
            &queue,
            format,
            Rectangle {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
            },
            Size::new(width, height),
            UiControls {
                sticker_scale: SECONDARY_STICKER_SCALE,
                face_gap: SECONDARY_FACE_GAP,
                face_gap_4d: SECONDARY_FACE_GAP_4D,
                viewer_distance: crate::math::VIEWER_DISTANCE,
                render_mode: RenderMode::Standard,
                theme,
                ground_truth_debug_face: None,
            },
        );

        let mut camera = Camera {
            eye: nalgebra::Point3::new(0.0, 0.0, 15.0),
            target: nalgebra::Point3::new(0.0, 0.0, 0.0),
            up: nalgebra::Vector3::new(0.0, 1.0, 0.0),
        };
        CameraController::new(15.0).update_camera(&mut camera);
        let projection = Projection {
            aspect: width as f32 / height as f32,
            fovy: std::f32::consts::FRAC_PI_4,
            znear: 0.1,
            zfar: 100.0,
        };
        #[rustfmt::skip]
        let rotation_4d = nalgebra::Matrix4::new(
            0.0, 0.0, 0.0, -1.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
        );
        renderer.update_camera(&queue, &camera, &projection);
        renderer.update_light(&queue, &camera);
        renderer.update_instances(
            &queue,
            &rotation_4d,
            SECONDARY_STICKER_SCALE,
            SECONDARY_FACE_GAP,
            SECONDARY_FACE_GAP_4D,
            crate::math::VIEWER_DISTANCE,
            0.0,
        );

        let visible_faces = [true; 8];
        let depth_order = if theme == Theme::Elemental {
            let instances = generate_sticker_instances(&Hypercube::solved());
            crate::shader_widget::depth_draw_order(
                &instances,
                &visible_faces,
                &rotation_4d,
                &camera,
                true,
                SECONDARY_STICKER_SCALE,
                SECONDARY_FACE_GAP,
                SECONDARY_FACE_GAP_4D,
                crate::math::VIEWER_DISTANCE,
            )
        } else {
            Vec::new()
        };

        renderer.update_depth_batches(&queue, &depth_order);
        renderer.capture_frame(&device, &queue, &visible_faces)
    }

    /// Compares `rgba` against a golden PNG checked into
    /// `test_fixtures/golden/<name>.png`. A missing golden is bootstrapped -
    /// written, then failed, so a first run doesn't silently trust its own
    /// output as correct baseline. `UPDATE_GOLDEN=1` overwrites the golden
    /// with `rgba` unconditionally, for intentionally accepting a new
    /// baseline.
    ///
    /// Compares per-channel bytes against `tolerance` rather than requiring
    /// an exact match: raymarched noise and float rounding can differ
    /// subtly across GPU vendors and drivers even for identical inputs, so
    /// exact-match golden images are not realistically portable between
    /// machines. On mismatch, writes `<name>.actual.png` alongside the
    /// golden for diffing.
    fn assert_matches_golden(name: &str, rgba: &[u8], width: u32, height: u32, tolerance: u8) {
        let golden_path = golden_dir().join(format!("{name}.png"));

        if std::env::var_os("UPDATE_GOLDEN").is_some() {
            write_golden_png(&golden_path, rgba, width, height);
            return;
        }

        if !golden_path.exists() {
            write_golden_png(&golden_path, rgba, width, height);
            panic!(
                "no golden image at {golden_path:?}; wrote the current render as a baseline - \
                 inspect it, then re-run to confirm it matches before committing it"
            );
        }

        let golden = image::open(&golden_path)
            .unwrap_or_else(|err| panic!("failed to read golden image {golden_path:?}: {err}"))
            .to_rgba8();

        assert_eq!(
            (golden.width(), golden.height()),
            (width, height),
            "golden image {golden_path:?} is {}x{}, but the render is {width}x{height}",
            golden.width(),
            golden.height(),
        );

        let mut max_diff = 0u8;
        let mut diff_count = 0usize;
        for (golden_byte, actual_byte) in golden.as_raw().iter().zip(rgba) {
            let diff = golden_byte.abs_diff(*actual_byte);
            max_diff = max_diff.max(diff);
            if diff > tolerance {
                diff_count += 1;
            }
        }

        if diff_count > 0 {
            let actual_path = golden_dir().join(format!("{name}.actual.png"));
            write_golden_png(&actual_path, rgba, width, height);
            panic!(
                "render doesn't match golden {golden_path:?}: {diff_count} byte(s) differ by \
                 more than {tolerance} (max diff {max_diff}); wrote the mismatch to \
                 {actual_path:?} for comparison"
            );
        }
    }

    fn golden_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_fixtures/golden")
    }

    fn write_golden_png(path: &std::path::Path, rgba: &[u8], width: u32, height: u32) {
        let dir = path
            .parent()
            .expect("golden path must have a parent directory");
        std::fs::create_dir_all(dir).expect("failed to create golden image directory");
        image::RgbaImage::from_raw(width, height, rgba.to_vec())
            .expect("pixel buffer size mismatch")
            .save(path)
            .unwrap_or_else(|err| panic!("failed to write {path:?}: {err}"));
    }

    #[test]
    fn golden_solved_cube_classic_theme() {
        let (pixels, width, height) = render_fixed_scene(Theme::Classic);
        assert_matches_golden("solved_cube_classic", &pixels, width, height, 2);
    }

    #[test]
    fn golden_solved_cube_elemental_theme() {
        let (pixels, width, height) = render_fixed_scene(Theme::Elemental);
        assert_matches_golden("solved_cube_elemental", &pixels, width, height, 2);
    }

    /// Diagnostic, not a regression test: dumps a real rendered frame,
    /// captured at `Rgb10a2Unorm` (confirmed via a live run to be the
    /// surface format this HDR-capable Linux setup actually negotiates), to
    /// a PNG for visual inspection - to check `capture_frame`'s handling
    /// against a real render rather than the synthetic pattern the tests
    /// above use.
    #[test]
    #[ignore]
    fn dump_capture_at_hdr_surface_format() {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("no GPU adapter available");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("failed to request a device");

        let format = wgpu::TextureFormat::Rgb10a2Unorm;
        let mut renderer = Renderer::new(
            &device,
            &queue,
            format,
            Rectangle {
                x: 0.0,
                y: 0.0,
                width: 256.0,
                height: 256.0,
            },
            Size::new(256, 256),
            UiControls {
                sticker_scale: 0.5,
                face_gap: 0.1,
                face_gap_4d: 1.0,
                viewer_distance: crate::math::VIEWER_DISTANCE,
                render_mode: RenderMode::Standard,
                theme: Theme::Elemental,
                ground_truth_debug_face: None,
            },
        );

        renderer.update_depth_batches(
            &queue,
            &[
                vec![
                    crate::shader_widget::DepthLayer::Fire(0),
                    crate::shader_widget::DepthLayer::Ice(1),
                ],
                vec![crate::shader_widget::DepthLayer::Fire(2)],
            ],
        );
        let (pixels, width, height) = renderer.capture_frame(&device, &queue, &[true; 8]);

        let path = std::env::var("DUMP_PATH").unwrap_or_else(|_| "/tmp/capture_dump.png".into());
        image::RgbaImage::from_raw(width, height, pixels)
            .expect("pixel buffer size mismatch")
            .save(&path)
            .expect("failed to write diagnostic dump");
        eprintln!("wrote {path}");
    }
}
