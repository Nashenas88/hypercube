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
use crate::math::{BASE_STICKER_SIZE, VIEWER_DISTANCE};
use crate::piece::{FACET_TABLE, Hypercube, StickerInstance, generate_sticker_instances};
use crate::shader_widget::UiControls;

/// GPU renderer for the hypercube visualization.
///
/// Manages all graphics resources including buffers, textures, pipelines, and rendering state.
/// Uses instanced rendering to efficiently draw all 216 hypercube stickers.
#[derive(Debug)]
pub(crate) struct Renderer {
    /// Bounds within the viewport to render to.
    bounds: Rectangle<f32>,
    /// Vertex buffer for sky quad
    sky_vertex_buffer: wgpu::Buffer,
    /// Index buffer for sky quad
    sky_index_buffer: wgpu::Buffer,
    /// Graphics pipeline for sky rendering
    sky_pipeline: wgpu::RenderPipeline,
    /// Graphics pipeline for standard rendering
    render_pipeline: wgpu::RenderPipeline,
    /// Graphics pipeline for normal visualization
    normal_pipeline: wgpu::RenderPipeline,
    /// Graphics pipeline for depth visualization
    depth_pipeline: wgpu::RenderPipeline,
    /// Graphics pipeline for debug AABB rendering
    debug_pipeline: wgpu::RenderPipeline,
    /// Current rendering mode
    current_render_mode: RenderMode,
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
    /// Bind group for main shader (transform, camera, light, normals, instances)
    main_bind_group: wgpu::BindGroup,
    /// Bind group for normal shader (transform, camera, normals, instances)
    normal_bind_group: wgpu::BindGroup,
    /// Bind group for debug shaders (transform, camera, instances)
    debug_bind_group: wgpu::BindGroup,
    /// Bind group for debug AABB rendering (camera, debug_instances)
    debug_aabb_bind_group: wgpu::BindGroup,
    /// Depth texture for z-buffering
    depth_texture: wgpu::Texture,
    /// Depth texture view for rendering
    depth_view: wgpu::TextureView,
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
    /// Padding for alignment
    _padding: f32,
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

/// Highlighting uniform data for sticker hover effects
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct HighlightingUniform {
    /// Index of the hovered sticker (u32::MAX if none)
    hovered_sticker_index: u32,
    /// `Hypercube::pieces` slot of the hovered sticker's piece (u32::MAX if none)
    hovered_piece_slot: u32,
    /// Padding for vec4 alignment
    _padding: [u32; 2],
    /// Color and intensity (in `a`) for the exact hovered sticker
    highlight_color: [f32; 4],
    /// Color and intensity (in `a`) for the rest of the hovered piece's stickers
    piece_highlight_color: [f32; 4],
}

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
        // new field. validate
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

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: viewport_size.width,
                height: viewport_size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
            _padding: [0; 2],
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

        let mut vertices = CUBE_VERTICES;
        vertices
            .iter_mut()
            // TODO divide by puzzle size
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

        // Main shader bind group layout (transform, camera, light, instances, highlighting, piece_slots)
        let main_bind_group_layout =
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
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
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

        // Create transform uniform buffer with initial slider values
        let transform_data = Transform4D {
            rotation_matrix: nalgebra::Matrix4::identity().into(),
            viewer_distance: VIEWER_DISTANCE,
            sticker_scale: ui_controls.sticker_scale,
            face_gap: ui_controls.face_gap,
            _padding: 0.0,
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
                    resource: light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: highlighting_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: piece_slot_buffer.as_entire_binding(),
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

        let mut composer = Composer::default();
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source: include_str!("shaders/math4d.wgsl"),
                file_path: "shaders/math4d.wgsl",
                ..Default::default()
            })
            .expect("shaders/math4d.wgsl failed to compose");
        let mut compose_shader =
            |source: &str, file_path: &str| match composer.make_naga_module(NagaModuleDescriptor {
                source,
                file_path,
                ..Default::default()
            }) {
                Ok(module) => module,
                Err(err) => panic!("{}", err.emit_to_string(&composer)),
            };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(compose_shader(
                include_str!("shaders/shader.wgsl"),
                "shaders/shader.wgsl",
            ))),
        });

        let sky_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky Pipeline Layout"),
            bind_group_layouts: &[&skybox_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&main_bind_group_layout],
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
                module: &shader,
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
                module: &shader,
                entry_point: Some("fs_sky"),
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

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
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
                module: &shader,
                entry_point: Some("fs_main"),
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

        Self {
            bounds,
            sky_vertex_buffer,
            sky_index_buffer,
            sky_pipeline,
            render_pipeline,
            normal_pipeline,
            depth_pipeline,
            debug_pipeline,
            current_render_mode: ui_controls.render_mode,
            vertex_buffer,
            face_index_buffer,
            last_indices_generation: None,
            last_sticker_generation: None,
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
            main_bind_group,
            normal_bind_group,
            debug_bind_group,
            debug_aabb_bind_group,
            depth_texture,
            depth_view,
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
            self.depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            self.depth_view = self
                .depth_texture
                .create_view(&wgpu::TextureViewDescriptor::default());
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

    /// Updates the instance buffer using compute shaders for 4D transformations.
    ///
    /// Runs the 4D transformation compute shader and copies the result to the instance buffer.
    ///
    /// # Arguments
    /// * `queue` - GPU queue for submitting commands
    /// * `rotation_4d` - Current 4D rotation matrix
    /// * `sticker_scale` - Scale factor for individual stickers (from sticker scale slider)
    /// * `face_gap` - 3D distance to push each face outward (from face gap slider)
    pub(crate) fn update_instances(
        &mut self,
        queue: &Queue,
        rotation_4d: &nalgebra::Matrix4<f32>,
        sticker_scale: f32,
        face_gap: f32,
    ) {
        // Update transform uniform
        let transform_data = Transform4D {
            rotation_matrix: (*rotation_4d).into(),
            viewer_distance: VIEWER_DISTANCE,
            sticker_scale,
            face_gap,
            _padding: 0.0,
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
    /// last generation uploaded, mirroring `update_indices`.
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
        self.last_sticker_generation = Some(generation);
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

    /// Renders a single frame of the hypercube visualization.
    ///
    /// Updates camera uniforms, acquires surface texture, and draws all instances
    /// with proper depth testing.
    ///
    /// # Arguments
    /// * `camera` - Current camera state for view matrix
    /// * `projection` - Current projection parameters
    /// * `visible_faces` - Per-`face_id` visibility (see `math::visible_faces`);
    ///   faces marked invisible are skipped entirely, issuing no draw call
    ///   and no vertex-shader invocations for their 27 instances.
    pub(crate) fn render(
        &self,
        encoder: &mut CommandEncoder,
        target: &TextureView,
        visible_faces: &[bool; 8],
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Don't clear, we already cleared selectively
                    store: wgpu::StoreOp::Store,
                },
                // TODO new field. validate
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
        render_pass.set_index_buffer(self.sky_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..6, 0, 0..1);

        // Then render the hypercube
        let (pipeline, bind_group) = match self.current_render_mode {
            RenderMode::Standard => (&self.render_pipeline, &self.main_bind_group),
            RenderMode::Normals => (&self.normal_pipeline, &self.normal_bind_group),
            RenderMode::Depth => (&self.depth_pipeline, &self.debug_bind_group),
        };
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.face_index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        // One draw per 4D face: `face_index_buffer` holds 8 winding-corrected
        // 36-index chunks (one per face_id, see `calculate_indices`), and
        // `FACET_TABLE` (piece.rs) is built in matching face-major blocks of
        // 27, so chunk N only ever reaches the instances it was computed
        // for. A single draw over all 288 indices and 216 instances would
        // feed every chunk to every instance, relying on backface culling to
        // silently discard the wrong ones (the bug perf_improvements.md #1
        // describes); slicing per face keeps culling meaningful instead.
        // Faces `visible_faces` marks invisible skip the draw call entirely,
        // rather than issuing it and relying on the vertex shader to cull.
        let indices_per_face = VERTEX_NORMAL_INDICES.len() as u32;
        let facets_per_face = self.num_stickers as u32 / 8;
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
                // TODO new field. validate
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
                render_mode: RenderMode::Standard,
            },
        )
    }
}
