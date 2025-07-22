//! GPU rendering system for the 4D hypercube visualization.
//!
//! This module handles all graphics rendering using wgpu, including GPU resource management,
//! render pipeline setup, and per-frame rendering of the hypercube instances.

use core::f32;

use iced::widget::shader::wgpu::{self, CommandEncoder, Device, Queue, TextureFormat, TextureView};
use iced::{Rectangle, Size};
use wgpu::util::DeviceExt;

use crate::RenderMode;
use crate::camera::{Camera, CameraUniform, Projection};
use crate::cube::{CUBE_VERTICES, FACE_CENTERS, FIXED_DIMS, Hypercube, VERTEX_NORMAL_INDICES};
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
    /// Current rendering mode
    current_render_mode: RenderMode,
    /// Buffer containing cube vertex positions
    vertex_buffer: wgpu::Buffer,
    /// Number of stickers (each generates 36 vertices)
    num_stickers: usize,
    /// Index buffers for each 4D face
    face_index_buffer: wgpu::Buffer,
    /// CPU-side camera uniform data
    camera_uniform: CameraUniform,
    /// GPU buffer containing camera matrices
    camera_buffer: wgpu::Buffer,
    /// CPU-side normals uniform data
    normals_uniform: NormalsUniform,
    /// GPU buffer containing normals data
    normals_buffer: wgpu::Buffer,
    /// CPU-side highlighting uniform data
    highlighting_uniform: HighlightingUniform,
    /// GPU buffer containing highlighting data
    highlighting_buffer: wgpu::Buffer,
    /// Bind group for main shader (transform, camera, light, normals, instances)
    main_bind_group: wgpu::BindGroup,
    /// Bind group for normal shader (transform, camera, normals, instances)
    normal_bind_group: wgpu::BindGroup,
    /// Bind group for debug shaders (transform, camera, instances)
    debug_bind_group: wgpu::BindGroup,
    /// Depth texture for z-buffering
    depth_texture: wgpu::Texture,
    /// Depth texture view for rendering
    depth_view: wgpu::TextureView,
    /// Transform uniform buffer for vertex shaders
    transform_buffer: wgpu::Buffer,
    /// Skybox bind group
    skybox_bind_group: wgpu::BindGroup,
}

/// Instance data for vertex shader - represents a sticker in 4D space
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct StickerInstance {
    /// 4D position of the sticker
    position_4d: [f32; 4],
    /// RGBA color of the sticker
    color: [f32; 4],
    /// Face ID (0-7) for this sticker
    face_id: u32,
    /// Padding for alignment
    _padding: [u32; 3],
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
    /// Spacing between faces
    face_spacing: f32,
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

/// Face data uniform - contains face centers and fixed dimensions for all 8 faces
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct FaceDataUniform {
    /// Face centers for all 8 faces (vec4<f32>)
    face_centers: [[f32; 4]; 8],
    /// Fixed dimensions for all 8 faces, only first index is used, rest are for padding
    fixed_dims: [[u32; 4]; 8],
}

/// Normals uniform data (8 faces × 6 normals each)
/// Note: WGSL vec3<f32> arrays have 16-byte alignment, so we pad to vec4<f32>
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct NormalsUniform {
    /// 48 normals (8 faces × 6 normals each), padded to vec4<f32> for alignment
    normals: [[f32; 4]; 48],
}

/// Highlighting uniform data for sticker hover effects
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct HighlightingUniform {
    /// Index of the hovered sticker (u32::MAX if none)
    hovered_sticker_index: u32,
    /// Highlighting intensity (0.0 to 1.0)
    highlight_intensity: f32,
    /// Padding for vec3 alignment
    _padding1: [f32; 2],
    /// Highlighting color (RGB)
    highlight_color: [f32; 3],
    /// Padding for alignment
    _padding2: f32,
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
            wgpu::ImageCopyTexture {
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
            wgpu::ImageDataLayout {
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

/// Generates instance data for the vertex shader from hypercube stickers
pub(crate) fn generate_sticker_instances(hypercube: &Hypercube) -> Vec<StickerInstance> {
    let mut instances = Vec::new();

    for (face_id, face) in hypercube.faces.iter().enumerate() {
        for sticker in &face.stickers {
            instances.push(StickerInstance {
                position_4d: [
                    sticker.position.x,
                    sticker.position.y,
                    sticker.position.z,
                    sticker.position.w,
                ],
                color: nalgebra::Vector4::from(sticker.color).into(),
                face_id: face_id as u32,
                _padding: [0; 3],
            });
        }
    }

    instances
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
    pub(crate) async fn new(
        device: &Device,
        queue: &Queue,
        format: TextureFormat,
        bounds: Rectangle<f32>,
        viewport_size: Size<u32>,
        hypercube: &Hypercube,
        ui_controls: UiControls,
    ) -> Self {
        let camera_uniform = CameraUniform::new();

        // Create light uniform with sun-like directional light
        let light_dir = nalgebra::Vector3::new(0.5, -1.0, 0.3).normalize();
        let light_uniform = LightUniform {
            direction: [light_dir.x, light_dir.y, light_dir.z], // Sun coming from upper right
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

        // Create face data uniform from constants
        let face_data_uniform = FaceDataUniform {
            face_centers: FACE_CENTERS.map(|v| [v.x, v.y, v.z, v.w]),
            fixed_dims: FIXED_DIMS.map(|d| [d as u32, 0, 0, 0]),
        };
        let face_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Face Data Buffer"),
            contents: bytemuck::cast_slice(&[face_data_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create initial normals uniform (will be updated later)
        let normals_uniform = NormalsUniform {
            normals: [[0.0; 4]; 48],
        };

        let normals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Normals Buffer"),
            contents: bytemuck::cast_slice(&[normals_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create initial highlighting uniform (no sticker highlighted)
        let highlighting_uniform = HighlightingUniform {
            hovered_sticker_index: u32::MAX, // No sticker highlighted
            highlight_intensity: 0.3,        // 30% intensity
            _padding1: [0.0; 2],
            highlight_color: [1.0, 1.0, 0.0], // Yellow highlight
            _padding2: 0.0,
        };

        let highlighting_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Highlighting Buffer"),
            contents: bytemuck::cast_slice(&[highlighting_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sticker_instances = generate_sticker_instances(hypercube);
        let num_stickers = sticker_instances.len();

        // Create instance buffer for sticker data
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&sticker_instances),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let mut vertices = CUBE_VERTICES;
        vertices
            .iter_mut()
            // TODO divide by puzzle size
            .for_each(|v| v.iter_mut().for_each(|i| *i /= 3.0));
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

        // Main shader bind group layout (transform, camera, light, face_data, normals, instances)
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::VERTEX,
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("Main Bind Group Layout"),
            });

        // Normal shader bind group layout (transform, camera, face_data, normals, instances)
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
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

        // Debug shaders bind group layout (transform, camera, face_data, instances)
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
                ],
                label: Some("Debug Bind Group Layout"),
            });

        // Create transform uniform buffer with initial slider values
        let transform_data = Transform4D {
            rotation_matrix: nalgebra::Matrix4::identity().into(),
            viewer_distance: 3.0,
            sticker_scale: ui_controls.sticker_scale,
            face_spacing: ui_controls.face_scale,
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
                    resource: face_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: normals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: highlighting_buffer.as_entire_binding(),
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
                    resource: face_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: normals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
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
                    resource: face_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: instance_buffer.as_entire_binding(),
                },
            ],
            label: Some("Debug Bind Group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shader.wgsl").into()),
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
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_sky",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_sky",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/normal_shader.wgsl").into()),
        });

        let normal_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Normal Pipeline"),
            layout: Some(&normal_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &normal_shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &normal_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/depth_shader.wgsl").into()),
        });

        let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Depth Pipeline"),
            layout: Some(&debug_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &depth_shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &depth_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
            current_render_mode: ui_controls.render_mode,
            vertex_buffer,
            face_index_buffer,
            num_stickers,
            camera_uniform,
            camera_buffer,
            normals_uniform,
            normals_buffer,
            highlighting_uniform,
            highlighting_buffer,
            main_bind_group,
            normal_bind_group,
            debug_bind_group,
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
    /// * `face_scale` - Scale factor for face spacing (from face scale slider)
    pub(crate) fn update_instances(
        &mut self,
        queue: &Queue,
        rotation_4d: &nalgebra::Matrix4<f32>,
        sticker_scale: f32,
        face_scale: f32,
    ) {
        // Update transform uniform
        let transform_data = Transform4D {
            rotation_matrix: (*rotation_4d).into(),
            viewer_distance: 3.0,
            sticker_scale,
            face_spacing: face_scale,
            _padding: 0.0,
        };
        queue.write_buffer(
            &self.transform_buffer,
            0,
            bytemuck::cast_slice(&[transform_data]),
        );
    }

    /// Updates the normals uniform buffer with pre-calculated normals.
    ///
    /// # Arguments
    /// * `queue` - GPU command queue for buffer updates
    /// * `normals` - Pre-calculated normals (48 vec3s: 8 faces × 6 normals each)
    pub(crate) fn update_normals(&mut self, queue: &Queue, normals: &[nalgebra::Vector3<f32>]) {
        // Convert Vec<Vector3> to [[f32; 4]; 48] (pad to vec4 for WGSL alignment)
        for (i, normal) in normals.iter().enumerate().take(48) {
            self.normals_uniform.normals[i] = [normal.x, normal.y, normal.z, 0.0];
        }

        queue.write_buffer(
            &self.normals_buffer,
            0,
            bytemuck::cast_slice(&[self.normals_uniform]),
        );
    }

    pub(crate) fn update_indices(&mut self, queue: &Queue, indices: &[u16]) {
        queue.write_buffer(&self.face_index_buffer, 0, bytemuck::cast_slice(indices));
    }

    /// Updates the highlighting uniform buffer with the currently hovered sticker.
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

        queue.write_buffer(
            &self.highlighting_buffer,
            0,
            bytemuck::cast_slice(&[self.highlighting_uniform]),
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
    pub(crate) fn render(&self, encoder: &mut CommandEncoder, target: &TextureView) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Don't clear, we already cleared selectively
                    store: wgpu::StoreOp::Store,
                },
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

        // Draw all cubes using instanced rendering (36 vertices per cube, num_stickers instances)
        render_pass.draw_indexed(
            0..VERTEX_NORMAL_INDICES.len() as u32 * 8,
            0,
            0..self.num_stickers as u32,
        );
    }
}
