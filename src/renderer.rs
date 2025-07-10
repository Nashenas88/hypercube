//! GPU rendering system for the 4D hypercube visualization.
//! 
//! This module handles all graphics rendering using wgpu, including GPU resource management,
//! render pipeline setup, and per-frame rendering of the hypercube instances.

use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;
use nalgebra::Vector3;

use crate::camera::{Camera, Projection, CameraUniform};
use crate::cube::{Hypercube, VERTICES, INDICES};
use crate::math::generate_instances;

/// GPU renderer for the hypercube visualization.
/// 
/// Manages all graphics resources including buffers, textures, pipelines, and rendering state.
/// Uses instanced rendering to efficiently draw all 216 hypercube stickers.
pub struct Renderer<'a> {
    /// Reference to the window for surface operations
    window: Arc<Window>,
    /// wgpu surface for presenting rendered frames
    surface: wgpu::Surface<'a>,
    /// GPU device for creating resources
    device: wgpu::Device,
    /// Command queue for GPU operations
    queue: wgpu::Queue,
    /// Surface configuration for presentation
    config: wgpu::SurfaceConfiguration,
    /// Current window size for aspect ratio calculations
    pub size: winit::dpi::PhysicalSize<u32>,
    /// Graphics pipeline for cube rendering
    render_pipeline: wgpu::RenderPipeline,
    /// Buffer containing cube vertex data
    vertex_buffer: wgpu::Buffer,
    /// Buffer containing cube index data
    index_buffer: wgpu::Buffer,
    /// Number of indices in the index buffer
    num_indices: u32,
    /// Buffer containing per-instance transformation data
    instance_buffer: wgpu::Buffer,
    /// Number of instances to render
    num_instances: u32,
    /// CPU-side camera uniform data
    camera_uniform: CameraUniform,
    /// GPU buffer containing camera matrices
    camera_buffer: wgpu::Buffer,
    /// Bind group for camera uniform buffer
    camera_bind_group: wgpu::BindGroup,
    /// Depth texture for z-buffering
    depth_texture: wgpu::Texture,
    /// Depth texture view for rendering
    depth_view: wgpu::TextureView,
}

/// GPU-compatible instance data for rendering individual cubes.
/// 
/// Contains transformation matrix and color data that gets uploaded to the GPU
/// for instanced rendering of hypercube stickers.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    /// 4x4 model transformation matrix
    model: [[f32; 4]; 4],
    /// RGBA color values
    color: [f32; 4],
}

/// CPU-side instance data for a single hypercube sticker.
/// 
/// Contains position and color information that gets converted to GPU format.
pub struct Instance {
    /// 3D position of the sticker after 4D projection
    pub position: Vector3<f32>,
    /// RGBA color of the sticker
    pub color: nalgebra::Vector4<f32>,
}

impl Instance {
    /// Converts CPU instance data to GPU-compatible format.
    /// 
    /// Creates transformation matrix and formats color data for upload to GPU.
    /// 
    /// # Returns
    /// GPU-compatible instance data ready for rendering
    pub fn to_raw(&self) -> InstanceRaw {
        const STICKER_SCALE: f32 = 0.8;
        let scale_matrix = nalgebra::Matrix4::new_scaling(STICKER_SCALE);
        let translation_matrix = nalgebra::Matrix4::new_translation(&self.position);
        InstanceRaw {
            model: (translation_matrix * scale_matrix).into(),
            color: self.color.into(),
        }
    }
}

impl<'a> Renderer<'a> {
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
    pub async fn new(window: Arc<Window>, hypercube: &Hypercube) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let camera_uniform = CameraUniform::new();

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
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

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                }
            ],
            label: Some("Camera Bind Group Layout"),
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }
            ],
            label: Some("Camera Bind Group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![
                            1 => Float32x4,
                            2 => Float32x4, 
                            3 => Float32x4,
                            4 => Float32x4,
                            5 => Float32x4,
                        ],
                    }
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
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

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = INDICES.len() as u32;

        let identity = nalgebra::Matrix4::identity();
        let instances = generate_instances(hypercube, &identity);
        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let num_instances = instances.len() as u32;

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            instance_buffer,
            num_instances,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            depth_texture,
            depth_view,
        }
    }

    /// Returns a reference to the window being rendered to.
    /// 
    /// # Returns
    /// Reference to the window for event handling and queries
    pub fn window(&self) -> &Window {
        &self.window
    }

    /// Handles window resize events by updating surface and depth buffer.
    /// 
    /// Recreates size-dependent resources like the depth texture when the window
    /// size changes.
    /// 
    /// # Arguments
    /// * `new_size` - New window dimensions in pixels
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            
            self.depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
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
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            
            self.depth_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        }
    }

    /// Updates the instance buffer with current hypercube state.
    /// 
    /// Regenerates all instance data based on current 4D rotation and uploads
    /// to GPU for the next frame.
    /// 
    /// # Arguments
    /// * `hypercube` - Current hypercube state
    /// * `rotation_4d` - Current 4D rotation matrix
    pub fn update_instances(&mut self, hypercube: &Hypercube, rotation_4d: &nalgebra::Matrix4<f32>) {
        let instances = generate_instances(hypercube, rotation_4d);
        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instance_data));
    }

    /// Renders a single frame of the hypercube visualization.
    /// 
    /// Updates camera uniforms, acquires surface texture, and draws all instances
    /// with proper depth testing.
    /// 
    /// # Arguments
    /// * `camera` - Current camera state for view matrix
    /// * `projection` - Current projection parameters
    pub fn render(&mut self, camera: &Camera, projection: &Projection) -> Result<(), wgpu::SurfaceError> {
        self.camera_uniform.update_view_proj(camera, projection);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
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

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..self.num_instances);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}