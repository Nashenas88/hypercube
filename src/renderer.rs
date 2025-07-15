//! GPU rendering system for the 4D hypercube visualization.
//!
//! This module handles all graphics rendering using wgpu, including GPU resource management,
//! render pipeline setup, and per-frame rendering of the hypercube instances.

use core::f32;

use iced::widget::shader::wgpu::{self, CommandEncoder, Device, Queue, TextureFormat, TextureView};
use iced::{Rectangle, Size};
use wgpu::util::DeviceExt;

use crate::camera::{Camera, CameraUniform, Projection};
use crate::cube::{Hypercube, INDICES};

/// GPU renderer for the hypercube visualization.
///
/// Manages all graphics resources including buffers, textures, pipelines, and rendering state.
/// Uses instanced rendering to efficiently draw all 216 hypercube stickers.
#[derive(Debug)]
pub(crate) struct Renderer {
    /// Bounds within the viewport to render to.
    bounds: Rectangle<f32>,
    /// Graphics pipeline for cube rendering
    render_pipeline: wgpu::RenderPipeline,
    /// Buffer containing cube index data for all cubes
    index_buffer: wgpu::Buffer,
    /// Number of indices in the index buffer
    num_indices: u32,
    /// Buffer containing vertex data from compute shader and used for rendering
    vertex_buffer: wgpu::Buffer,
    /// Number of stickers (each generates 8 vertices)
    num_stickers: usize,
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
    /// Clear quad render pipeline
    clear_pipeline: wgpu::RenderPipeline,
    /// Clear quad vertex buffer
    clear_vertex_buffer: wgpu::Buffer,
    /// Clear quad index buffer
    clear_index_buffer: wgpu::Buffer,
    /// Compute pipeline for 4D transformations
    compute_pipeline: wgpu::ComputePipeline,
    /// Transform uniform buffer for compute shader
    transform_buffer: wgpu::Buffer,
    /// Bind group for compute shader
    compute_bind_group: wgpu::BindGroup,
}

/// GPU-compatible vertex data output from compute shader.
///
/// Each sticker generates 8 vertices that are already transformed and projected.
/// Uses 4-component position for proper GPU alignment.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct VertexRaw {
    /// 3D position after 4D projection (4th component unused but needed for alignment)
    position: [f32; 4],
    /// RGBA color values
    color: [f32; 4],
}

/// Input data for compute shader - represents a sticker in 4D space
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct StickerInput {
    /// 4D position of the sticker
    position_4d: [f32; 4],
    /// RGBA color of the sticker
    color: [f32; 4],
    /// 4D position of the face center this sticker belongs to
    face_center_4d: [f32; 4],
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

/// Generates input data for the compute shader from hypercube stickers
pub(crate) fn generate_sticker_inputs(hypercube: &Hypercube) -> Vec<StickerInput> {
    let mut inputs = Vec::new();

    for face in &hypercube.faces {
        for sticker in &face.stickers {
            inputs.push(StickerInput {
                position_4d: [
                    sticker.position.x,
                    sticker.position.y,
                    sticker.position.z,
                    sticker.position.w,
                ],
                color: nalgebra::Vector4::from(sticker.color).into(),
                face_center_4d: [face.center.x, face.center.y, face.center.z, face.center.w],
            });
        }
    }

    inputs
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
        format: TextureFormat,
        bounds: Rectangle<f32>,
        viewport_size: Size<u32>,
        hypercube: &Hypercube,
        sticker_scale: f32,
        face_scale: f32,
    ) -> Self {
        let camera_uniform = CameraUniform::new();

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

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("Camera Bind Group Layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("Camera Bind Group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shader.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<VertexRaw>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x4,  // position (4-component for alignment)
                        1 => Float32x4,  // color
                    ],
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
                cull_mode: None, // Disable culling to see all faces
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

        let sticker_inputs = generate_sticker_inputs(hypercube);
        let num_stickers = sticker_inputs.len();

        // Generate indices for all cubes (each cube needs its own set of indices)
        let mut all_indices = Vec::new();
        for cube_idx in 0..num_stickers {
            let vertex_offset = (cube_idx * 8) as u16;
            for &index in INDICES {
                all_indices.push(index + vertex_offset);
            }
        }

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&all_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = all_indices.len() as u32;

        // Create single vertex buffer used by both compute shader and vertex shader
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (num_stickers * 8 * std::mem::size_of::<VertexRaw>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        // Create clear quad geometry (full-screen quad in NDC)
        let clear_vertices: &[[f32; 2]] = &[
            [-1.0, -1.0], // bottom-left
            [1.0, -1.0],  // bottom-right
            [1.0, 1.0],   // top-right
            [-1.0, 1.0],  // top-left
        ];
        let clear_indices: &[u16] = &[0, 1, 2, 0, 2, 3];

        let clear_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Clear Vertex Buffer"),
            contents: bytemuck::cast_slice(clear_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let clear_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Clear Index Buffer"),
            contents: bytemuck::cast_slice(clear_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Create clear shader
        let clear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Clear Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/clear.wgsl").into()),
        });

        // Create clear pipeline
        let clear_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Clear Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let clear_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Clear Pipeline"),
            layout: Some(&clear_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &clear_shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &clear_shader,
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
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Set up compute pipeline for 4D transformations
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/compute.wgsl").into()),
        });

        // Create input buffer for compute shader
        let compute_input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Compute Input Buffer"),
            contents: bytemuck::cast_slice(&sticker_inputs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Create transform uniform buffer with initial slider values
        let transform_data = Transform4D {
            rotation_matrix: nalgebra::Matrix4::identity().into(),
            viewer_distance: 3.0,
            sticker_scale,
            face_spacing: face_scale,
            _padding: 0.0,
        };
        let transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transform Buffer"),
            contents: bytemuck::cast_slice(&[transform_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create compute bind group layout
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("Compute Bind Group Layout"),
            });

        // Create compute bind group
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: transform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: compute_input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vertex_buffer.as_entire_binding(),
                },
            ],
            label: Some("Compute Bind Group"),
        });

        // Create compute pipeline
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        });

        Self {
            bounds,
            render_pipeline,
            index_buffer,
            num_indices,
            vertex_buffer,
            num_stickers,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            depth_texture,
            depth_view,
            clear_pipeline,
            clear_vertex_buffer,
            clear_index_buffer,
            compute_pipeline,
            transform_buffer,
            compute_bind_group,
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

    pub(crate) fn compute_instances(&self, encoder: &mut CommandEncoder) {
        // Run compute shader
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);

            // Dispatch with workgroups of 64, covering all stickers
            let workgroup_size = 64;
            let num_workgroups = (self.num_stickers as u32).div_ceil(workgroup_size);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // No copy needed - compute shader writes directly to vertex buffer
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
        // First pass: Clear only the bounds area with black quad
        {
            let mut clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Don't clear the entire target
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            clear_pass.set_viewport(
                self.bounds.x,
                self.bounds.y,
                self.bounds.width,
                self.bounds.height,
                0.0,
                1.0,
            );
            clear_pass.set_pipeline(&self.clear_pipeline);
            clear_pass.set_vertex_buffer(0, self.clear_vertex_buffer.slice(..));
            clear_pass
                .set_index_buffer(self.clear_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            clear_pass.draw_indexed(0..6, 0, 0..1);
        }

        // Second pass: Render the hypercube within bounds
        {
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
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            // Draw all cubes at once with properly offset indices
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }
    }
}
