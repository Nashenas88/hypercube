use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    event::{Event, WindowEvent, MouseButton, ElementState, DeviceEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use nalgebra::{Matrix4, Point3, Vector3};

mod cube;
use cube::{Color, Hypercube, INDICES, VERTICES, project_4d_to_3d};

// Constants for sticker layout
const STICKER_SCALE: f32 = 0.8;
const STICKER_SPACING: f32 = 1.2;
const SIDE_SPACING: f32 = 8.0;
const VIEWER_DISTANCE_4D: f32 = 3.0;
const MOUSE_SENSITIVITY: f32 = 0.5;
const PROJECTION_FOVY: f32 = 45.0;

struct Camera {
    eye: Point3<f32>,
    target: Point3<f32>,
    up: Vector3<f32>,
}

impl Camera {
    fn build_view_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(&self.eye, &self.target, &self.up)
    }
}

struct CameraController {
    distance: f32,
    yaw: f32,
    pitch: f32,
    is_right_mouse_pressed: bool,
    last_mouse_pos: Option<(f32, f32)>,
}

impl CameraController {
    fn new(distance: f32) -> Self {
        Self {
            distance,
            yaw: 0.0,
            pitch: 0.0,
            is_right_mouse_pressed: false,
            last_mouse_pos: None,
        }
    }

    fn update_camera(&self, camera: &mut Camera) {
        let yaw_rad = self.yaw.to_radians();
        let pitch_rad = self.pitch.to_radians();
        
        let x = self.distance * pitch_rad.cos() * yaw_rad.sin();
        let y = self.distance * pitch_rad.sin();
        let z = self.distance * pitch_rad.cos() * yaw_rad.cos();
        
        camera.eye = Point3::new(x, y, z);
        camera.target = Point3::new(0.0, 0.0, 0.0);
        camera.up = Vector3::new(0.0, 1.0, 0.0);
    }

    fn process_mouse_input(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Right {
            self.is_right_mouse_pressed = state == ElementState::Pressed;
            if !self.is_right_mouse_pressed {
                self.last_mouse_pos = None;
            }
        }
    }

    fn process_mouse_motion(&mut self, delta_x: f32, delta_y: f32) {
        if self.is_right_mouse_pressed {
            self.yaw += delta_x * MOUSE_SENSITIVITY;
            self.pitch -= delta_y * MOUSE_SENSITIVITY;
            
            // Clamp pitch to prevent camera flipping
            self.pitch = self.pitch.clamp(-89.0, 89.0);
        }
    }
}

struct Projection {
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Projection {
    fn build_projection_matrix(&self) -> Matrix4<f32> {
        nalgebra::Matrix4::new_perspective(
            self.aspect,
            self.fovy,
            self.znear,
            self.zfar,
        )
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_proj: nalgebra::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera, projection: &Projection) {
        self.view_proj = (projection.build_projection_matrix() * camera.build_view_matrix()).into();
    }
}

struct App {
    hypercube: Hypercube,
    camera: Camera,
    camera_controller: CameraController,
    projection: Projection,
}

impl App {
    fn new(window_width: u32, window_height: u32) -> Self {
        let hypercube = Hypercube::new();
        
        let mut camera = Camera {
            eye: Point3::new(0.0, 0.0, 15.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
        };
        
        let mut camera_controller = CameraController::new(15.0);
        camera_controller.update_camera(&mut camera);

        let projection = Projection {
            aspect: window_width as f32 / window_height as f32,
            fovy: PROJECTION_FOVY,
            znear: 0.1,
            zfar: 100.0,
        };

        Self {
            hypercube,
            camera,
            camera_controller,
            projection,
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::MouseInput { button, state, .. } => {
                self.camera_controller.process_mouse_input(*button, *state);
                true
            }
            _ => false,
        }
    }

    fn device_input(&mut self, event: &DeviceEvent) -> bool {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                self.camera_controller.process_mouse_motion(delta.0 as f32, delta.1 as f32);
                true
            }
            _ => false,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.projection.aspect = new_size.width as f32 / new_size.height as f32;
        }
    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        winit::window::WindowBuilder::new()
            .with_title("Hypercube")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .build(&event_loop)
            .unwrap(),
    );

    let window_size = window.inner_size();
    let mut app = App::new(window_size.width, window_size.height);
    let mut renderer = pollster::block_on(Renderer::new(window.clone(), &app.hypercube));

    event_loop
        .run(move |event, elwt| match event {
            Event::WindowEvent {
                window_id,
                event,
            } if window_id == renderer.window().id() => {
                if !app.input(&event) {
                    match event {
                        WindowEvent::CloseRequested => {
                            elwt.exit();
                        }
                        WindowEvent::Resized(physical_size) => {
                            app.resize(physical_size);
                            renderer.resize(physical_size);
                        }
                        WindowEvent::RedrawRequested => {
                            app.update();
                            // Note that this is not guaranteed to be called every frame.
                            // We should probably take a look at that later.
                            match renderer.render(&app.camera, &app.projection) {
                                Ok(_) => {},
                                // Reconfigure the surface if lost
                                Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size),
                                // The system is out of memory, we should probably quit
                                Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                                // All other errors (Outdated, Timeout) should be resolved by the next frame
                                Err(e) => eprintln!("{:?}", e),
                            }
                        }
                        _ => {}
                    }
                }
            },
            Event::DeviceEvent { event, .. } => {
                app.device_input(&event);
            }
            Event::AboutToWait => {
                renderer.window().request_redraw();
            }
            _ => {}
        })
        .unwrap();
}

struct Renderer<'a> {
    window: Arc<Window>,
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    instance_buffer: wgpu::Buffer,
    num_instances: u32,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    color: [f32; 4],
}

struct Instance {
    position: nalgebra::Vector3<f32>,
    color: nalgebra::Vector4<f32>,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        let scale_matrix = nalgebra::Matrix4::new_scaling(STICKER_SCALE);
        let translation_matrix = nalgebra::Matrix4::new_translation(&self.position);
        InstanceRaw {
            model: (translation_matrix * scale_matrix).into(),
            color: self.color.into(),
        }
    }
}

fn generate_instances(hypercube: &Hypercube) -> Vec<Instance> {
    let mut instances = Vec::new();
    
    for side in &hypercube.sides {
        for sticker in &side.stickers {
            // Project 4D position to 3D
            let projected_3d = project_4d_to_3d(sticker.position * STICKER_SPACING, VIEWER_DISTANCE_4D);
            
            instances.push(Instance {
                position: projected_3d,
                color: nalgebra::Vector4::from(sticker.color),
            });
        }
    }
    
    instances
}

impl<'a> Renderer<'a> {
    async fn new(window: Arc<Window>, hypercube: &Hypercube) -> Self {
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

        let mut camera_uniform = CameraUniform::new();

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
            depth_stencil: None,
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

        // Generate instances from the provided hypercube
        let instances = generate_instances(hypercube);
        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
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
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn render(&mut self, camera: &Camera, projection: &Projection) -> Result<(), wgpu::SurfaceError> {
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
                depth_stencil_attachment: None,
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