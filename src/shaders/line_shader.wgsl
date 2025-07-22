// Cylindrical line rendering shader for mouse ray visualization

// Vertex input for cylinder geometry
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

// Vertex output / Fragment input
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
}

// Camera uniform
struct CameraUniform {
    view_proj: mat4x4<f32>,
}

// Line transform uniform
struct LineTransform {
    model_matrix: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var<uniform> line_transform: LineTransform;

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Transform vertex position to world space
    let world_position = line_transform.model_matrix * vec4<f32>(model.position, 1.0);
    
    // Transform normal to world space (using upper 3x3 of model matrix)
    let normal_matrix = mat3x3<f32>(
        line_transform.model_matrix[0].xyz,
        line_transform.model_matrix[1].xyz,
        line_transform.model_matrix[2].xyz
    );
    out.world_normal = normalize(normal_matrix * model.normal);
    
    out.clip_position = camera.view_proj * world_position;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple lighting
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.5));
    let diffuse = max(dot(in.world_normal, light_dir), 0.2);
    
    let red_color = vec3<f32>(1.0, 0.0, 0.0);
    return vec4<f32>(red_color * diffuse, 1.0);
}