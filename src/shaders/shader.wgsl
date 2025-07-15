// Vertex shader

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

struct LightUniform {
    direction: vec3<f32>,
    _padding1: f32,
    color: vec3<f32>,
    _padding2: f32,
    ambient: vec3<f32>,
    _padding3: f32,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var<uniform> light: LightUniform;


struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
}


@vertex
fn vs_main(
    @location(0) vertex_position: vec4<f32>,
    @location(1) vertex_color: vec4<f32>,
    @location(2) vertex_normal: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Extract visibility flag from position.w component
    let visibility = vertex_position.w;
    
    // If not visible (culled), move vertex off-screen
    if (visibility < 0.5) {
        out.clip_position = vec4<f32>(0.0, 0.0, -1.0, 1.0); // Behind camera
        out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0); // Transparent
        out.world_position = vec3<f32>(0.0, 0.0, 0.0);
        out.world_normal = vec3<f32>(0.0, 0.0, 1.0);
    } else {
        // Use xyz components for actual position
        let position_3d = vec4<f32>(vertex_position.x, vertex_position.y, vertex_position.z, 1.0);
        out.clip_position = camera.view_proj * position_3d;
        out.color = vertex_color;
        out.world_position = vertex_position.xyz;
        
        // Use the normal directly from vertex data
        out.world_normal = vertex_normal.xyz;
    }
    
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Normalize the normal vector
    let normal = normalize(in.world_normal);
    
    // Calculate light direction (directional light)
    let light_dir = normalize(-light.direction);
    
    // Calculate view direction (camera position is at origin in view space)
    let view_dir = normalize(-in.world_position);
    
    // Ambient lighting
    let ambient = light.ambient * in.color.rgb;
    
    // Diffuse lighting (Lambertian)
    let diffuse_strength = max(dot(normal, light_dir), 0.0);
    let diffuse = diffuse_strength * light.color * in.color.rgb;
    
    // Specular lighting (Blinn-Phong)
    let half_dir = normalize(light_dir + view_dir);
    let specular_strength = pow(max(dot(normal, half_dir), 0.0), 32.0);
    let specular = specular_strength * light.color * 0.3; // Reduced specular intensity
    
    // Combine all lighting components
    let final_color = ambient + diffuse + specular;
    
    return vec4<f32>(final_color, in.color.a);
}