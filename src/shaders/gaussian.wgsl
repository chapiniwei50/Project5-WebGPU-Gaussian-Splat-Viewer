struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_pos: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) conic: vec3<f32>,
};

struct Splat {
    v_0: u32,
    v_1: u32,
    pos: u32,
    color_0: u32,
    color_1: u32,
};

@group(0) @binding(2) var<storage, read> points_2d: array<Splat>;
@group(1) @binding(4) var<storage, read> indices: array<u32>;

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32, 
           @builtin(instance_index) in_instance_index: u32) -> VertexOutput {
    var out: VertexOutput;

    let vertex = points_2d[indices[in_instance_index]];
    let v1 = unpack2x16float(vertex.v_0);
    let v2 = unpack2x16float(vertex.v_1);
    let v_center = unpack2x16float(vertex.pos);

    let x = f32(in_vertex_index % 2u == 0u) * 2.0 - 1.0;
    let y = f32(in_vertex_index < 2u) * 2.0 - 1.0;

    let position = vec2<f32>(x, y) * 2.3539888583335364;
    let offset = 2.0 * mat2x2<f32>(v1, v2) * position;
    
    out.position = vec4<f32>(v_center + offset, 0.0, 1.0);
    out.screen_pos = position;
    out.color = vec4<f32>(unpack2x16float(vertex.color_0), unpack2x16float(vertex.color_1));
    
    let cov = mat2x2<f32>(v1, v2);
    let det = cov[0].x * cov[1].y - cov[0].y * cov[1].x;
    if (abs(det) < 0.0001) {
        out.conic = vec3<f32>(0.0, 0.0, 0.0);
    } else {
        let inv_det = 1.0 / det;
        let conic_mat = mat2x2<f32>(
            cov[1].y * inv_det, -cov[0].y * inv_det,
            -cov[1].x * inv_det, cov[0].x * inv_det
        );
        out.conic = vec3<f32>(conic_mat[0].x, conic_mat[0].y, conic_mat[1].y);
    }
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let conic = vec3<f32>(in.conic.x, in.conic.y, in.conic.z);

    let delta = in.screen_pos;
    let power = -0.5 * (conic.x * delta.x * delta.x + 
                        conic.z * delta.y * delta.y) - 
                        conic.y * delta.x * delta.y;
    
    let power_clamped = clamp(power, -10.0, 10.0);
    let alpha = min(0.99, in.color.a * exp(power_clamped));
    
    if (alpha < 1.0 / 255.0) {
        discard;
    }
    
    return vec4<f32>(in.color.rgb, alpha);
}