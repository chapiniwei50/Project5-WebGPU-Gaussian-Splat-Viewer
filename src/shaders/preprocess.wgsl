struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Gaussian {
    pos_opacity: array<u32, 2>,
    cov: array<u32, 3>
}

struct Splat {
    v_0: u32,
    v_1: u32,
    pos: u32,
    color_0: u32,
    color_1: u32
};

struct SortInfos {
    keys_size: atomic<u32>,
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
};

struct RenderSettings {
    clipping_box_min: vec4<f32>,
    clipping_box_max: vec4<f32>,
    gaussian_scaling: f32,
    max_sh_deg: u32,
    show_env_map: u32,
    mip_spatting: u32,
    kernel_size: f32,
    walltime: f32,
    scene_extend: f32,
    pad0: u32,
    center: vec3<f32>,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1) var<storage, read> sh_coefs: array<array<u32, 24>>;
@group(1) @binding(2) var<storage, read_write> points_2d: array<Splat>;
@group(2) @binding(0) var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1) var<storage, read_write> sort_depths: array<u32>;
@group(2) @binding(2) var<storage, read_write> sort_indices: array<u32>;
@group(3) @binding(0) var<uniform> render_settings: RenderSettings;

fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let r = unpack2x16float(sh_coefs[splat_idx][(c_idx * 3u + 0u) / 2u])[(c_idx * 3u + 0u) % 2u];
    let g = unpack2x16float(sh_coefs[splat_idx][(c_idx * 3u + 1u) / 2u])[(c_idx * 3u + 1u) % 2u];
    let b = unpack2x16float(sh_coefs[splat_idx][(c_idx * 3u + 2u) / 2u])[(c_idx * 3u + 2u) % 2u];
    return vec3<f32>(r, g, b);
}

fn evaluate_sh(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = 0.28209479177387814 * sh_coef(v_idx, 0u);

    if (sh_deg > 0u) {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += -0.4886025119029199 * y * sh_coef(v_idx, 1u) + 0.4886025119029199 * z * sh_coef(v_idx, 2u) - 0.4886025119029199 * x * sh_coef(v_idx, 3u);

        if (sh_deg > 1u) {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let yz = y * z;
            let xz = x * z;

            result += 1.0925484305920792 * xy * sh_coef(v_idx, 4u) + 
                     -1.0925484305920792 * yz * sh_coef(v_idx, 5u) + 
                     0.31539156525252005 * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + 
                     -1.0925484305920792 * xz * sh_coef(v_idx, 7u) + 
                     0.5462742152960396 * (xx - yy) * sh_coef(v_idx, 8u);
        }
    }
    result += 0.5;
    return result;
}

fn cov_coefs(v_idx: u32) -> array<f32, 6> {
    let a = unpack2x16float(gaussians[v_idx].cov[0]);
    let b = unpack2x16float(gaussians[v_idx].cov[1]);
    let c = unpack2x16float(gaussians[v_idx].cov[2]);
    return array<f32, 6>(a.x, a.y, b.x, b.y, c.x, c.y);
}

@compute @workgroup_size(256, 1, 1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    let focal = camera.focal;
    let viewport = camera.viewport;
    let vertex = gaussians[idx];
    let a = unpack2x16float(vertex.pos_opacity[0]);
    let b = unpack2x16float(vertex.pos_opacity[1]);
    let xyz = vec3<f32>(a.x, a.y, b.x);
    let opacity = b.y;

    var camspace = camera.view * vec4<f32>(xyz, 1.0);
    let pos2d = camera.proj * camspace;
    let bounds = 1.2 * pos2d.w;

    if (pos2d.z <= 0.0 || pos2d.z >= pos2d.w || 
        pos2d.x < -bounds || pos2d.x > bounds || 
        pos2d.y < -bounds || pos2d.y > bounds) {
        return;
    }

    let cov_sparse = cov_coefs(idx);
    let scaling = render_settings.gaussian_scaling;

    let diagonal1 = cov_sparse[0] + 0.3;
    let offDiagonal = cov_sparse[1];
    let diagonal2 = cov_sparse[3] + 0.3;

    let mid = 0.5 * (diagonal1 + diagonal2);
    let radius = length(vec2<f32>((diagonal1 - diagonal2) / 2.0, offDiagonal));
    let lambda1 = mid + radius;
    let lambda2 = max(mid - radius, 0.1);

    let diagonalVector = normalize(vec2<f32>(offDiagonal, lambda1 - diagonal1));
    let v1 = sqrt(2.0 * lambda1) * diagonalVector;
    let v2 = sqrt(2.0 * lambda2) * vec2<f32>(diagonalVector.y, -diagonalVector.x);

    let v_center = pos2d.xy / pos2d.w;

    let dc_color = sh_coef(idx, 0u);
    let color = vec4<f32>(max(vec3<f32>(0.0), dc_color + 0.5), opacity);

    let store_idx = atomicAdd(&sort_infos.keys_size, 1u);
    let v = vec4<f32>(v1 / viewport, v2 / viewport);
    
    points_2d[store_idx] = Splat(
        pack2x16float(v.xy),
        pack2x16float(v.zw),
        pack2x16float(v_center),
        pack2x16float(color.rg),
        pack2x16float(color.ba)
    );

    sort_depths[store_idx] = bitcast<u32>(-camspace.z);
    sort_indices[store_idx] = store_idx;
}