import { Float16Array } from '@petamoriken/float16';
import { log, time, timeLog } from './simple-console';
import { decodeHeader, readRawVertex, nShCoeffs } from './plyreader';

const c_size_3d_gaussian = 20;
const c_size_sh_coef = 96;
const c_size_splat_2d = 24;

export type PointCloud = Awaited<ReturnType<typeof load>>;

export async function load(file: string, device: GPUDevice) {
    const blob = new Blob([file]);
    const arrayBuffer = await new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = function (event) {
            resolve(event.target.result);
        };

        reader.onerror = reject;
        reader.readAsArrayBuffer(blob);
    });

    const [vertexCount, propertyTypes, vertexData] = decodeHeader(arrayBuffer as ArrayBuffer);

    var nRestCoeffs = 0;
    for (const propertyName in propertyTypes) {
        if (propertyName.startsWith('f_rest_')) {
            nRestCoeffs += 1;
        }
    }
    const nCoeffsPerColor = nRestCoeffs / 3;
    const sh_deg = Math.sqrt(nCoeffsPerColor + 1) - 1;
    const num_coefs = nShCoeffs(sh_deg);

    const num_points = vertexCount;

    log(`processing loaded attributes...`);
    time();

    const gaussian_3d_buffer = device.createBuffer({
        size: num_points * c_size_3d_gaussian,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
    });
    const gaussian = new Float16Array(gaussian_3d_buffer.getMappedRange());

    const sh_buffer = device.createBuffer({
        size: num_points * c_size_sh_coef,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
    });
    const sh = new Uint32Array(sh_buffer.getMappedRange());

    const splat_2d_buffer = device.createBuffer({
        size: num_points * c_size_splat_2d,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    let firstGaussianSHValues: number[] = [];

    var readOffset = 0;
    for (let i = 0; i < num_points; i++) {
        const [newReadOffset, rawVertex] = readRawVertex(readOffset, vertexData, propertyTypes);
        readOffset = newReadOffset;

        const gaussian_offset = i * (c_size_3d_gaussian / 2);
        const sh_offset = i * (c_size_sh_coef / 4);

        if (i === 0) {
            firstGaussianSHValues = [rawVertex.f_dc_0, rawVertex.f_dc_1, rawVertex.f_dc_2];
        }

        gaussian[gaussian_offset + 0] = rawVertex.x;
        gaussian[gaussian_offset + 1] = rawVertex.y;
        gaussian[gaussian_offset + 2] = rawVertex.z;

        gaussian[gaussian_offset + 3] = rawVertex.opacity;

        gaussian[gaussian_offset + 4] = rawVertex.scale_0;
        gaussian[gaussian_offset + 5] = rawVertex.scale_1;
        gaussian[gaussian_offset + 6] = rawVertex.scale_2;
        gaussian[gaussian_offset + 7] = 0;
        gaussian[gaussian_offset + 8] = 0;
        gaussian[gaussian_offset + 9] = 0;

        const dc_r = rawVertex.f_dc_0;
        const dc_g = rawVertex.f_dc_1;
        const dc_b = rawVertex.f_dc_2;

        sh[sh_offset + 0] = pack2x16floatToU32(dc_r, dc_g);
        sh[sh_offset + 1] = pack2x16floatToU32(dc_b, 0);
        sh[sh_offset + 2] = pack2x16floatToU32(0, 0);

        for (let j = 3; j < 24; j++) {
            sh[sh_offset + j] = 0;
        }
    }

    gaussian_3d_buffer.unmap();
    sh_buffer.unmap();

    function pack2x16floatToU32(a: number, b: number): number {
        const fa = new Float16Array([a]);
        const fb = new Float16Array([b]);
        const va = new Uint16Array(fa.buffer)[0];
        const vb = new Uint16Array(fb.buffer)[0];
        return (vb << 16) | va;
    }

    timeLog();

    return {
        num_points: num_points,
        sh_deg: sh_deg,
        gaussian_3d_buffer,
        sh_buffer,
        splat_2d_buffer,
    };
}