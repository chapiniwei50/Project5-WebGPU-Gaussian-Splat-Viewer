import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
    update_settings?: (scaling: number) => void;
}

export default function get_renderer(
    pc: PointCloud,
    device: GPUDevice,
    presentation_format: GPUTextureFormat,
    camera_buffer: GPUBuffer,
): GaussianRenderer {

    const sorter = get_sorter(pc.num_points, device);

    const preprocess_pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: device.createShaderModule({ code: preprocessWGSL }),
            entryPoint: 'preprocess',
        },
    });

    const preprocessGroup1 = device.createBindGroup({
        layout: preprocess_pipeline.getBindGroupLayout(1),
        entries: [
            { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
            { binding: 1, resource: { buffer: pc.sh_buffer } },
            { binding: 2, resource: { buffer: pc.splat_2d_buffer } },
        ],
    });

    const cameraGroup = device.createBindGroup({
        layout: preprocess_pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: camera_buffer } },
        ],
    });

    const sortGroup = device.createBindGroup({
        layout: preprocess_pipeline.getBindGroupLayout(2),
        entries: [
            { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
            { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
            { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
        ],
    });

    const settingsBuffer = device.createBuffer({
        size: 80,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const settingsData = new Float32Array(20);
    settingsData[0] = -10.0;
    settingsData[1] = -10.0;
    settingsData[2] = -10.0;
    settingsData[3] = 0.0;
    settingsData[4] = 10.0;
    settingsData[5] = 10.0;
    settingsData[6] = 10.0;
    settingsData[7] = 0.0;
    settingsData[8] = 1.0;
    settingsData[9] = 3;
    settingsData[10] = 0;
    settingsData[11] = 0;
    settingsData[12] = 0.3;
    settingsData[13] = 0.0;
    settingsData[14] = 0.0;
    settingsData[15] = 0;
    settingsData[16] = 0.0;
    settingsData[17] = 0.0;
    settingsData[18] = 0.0;
    settingsData[19] = 0;

    device.queue.writeBuffer(settingsBuffer, 0, settingsData);

    const settingsGroup = device.createBindGroup({
        layout: preprocess_pipeline.getBindGroupLayout(3),
        entries: [
            { binding: 0, resource: { buffer: settingsBuffer } },
        ],
    });

    const renderPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: device.createShaderModule({ code: renderWGSL }),
            entryPoint: 'vs_main'
        },
        fragment: {
            module: device.createShaderModule({ code: renderWGSL }),
            entryPoint: 'fs_main',
            targets: [{
                format: presentation_format,
                blend: {
                    color: {
                        srcFactor: 'one',
                        dstFactor: 'one-minus-src-alpha',
                        operation: 'add'
                    },
                    alpha: {
                        srcFactor: 'one',
                        dstFactor: 'one-minus-src-alpha',
                        operation: 'add'
                    },
                },
            }],
        },
        primitive: {
            topology: 'triangle-strip',
        },
    });

    const renderSplatsGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 2, resource: { buffer: pc.splat_2d_buffer } },
        ],
    });

    const renderIndicesGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(1),
        entries: [
            { binding: 4, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
        ],
    });

    const drawIndirectBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    device.queue.writeBuffer(drawIndirectBuffer, 0, new Uint32Array([4, 0, 0, 0]));

    let frameCount = 0;
    let lastFpsTime = performance.now();
    let lastFrameTime = performance.now();
    let frameTimes: number[] = [];
    let totalFrames = 0;

    const readbackBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    async function readVisibleCount() {
        try {
            await readbackBuffer.mapAsync(GPUMapMode.READ);
            const array = new Uint32Array(readbackBuffer.getMappedRange());
            const visibleCount = array[0];
            readbackBuffer.unmap();
            return visibleCount;
        } catch (e) {
            return 0;
        }
    }
    return {
        frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
            frameCount++;
            totalFrames++;

            const currentTime = performance.now();
            const timeSinceLastFps = currentTime - lastFpsTime;

            if (timeSinceLastFps > 1000) {
                const fps = Math.round((frameCount * 1000) / timeSinceLastFps);
                frameCount = 0;
                lastFpsTime = currentTime;
            }

            const frameTime = currentTime - lastFrameTime;
            frameTimes.push(frameTime);
            if (frameTimes.length > 60) frameTimes.shift();
            lastFrameTime = currentTime;

            device.queue.writeBuffer(sorter.sort_info_buffer, 0, new Uint32Array([0]));

            const computePass = encoder.beginComputePass();
            computePass.setPipeline(preprocess_pipeline);
            computePass.setBindGroup(0, cameraGroup);
            computePass.setBindGroup(1, preprocessGroup1);
            computePass.setBindGroup(2, sortGroup);
            computePass.setBindGroup(3, settingsGroup);
            computePass.dispatchWorkgroups(Math.ceil(pc.num_points / 256));
            computePass.end();

            sorter.sort(encoder);



            encoder.copyBufferToBuffer(
                sorter.sort_info_buffer,
                0,
                drawIndirectBuffer,
                4,
                4
            );

            const renderPass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: texture_view,
                    clearValue: [0, 0, 0, 0],
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            renderPass.setPipeline(renderPipeline);
            renderPass.setBindGroup(0, renderSplatsGroup);
            renderPass.setBindGroup(1, renderIndicesGroup);
            renderPass.drawIndirect(drawIndirectBuffer, 0);
            renderPass.end();

        },
     
        update_settings: (scaling: number) => {
            const updateData = new Float32Array(1);
            updateData[0] = scaling;
            device.queue.writeBuffer(settingsBuffer, 32, updateData);
        },

        camera_buffer,
    };
}