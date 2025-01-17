use ndarray_images::Image;
use rand::prelude::*;
use terrain::{noise, utils};

use ndarray::{Array2, ArrayView2};
use wgpu::util::DeviceExt;

#[tokio::main]
async fn main() {
    // Map settings
    let resolution = (512, 512);

    // Initialize the random number generator
    let mut rng = rand::thread_rng();

    // Generate a height map
    let height = noise::sample_perlin(
        resolution,
        &[
            ((2, 2), 1.0),
            ((3, 3), 0.5),
            ((5, 5), 0.25),
            ((7, 7), 0.125),
            ((11, 11), 0.0625),
            ((13, 13), 0.03125),
            ((17, 17), 0.015625),
        ],
        &mut rng,
    );

    let mut height = invert_array_with_wgpu(&height).await;
    height += 1.0;
    height = utils::normalize(height);

    // Save the height map
    height
        .save("output/height.png")
        .expect("Failed to save height map");
}

async fn invert_array_with_wgpu(input: &Array2<f32>) -> Array2<f32> {
    // 1) Set up WGPU instance, adapter, device, and queue
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let (device, queue) = adapter
        .request_device(&Default::default(), None)
        .await
        .unwrap();

    // Flatten the 2D array and convert to bytes
    let (rows, cols) = input.dim();
    let flat_input = input.as_slice().unwrap();
    let input_bytes = bytemuck::cast_slice(flat_input);

    // 2) Create buffers
    let size =
        (rows * cols) as wgpu::BufferAddress * std::mem::size_of::<f32>() as wgpu::BufferAddress;
    let staging_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Staging Buffer"),
        contents: input_bytes,
        usage: wgpu::BufferUsages::COPY_SRC,
    });

    let storage_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Storage Buffer"),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Copy input data into the storage buffer
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&staging_buf, 0, &storage_buf, 0, size);
    queue.submit(Some(encoder.finish()));

    // 3) Prepare compute shader (WGSL)
    // Inverts each value: data[index] = -data[index];
    let shader_src = r#"
            @group(0) @binding(0)
            var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
                let rows: u32 = 512u;
                let cols: u32 = 256u;
                let index = gid.y * cols + gid.x;
                if (gid.x < cols && gid.y < rows) {
                    data[index] = -data[index];
                }
            }
        "#;

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Invert Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    // 4) Create compute pipeline
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
        label: None,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            ..Default::default()
        },
        cache: None,
    });

    // 5) Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(storage_buf.as_entire_buffer_binding()),
        }],
        label: None,
    });

    // 6) Dispatch compute
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        // We dispatch enough groups to cover rows x cols
        // e.g., 8x8 sized groups => divide up total element count
        let workgroup_size = (8, 8, 1);
        let nx = ((cols as f32) / workgroup_size.0 as f32).ceil() as u32;
        let ny = ((rows as f32) / workgroup_size.1 as f32).ceil() as u32;
        cpass.dispatch_workgroups(nx, ny, 1);
    }
    queue.submit(Some(encoder.finish()));

    // 7) Copy results back to CPU
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&storage_buf, 0, &readback_buf, 0, size);
    queue.submit(Some(encoder.finish()));

    // Wait for GPU to finish, then map the buffer
    {
        let buffer_slice = readback_buf.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        receiver.receive().await.unwrap().unwrap();
    }

    let data = readback_buf.slice(..).get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback_buf.unmap();

    // Reshape flattened result into Array2
    Array2::from_shape_vec((rows, cols), result).unwrap()
}
