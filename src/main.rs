use ndarray::Array2;
use ndarray_images::Image;
use terrain::{noise, utils};
use wgpu::util::DeviceExt;

const INVERT_SHADER_WGSL: &str = r#"
    @group(0) @binding(0)
    var<storage, read_write> data: array<f32>;

    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
        let index = gid.y * u32(@cols) + gid.x;
        if (gid.x < u32(@cols) && gid.y < u32(@rows)) {
            data[index] = -data[index];
        }
    }
"#;
const INVERT_SHADER_LAYOUT: &[wgpu::BindGroupLayoutEntry] = &[wgpu::BindGroupLayoutEntry {
    binding: 0,
    visibility: wgpu::ShaderStages::COMPUTE,
    ty: wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
    },
    count: None,
}];

const MULTIPLY_SHADER_WGSL: &str = r#"
    @group(0) @binding(0)
    var<storage, read_write> data: array<f32>;

    @group(0) @binding(1)
    var<uniform> scale: f32;

    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
        let index = gid.y * u32(@cols) + gid.x;
        if (gid.x < u32(@cols) && gid.y < u32(@rows)) {
            data[index] = data[index] * scale;
        }
    }
"#;
const MULTIPLY_SHADER_LAYOUT: &[wgpu::BindGroupLayoutEntry] = &[
    // data buffer
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
    // scale uniform
    wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
];

const ADD_SHADER_WGSL: &str = r#"
    @group(0) @binding(0)
    var<storage, read_write> data: array<f32>;

    @group(0) @binding(1)
    var<uniform> offset: f32;

    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
        let index = gid.y * u32(@cols) + gid.x;
        if (gid.x < u32(@cols) && gid.y < u32(@rows)) {
            data[index] = data[index] + offset;
        }
    }
"#;
const ADD_SHADER_LAYOUT: &[wgpu::BindGroupLayoutEntry] = &[
    // data buffer
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
    // offset uniform
    wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
];

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

    let inverter = Gpu::new(resolution.0 as u32, resolution.1 as u32).await;
    let mut height = inverter.invert(&height).await;

    height += 1.0;
    height = utils::normalize(height);

    // Save the height map
    height
        .save("output/height.png")
        .expect("Failed to save height map");
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Invert pipeline
    invert_pipeline: wgpu::ComputePipeline,
    invert_bind_group_layout: wgpu::BindGroupLayout,

    // Multiply pipeline
    multiply_pipeline: wgpu::ComputePipeline,
    multiply_bind_group_layout: wgpu::BindGroupLayout,

    // Add pipeline
    add_pipeline: wgpu::ComputePipeline,
    add_bind_group_layout: wgpu::BindGroupLayout,

    workgroup_size: (u32, u32, u32),
    rows: u32,
    cols: u32,
}

impl Gpu {
    pub async fn new(rows: u32, cols: u32) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter
            .request_device(&Default::default(), None)
            .await
            .unwrap();

        // 1) Invert pipeline
        let (invert_pipeline, invert_bind_group_layout) = Self::create_pipeline(
            &device,
            INVERT_SHADER_WGSL,
            INVERT_SHADER_LAYOUT,
            "Invert Pipeline",
            rows,
            cols,
        );

        // 2) Multiply pipeline
        let (multiply_pipeline, multiply_bind_group_layout) = Self::create_pipeline(
            &device,
            MULTIPLY_SHADER_WGSL,
            MULTIPLY_SHADER_LAYOUT,
            "Multiply Pipeline",
            rows,
            cols,
        );

        // 3) Add pipeline
        let (add_pipeline, add_bind_group_layout) = Self::create_pipeline(
            &device,
            ADD_SHADER_WGSL,
            ADD_SHADER_LAYOUT,
            "Add Pipeline",
            rows,
            cols,
        );

        Self {
            device,
            queue,
            invert_pipeline,
            invert_bind_group_layout,
            multiply_pipeline,
            multiply_bind_group_layout,
            add_pipeline,
            add_bind_group_layout,
            workgroup_size: (8, 8, 1),
            rows,
            cols,
        }
    }

    fn create_pipeline(
        device: &wgpu::Device,
        wgsl_code: &str,
        layout_entries: &[wgpu::BindGroupLayoutEntry],
        label: &str,
        rows: u32,
        cols: u32,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader_src = wgsl_code
            .replace("@rows", &rows.to_string())
            .replace("@cols", &cols.to_string());

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: layout_entries,
            label: None,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                ..Default::default()
            },
            cache: None,
        });

        (pipeline, bind_group_layout)
    }

    // Multiply input by -1
    // -I
    pub async fn invert(&self, input: &Array2<f32>) -> Array2<f32> {
        // Reuses your existing pipeline, layout, and method approach
        self.run_compute(
            input,
            None, // no uniform buffer needed
            &self.invert_pipeline,
            &self.invert_bind_group_layout,
        )
        .await
    }

    // Multiply input by a scalar
    // I * x
    pub async fn multiply(&self, input: &Array2<f32>, scale: f32) -> Array2<f32> {
        // Create a uniform buffer containing 'scale'
        let scale_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Multiply - scaler Buffer"),
                contents: bytemuck::cast_slice(&[scale]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        self.run_compute(
            input,
            Some(&scale_buffer),
            &self.multiply_pipeline,
            &self.multiply_bind_group_layout,
        )
        .await
    }

    // Add a scalar to input
    // I + x
    pub async fn add(&self, input: &Array2<f32>, offset: f32) -> Array2<f32> {
        let offset_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Add - offset Buffer"),
                contents: bytemuck::cast_slice(&[offset]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        self.run_compute(
            input,
            Some(&offset_buf),
            &self.add_pipeline,
            &self.add_bind_group_layout,
        )
        .await
    }

    // Generalised helper to run a compute pass
    async fn run_compute(
        &self,
        input: &Array2<f32>,
        scale_buf: Option<&wgpu::Buffer>,
        pipeline: &wgpu::ComputePipeline,
        layout: &wgpu::BindGroupLayout,
    ) -> Array2<f32> {
        let (rows, cols) = (self.rows, self.cols);

        let flat_input = input.as_slice().unwrap();
        let input_bytes = bytemuck::cast_slice(flat_input);

        let size = (rows * cols) as wgpu::BufferAddress
            * std::mem::size_of::<f32>() as wgpu::BufferAddress;

        // Create staging + storage buffers
        let staging_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Staging Buffer"),
                contents: input_bytes,
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        let storage_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Storage Buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Copy the data over
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Init Command Encoder"),
            });
        encoder.copy_buffer_to_buffer(&staging_buf, 0, &storage_buf, 0, size);
        self.queue.submit(Some(encoder.finish()));

        // Create the bind group
        let entries = match scale_buf {
            Some(s) => vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(storage_buf.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(s.as_entire_buffer_binding()),
                },
            ],
            None => vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(storage_buf.as_entire_buffer_binding()),
            }],
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &entries,
            label: None,
        });

        // Dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let (wx, wy, wz) = self.workgroup_size;
            let nx = (cols as f32 / wx as f32).ceil() as u32;
            let ny = (rows as f32 / wy as f32).ceil() as u32;
            cpass.dispatch_workgroups(nx, ny, wz);
        }
        self.queue.submit(Some(encoder.finish()));

        // Read back results
        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Copy Encoder"),
            });
        encoder.copy_buffer_to_buffer(&storage_buf, 0, &readback_buf, 0, size);
        self.queue.submit(Some(encoder.finish()));

        {
            let buffer_slice = readback_buf.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
            self.device.poll(wgpu::Maintain::Wait);
            receiver.receive().await.unwrap().unwrap();
        }

        let data = readback_buf.slice(..).get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buf.unmap();

        ndarray::Array2::from_shape_vec((rows as usize, cols as usize), result).unwrap()
    }
}
