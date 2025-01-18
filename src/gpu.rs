use ndarray::Array2;
use wgpu::util::DeviceExt;

const INVERT_SHADER_WGSL: &str = include_str!("shaders/invert.wgsl");
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

const MULTIPLY_SHADER_WGSL: &str = include_str!("shaders/multiply.wgsl");
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

const ADD_SHADER_WGSL: &str = include_str!("shaders/add.wgsl");
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

const NORMALISE_SHADER_WGSL: &str = include_str!("shaders/normalise.wgsl");
const NORMALISE_SHADER_LAYOUT: &[wgpu::BindGroupLayoutEntry] = &[
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

const SIMULATION_SHADER_WGSL: &str = include_str!("shaders/simulation.wgsl");
const SIMULATION_SHADER_LAYOUT: &[wgpu::BindGroupLayoutEntry] = &[
    // heightmap read-only
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
    // distance read-write
    wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
];

const GRADIENT_SHADER_WGSL: &str = include_str!("shaders/gradient.wgsl");
const GRADIENT_SHADER_LAYOUT: &[wgpu::BindGroupLayoutEntry] = &[
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
    wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
];

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

    // Simulation pipeline
    simulation_pipeline: wgpu::ComputePipeline,
    simulation_bind_group_layout: wgpu::BindGroupLayout,

    // Normalise pipeline
    normalise_pipeline: wgpu::ComputePipeline,
    normalise_bind_group_layout: wgpu::BindGroupLayout,

    // Gradient pipeline
    gradient_pipeline: wgpu::ComputePipeline,
    gradient_bind_group_layout: wgpu::BindGroupLayout,

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
            &INVERT_SHADER_WGSL,
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

        // 4) Normalise pipeline
        let (normalise_pipeline, normalise_bind_group_layout) = Self::create_pipeline(
            &device,
            NORMALISE_SHADER_WGSL,
            NORMALISE_SHADER_LAYOUT,
            "Normalise Pipeline",
            rows,
            cols,
        );

        // 5) Simulation pipeline
        let (simulation_pipeline, simulation_bind_group_layout) = Self::create_pipeline(
            &device,
            SIMULATION_SHADER_WGSL,
            SIMULATION_SHADER_LAYOUT,
            "Simulation Pipeline",
            rows,
            cols,
        );

        // 6) Gradient pipeline
        let (gradient_pipeline, gradient_bind_group_layout) = Self::create_pipeline(
            &device,
            GRADIENT_SHADER_WGSL,
            GRADIENT_SHADER_LAYOUT,
            "Gradient Pipeline",
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
            normalise_pipeline,
            normalise_bind_group_layout,
            add_pipeline,
            add_bind_group_layout,
            simulation_pipeline,
            simulation_bind_group_layout,
            gradient_pipeline,
            gradient_bind_group_layout,
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

    // Scale input to [0, 1]
    // (I - min(I)) / (max(I) - min(I))
    pub async fn normalise(&self, input: &Array2<f32>) -> Array2<f32> {
        // 1) Calculate min and max on CPU
        let min_val = input.fold(f32::MAX, |a, &b| a.min(b));
        let max_val = input.fold(f32::MIN, |a, &b| a.max(b));

        // Avoid division by zero if max == min
        if (max_val - min_val).abs() < f32::EPSILON {
            return input.clone(); // or fill with 0.0, etc.
        }

        let range = max_val - min_val;

        // 2) Create uniform buffer [min_val, range]
        let param = [min_val, range];
        let param_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Normalise Param Buffer"),
                contents: bytemuck::cast_slice(&param),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // 3) Dispatch the normalise pipeline
        self.run_compute(
            input,
            Some(&param_buffer),
            &self.normalise_pipeline,
            &self.normalise_bind_group_layout,
        )
        .await
    }

    pub async fn simulate(&self, heightmap: &Array2<f32>) -> Array2<f32> {
        let mut distances = Array2::<f32>::zeros(heightmap.raw_dim());
        self.run_simulation(heightmap, &mut distances).await;

        distances
    }

    pub async fn gradient(&self, heightmap: &Array2<f32>) -> ndarray::Array3<f32> {
        let (rows, cols) = (self.rows, self.cols);

        // We'll build an output buffer with shape [rows, cols, 2].
        // That’s 2 floats per pixel => total len = rows * cols * 2.
        let out_len = (rows * cols * 2) as usize;
        let gradient_data = vec![0.0f32; out_len]; // CPU side

        // 1) Create staging buffers
        let height_bytes = bytemuck::cast_slice(heightmap.as_slice().unwrap());
        let staging_height = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Staging Heightmap"),
                contents: height_bytes,
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // We'll also create a staging buffer for the gradient output (initially all zero).
        let out_bytes = bytemuck::cast_slice(&gradient_data);
        let staging_grad = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Staging Gradient"),
                contents: out_bytes,
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // 2) Create GPU storage buffers
        let float_size = std::mem::size_of::<f32>() as wgpu::BufferAddress;
        let height_buf_size = (rows * cols) as wgpu::BufferAddress * float_size;
        let grad_buf_size = (rows * cols * 2) as wgpu::BufferAddress * float_size;

        let height_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Heightmap (storage)"),
            size: height_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grad_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gradient (storage)"),
            size: grad_buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 3) Copy data to the GPU
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gradient Init Encoder"),
            });
        encoder.copy_buffer_to_buffer(&staging_height, 0, &height_buffer, 0, height_buf_size);
        encoder.copy_buffer_to_buffer(&staging_grad, 0, &grad_buffer, 0, grad_buf_size);
        self.queue.submit(Some(encoder.finish()));

        // 4) Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.gradient_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        height_buffer.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(grad_buffer.as_entire_buffer_binding()),
                },
            ],
            label: Some("Gradient Bind Group"),
        });

        // 5) Dispatch compute
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gradient Compute Encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gradient Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.gradient_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let (wx, wy, wz) = self.workgroup_size;
            let nx = (cols as f32 / wx as f32).ceil() as u32;
            let ny = (rows as f32 / wy as f32).ceil() as u32;
            cpass.dispatch_workgroups(nx, ny, wz);
        }
        self.queue.submit(Some(encoder.finish()));

        // 6) Copy results back
        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gradient Readback Buffer"),
            size: grad_buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gradient Copy Encoder"),
            });
        encoder.copy_buffer_to_buffer(&grad_buffer, 0, &readback_buf, 0, grad_buf_size);
        self.queue.submit(Some(encoder.finish()));

        {
            let buffer_slice = readback_buf.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
            self.device.poll(wgpu::Maintain::Wait);
            receiver.receive().await.unwrap().unwrap();
        }

        // 7) Rebuild into Array3<f32> of shape (rows, cols, 2)
        let data = readback_buf.slice(..).get_mapped_range();
        let gpu_result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buf.unmap();

        // Convert the flat vector into shape [rows, cols, 2]
        ndarray::Array3::from_shape_vec((rows as usize, cols as usize, 2), gpu_result).unwrap()
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

    async fn run_simulation(&self, heightmap: &Array2<f32>, distances: &mut Array2<f32>) {
        let (rows, cols) = (self.rows, self.cols);
        let height_bytes = bytemuck::cast_slice(heightmap.as_slice().unwrap());
        let dist_bytes = bytemuck::cast_slice(distances.as_slice().unwrap());

        let size = (rows * cols) as wgpu::BufferAddress
            * std::mem::size_of::<f32>() as wgpu::BufferAddress;

        // 1) Create staging buffers
        let staging_height = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Heightmap (staging)"),
                contents: height_bytes,
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        let staging_dist = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Distance (staging)"),
                contents: dist_bytes, // all zero
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // 2) Create GPU storage buffers
        let height_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Heightmap (storage)"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dist_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Distance (storage)"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 3) Copy to GPU
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sim Init Encoder"),
            });
        encoder.copy_buffer_to_buffer(&staging_height, 0, &height_buffer, 0, size);
        encoder.copy_buffer_to_buffer(&staging_dist, 0, &dist_buffer, 0, size);
        self.queue.submit(Some(encoder.finish()));

        // 4) Create the bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.simulation_bind_group_layout, // from pipeline creation
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        height_buffer.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(dist_buffer.as_entire_buffer_binding()),
                },
            ],
            label: Some("Sim Bind Group"),
        });

        // 5) Dispatch the compute
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sim Compute Encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Simulation Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.simulation_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let (wx, wy, wz) = self.workgroup_size;
            let nx = (cols as f32 / wx as f32).ceil() as u32;
            let ny = (rows as f32 / wy as f32).ceil() as u32;
            cpass.dispatch_workgroups(nx, ny, wz);
        }
        self.queue.submit(Some(encoder.finish()));

        // 6) Copy results back to CPU
        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sim Readback Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sim Copy Encoder"),
            });
        encoder.copy_buffer_to_buffer(&dist_buffer, 0, &readback_buf, 0, size);
        self.queue.submit(Some(encoder.finish()));

        {
            let buffer_slice = readback_buf.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
            self.device.poll(wgpu::Maintain::Wait);
            receiver.receive().await.unwrap().unwrap();
        }

        // 7) Fill 'distances' array
        let data = readback_buf.slice(..).get_mapped_range();
        let gpu_result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buf.unmap();

        // Overwrite the distances array with GPU results
        distances
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(&gpu_result);
    }
}
