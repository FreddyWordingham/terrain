use ndarray::{Array2, Array3};
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

const MAGNITUDE_SHADER_WGSL: &str = include_str!("shaders/magnitude.wgsl");
const MAGNITUDE_SHADER_LAYOUT: &[wgpu::BindGroupLayoutEntry] = &[
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

    // Single-buffer ops
    invert_pipeline: wgpu::ComputePipeline,
    invert_bind_group_layout: wgpu::BindGroupLayout,
    multiply_pipeline: wgpu::ComputePipeline,
    multiply_bind_group_layout: wgpu::BindGroupLayout,
    add_pipeline: wgpu::ComputePipeline,
    add_bind_group_layout: wgpu::BindGroupLayout,

    // Two-buffer ops
    simulation_pipeline: wgpu::ComputePipeline,
    simulation_bind_group_layout: wgpu::BindGroupLayout,
    normalise_pipeline: wgpu::ComputePipeline,
    normalise_bind_group_layout: wgpu::BindGroupLayout,
    gradient_pipeline: wgpu::ComputePipeline,
    gradient_bind_group_layout: wgpu::BindGroupLayout,
    magnitude_pipeline: wgpu::ComputePipeline,
    magnitude_bind_group_layout: wgpu::BindGroupLayout,

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

        // 7) Gradient magnitude pipeline
        let (magnitude_pipeline, magnitude_bind_group_layout) = Self::create_pipeline(
            &device,
            MAGNITUDE_SHADER_WGSL,
            MAGNITUDE_SHADER_LAYOUT,
            "Magnitude Pipeline",
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
            magnitude_pipeline,
            magnitude_bind_group_layout,
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

    /// For ops reading/writing a single Array2<f32> (invert, add, multiply, normalise, etc.).
    async fn run_single_buffer_op(
        &self,
        input: &Array2<f32>,
        optional_uniform: Option<&wgpu::Buffer>,
        pipeline: &wgpu::ComputePipeline,
        layout: &wgpu::BindGroupLayout,
    ) -> Array2<f32> {
        let (rows, cols) = (self.rows, self.cols);
        let size_bytes = (rows * cols * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress;

        // 1) Create staging + GPU storage
        let staging_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Staging Single"),
                contents: bytemuck::cast_slice(input.as_slice().unwrap()),
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        let gpu_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Single-Buffer"),
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Copy input data
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Single Init Encoder"),
                });
            encoder.copy_buffer_to_buffer(&staging_buf, 0, &gpu_buf, 0, size_bytes);
            self.queue.submit(Some(encoder.finish()));
        }

        // 2) Create bind group
        let bind_group_entries = match optional_uniform {
            Some(uniform_buf) => vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(gpu_buf.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(uniform_buf.as_entire_buffer_binding()),
                },
            ],
            None => vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(gpu_buf.as_entire_buffer_binding()),
            }],
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &bind_group_entries,
            label: Some("Single BindGroup"),
        });

        // 3) Dispatch
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Single Compute Encoder"),
                });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Single Pass"),
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
        }

        // 4) Read back
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Single Readback"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Single Copy Encoder"),
                });
            encoder.copy_buffer_to_buffer(&gpu_buf, 0, &readback, 0, size_bytes);
            self.queue.submit(Some(encoder.finish()));
        }

        // Wait and map
        {
            let slice = readback.slice(..);
            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
            self.device.poll(wgpu::Maintain::Wait);
            rx.receive().await.unwrap().unwrap();
        }

        let data = readback.slice(..).get_mapped_range();
        let result = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        readback.unmap();

        Array2::from_shape_vec((rows as usize, cols as usize), result).unwrap()
    }

    /// For ops reading one buffer and writing to another (gradient, magnitude, simulation).
    async fn run_two_buffer_op(
        &self,
        input_bytes: &[f32],  // CPU side array for 'input' buffer
        output_bytes: &[f32], // CPU side array for 'output' buffer (often empty/zero init)
        pipeline: &wgpu::ComputePipeline,
        layout: &wgpu::BindGroupLayout,
        bytes_per_input_item: usize,
        bytes_per_output_item: usize,
    ) -> Vec<f32> {
        // Typically, `bytes_per_input_item` is 4 (float), or 8 if your data uses 2 floats per pixel.
        // Same pattern for output.

        let size_in = (input_bytes.len() * bytes_per_input_item) as wgpu::BufferAddress;
        let size_out = (output_bytes.len() * bytes_per_output_item) as wgpu::BufferAddress;

        // 1) Staging + GPU Buffers
        let staging_in = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Staging In"),
                contents: bytemuck::cast_slice(input_bytes),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        let staging_out = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Staging Out"),
                contents: bytemuck::cast_slice(output_bytes),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        let gpu_in = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU In"),
            size: size_in,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gpu_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Out"),
            size: size_out,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 2) Copy CPU → GPU
        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Init Two-Buffers"),
                });
            enc.copy_buffer_to_buffer(&staging_in, 0, &gpu_in, 0, size_in);
            enc.copy_buffer_to_buffer(&staging_out, 0, &gpu_out, 0, size_out);
            self.queue.submit(Some(enc.finish()));
        }

        // 3) Bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(gpu_in.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(gpu_out.as_entire_buffer_binding()),
                },
            ],
            label: Some("Two-Buffers BindGroup"),
        });

        // 4) Dispatch
        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compute Two-Buffers"),
                });
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Two-Buffers Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);

                // In your real code, compute (nx, ny, 1) from rows, cols, etc.
                cpass.dispatch_workgroups(16, 16, 1);
            }
            self.queue.submit(Some(enc.finish()));
        }

        // 5) Read back GPU → CPU
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Two-Buffers Readback"),
            size: size_out,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy Two-Buffers"),
                });
            enc.copy_buffer_to_buffer(&gpu_out, 0, &readback, 0, size_out);
            self.queue.submit(Some(enc.finish()));
        }

        // Map + return as Vec<f32>
        {
            let slice = readback.slice(..);
            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
            self.device.poll(wgpu::Maintain::Wait);
            rx.receive().await.unwrap().unwrap();
        }
        let data = readback.slice(..).get_mapped_range();
        let result_floats: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback.unmap();
        result_floats
    }

    // Multiply input by -1
    // -I
    pub async fn invert(&self, input: &Array2<f32>) -> Array2<f32> {
        self.run_single_buffer_op(
            input,
            None,
            &self.invert_pipeline,
            &self.invert_bind_group_layout,
        )
        .await
    }

    // Multiply input by a scalar
    // I * x
    pub async fn multiply(&self, input: &Array2<f32>, scale: f32) -> Array2<f32> {
        // 1) Make a uniform buffer for 'scale'
        let scale_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Multiply Uniform"),
                contents: bytemuck::cast_slice(&[scale]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        // 2) Pass it as `Some(&scale_buf)`
        self.run_single_buffer_op(
            input,
            Some(&scale_buf),
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
        self.run_single_buffer_op(
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
        // 1) Find min and max on CPU
        let min_val = input.fold(f32::MAX, |a, &b| a.min(b));
        let max_val = input.fold(f32::MIN, |a, &b| a.max(b));

        // Avoid division by zero if uniform data would have 0 range
        if (max_val - min_val).abs() < f32::EPSILON {
            // No changes if all data is the same
            return input.clone();
        }

        let range = max_val - min_val;

        // 2) Create a small uniform buffer containing (min_val, range)
        let param = [min_val, range];
        let param_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Normalise Param Buffer"),
                contents: bytemuck::cast_slice(&param),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // 3) Dispatch using run_single_buffer_op
        self.run_single_buffer_op(
            input,
            Some(&param_buffer),
            &self.normalise_pipeline,
            &self.normalise_bind_group_layout,
        )
        .await
    }

    pub async fn simulate(&self, heightmap: &Array2<f32>) -> Array2<f32> {
        let (rows, cols) = (self.rows, self.cols);
        let mut distances = Array2::<f32>::zeros((rows as usize, cols as usize));

        // Convert to slices
        let in_data = heightmap.as_slice().unwrap();
        let out_data = distances.as_slice().unwrap(); // starts at 0

        // We assume each pixel is one float in the input, one float in output
        let result_floats = self
            .run_two_buffer_op(
                in_data,
                out_data,
                &self.simulation_pipeline,
                &self.simulation_bind_group_layout,
                4, // input is f32 => 4 bytes each
                4, // output is f32 => 4 bytes each
            )
            .await;

        // Copy them into `distances`
        distances
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(&result_floats);

        distances
    }

    pub async fn gradient(&self, heightmap: &Array2<f32>) -> Array3<f32> {
        let (rows, cols) = (self.rows, self.cols);
        // Output shape = (rows, cols, 2) => total 2 * rows * cols floats
        let out_len = (rows * cols * 2) as usize;

        let in_data = heightmap.as_slice().unwrap();
        let out_data = vec![0.0f32; out_len];

        // Send them to run_two_buffer_op
        let result_floats = self
            .run_two_buffer_op(
                in_data,
                &out_data,
                &self.gradient_pipeline,
                &self.gradient_bind_group_layout,
                4, // each input item is one f32
                4, // each output item is one f32, but 2 per pixel => we handle that as an array
            )
            .await;

        // Convert result_floats into Array3
        Array3::from_shape_vec((rows as usize, cols as usize, 2), result_floats).unwrap()
    }

    pub async fn magnitude(&self, gradient: &Array3<f32>) -> Array2<f32> {
        let (rows, cols) = (self.rows, self.cols);

        // Flatten shape [rows, cols, 2] => 2 * rows * cols floats for the gradient
        let grad_data = gradient.as_slice().unwrap();
        // We'll output [rows, cols] => rows * cols floats
        let mut magnitudes = Array2::<f32>::zeros((rows as usize, cols as usize));
        let mag_data = magnitudes.as_slice().unwrap(); // initially all zero

        // We'll read from the gradient, write into the magnitudes array
        let result_floats = self
            .run_two_buffer_op(
                grad_data, // input
                mag_data,  // output
                &self.magnitude_pipeline,
                &self.magnitude_bind_group_layout,
                4, // each gradient component is f32 => 4 bytes, total 2 comps per pixel => pass them as f32 slice
                4, // each magnitude is a single f32 => 4 bytes
            )
            .await;

        // Copy results back into the Array2
        magnitudes
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(&result_floats);

        magnitudes
    }
}
