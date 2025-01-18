use std::collections::HashMap;

use ndarray::{Array2, Array3};
use wgpu::util::DeviceExt;

const INVERT_LABEL: &str = "invert";
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

const MULTIPLY_LABEL: &str = "multiply";
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

const ADD_LABEL: &str = "add";
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

const NORMALISE_LABEL: &str = "normalise";
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

const GRADIENT_LABEL: &str = "gradient";
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

const FLOW_LABEL: &str = "flow";
const FLOW_SHADER_WGSL: &str = include_str!("shaders/flow.wgsl");
const FLOW_SHADER_LAYOUT: &[wgpu::BindGroupLayoutEntry] = &[
    // Heightmap (read-only)
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
    // Gradient map (read-write)
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
    // Output buffer (read-write)
    wgpu::BindGroupLayoutEntry {
        binding: 2,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
];

const MAGNITUDE_LABEL: &str = "magnitude";
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

const COLOUR_LABEL: &str = "colour";
const COLOUR_SHADER_WGSL: &str = include_str!("shaders/colour.wgsl");
const COLOUR_SHADER_LAYOUT: &[wgpu::BindGroupLayoutEntry] = &[
    // heightmap (read-only)
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
    // colourmap (read-only)
    wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
    // rgb_out (read-write)
    wgpu::BindGroupLayoutEntry {
        binding: 2,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
];

const SMOOTH_LABEL: &str = "smooth";
const SMOOTH_SHADER_WGSL: &str = include_str!("shaders/smooth.wgsl");
const SMOOTH_SHADER_LAYOUT: &[wgpu::BindGroupLayoutEntry] = &[
    // data (read-only)
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
    // output (read-write)
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

const QUANTISE_LABEL: &str = "quantize";
const QUANTISE_SHADER_WGSL: &str = include_str!("shaders/quantize.wgsl");
const QUANTISE_SHADER_LAYOUT: &[wgpu::BindGroupLayoutEntry] = &[
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
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
    wgpu::BindGroupLayoutEntry {
        binding: 2,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
];

const CONTOUR_LABEL: &str = "contour";
const CONTOUR_SHADER_WGSL: &str = include_str!("shaders/contour.wgsl");
const CONTOUR_SHADER_LAYOUT: &[wgpu::BindGroupLayoutEntry] = &[
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
    pipelines: HashMap<&'static str, (wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
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

        // Invert pipeline
        let mut pipelines = HashMap::new();
        let (invert_pipeline, invert_bind_group_layout) = Self::create_pipeline(
            &device,
            &INVERT_SHADER_WGSL,
            INVERT_SHADER_LAYOUT,
            &format!("{} Pipeline", INVERT_LABEL),
            rows,
            cols,
        );
        pipelines.insert(INVERT_LABEL, (invert_pipeline, invert_bind_group_layout));

        // Multiply pipeline
        let (multiply_pipeline, multiply_bind_group_layout) = Self::create_pipeline(
            &device,
            MULTIPLY_SHADER_WGSL,
            MULTIPLY_SHADER_LAYOUT,
            &format!("{} Pipeline", MULTIPLY_LABEL),
            rows,
            cols,
        );
        pipelines.insert(
            MULTIPLY_LABEL,
            (multiply_pipeline, multiply_bind_group_layout),
        );

        // Add pipeline
        let (add_pipeline, add_bind_group_layout) = Self::create_pipeline(
            &device,
            ADD_SHADER_WGSL,
            ADD_SHADER_LAYOUT,
            &format!("{} Pipeline", ADD_LABEL),
            rows,
            cols,
        );
        pipelines.insert(ADD_LABEL, (add_pipeline, add_bind_group_layout));

        // Normalise pipeline
        let (normalise_pipeline, normalise_bind_group_layout) = Self::create_pipeline(
            &device,
            NORMALISE_SHADER_WGSL,
            NORMALISE_SHADER_LAYOUT,
            &format!("{} Pipeline", NORMALISE_LABEL),
            rows,
            cols,
        );
        pipelines.insert(
            NORMALISE_LABEL,
            (normalise_pipeline, normalise_bind_group_layout),
        );

        // Gradient pipeline
        let (gradient_pipeline, gradient_bind_group_layout) = Self::create_pipeline(
            &device,
            GRADIENT_SHADER_WGSL,
            GRADIENT_SHADER_LAYOUT,
            &format!("{} Pipeline", GRADIENT_LABEL),
            rows,
            cols,
        );
        pipelines.insert(
            GRADIENT_LABEL,
            (gradient_pipeline, gradient_bind_group_layout),
        );

        // Gradient magnitude pipeline
        let (magnitude_pipeline, magnitude_bind_group_layout) = Self::create_pipeline(
            &device,
            MAGNITUDE_SHADER_WGSL,
            MAGNITUDE_SHADER_LAYOUT,
            &format!("{} Pipeline", MAGNITUDE_LABEL),
            rows,
            cols,
        );
        pipelines.insert(
            MAGNITUDE_LABEL,
            (magnitude_pipeline, magnitude_bind_group_layout),
        );

        // Flow pipeline
        let (flow_pipeline, flow_bind_group_layout) = Self::create_pipeline(
            &device,
            FLOW_SHADER_WGSL,
            FLOW_SHADER_LAYOUT,
            &format!("{} Pipeline", FLOW_LABEL),
            rows,
            cols,
        );
        pipelines.insert(FLOW_LABEL, (flow_pipeline, flow_bind_group_layout));

        // Smooth pipeline
        let (smooth_pipeline, smooth_bind_group_layout) = Self::create_pipeline(
            &device,
            SMOOTH_SHADER_WGSL,
            SMOOTH_SHADER_LAYOUT,
            &format!("{} Pipeline", SMOOTH_LABEL),
            rows,
            cols,
        );
        pipelines.insert(SMOOTH_LABEL, (smooth_pipeline, smooth_bind_group_layout));

        // Contour pipeline
        let (contour_pipeline, contour_bind_group_layout) = Self::create_pipeline(
            &device,
            CONTOUR_SHADER_WGSL,
            CONTOUR_SHADER_LAYOUT,
            &format!("{} Pipeline", CONTOUR_LABEL),
            rows,
            cols,
        );
        pipelines.insert(CONTOUR_LABEL, (contour_pipeline, contour_bind_group_layout));

        // Colour pipeline
        let (colour_pipeline, colour_bind_group_layout) = Self::create_pipeline(
            &device,
            COLOUR_SHADER_WGSL,
            COLOUR_SHADER_LAYOUT,
            &format!("{} Pipeline", COLOUR_LABEL),
            rows,
            cols,
        );
        pipelines.insert(COLOUR_LABEL, (colour_pipeline, colour_bind_group_layout));

        // Quantise pipeline
        let (quantize_pipeline, quantize_bind_group_layout) = Self::create_pipeline(
            &device,
            QUANTISE_SHADER_WGSL,
            QUANTISE_SHADER_LAYOUT,
            &format!("{} Pipeline", QUANTISE_LABEL),
            rows,
            cols,
        );
        pipelines.insert(
            QUANTISE_LABEL,
            (quantize_pipeline, quantize_bind_group_layout),
        );

        Self {
            device,
            queue,
            pipelines,
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

    // SINGLE BUFFER OP (with optional uniform)
    async fn run_single_buffer_op(
        &self,
        input: &Array2<f32>,
        pipeline: &wgpu::ComputePipeline,
        layout: &wgpu::BindGroupLayout,
        optional_uniform: Option<&wgpu::Buffer>,
    ) -> Array2<f32> {
        let size_bytes =
            (self.rows * self.cols * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress;

        // 1) Stage + GPU buffer
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

        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Single Init Encoder"),
                });
            encoder.copy_buffer_to_buffer(&staging_buf, 0, &gpu_buf, 0, size_bytes);
            self.queue.submit(Some(encoder.finish()));
        }

        // 2) Bind group
        let bind_entries = if let Some(u_buf) = optional_uniform {
            vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(gpu_buf.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(u_buf.as_entire_buffer_binding()),
                },
            ]
        } else {
            vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(gpu_buf.as_entire_buffer_binding()),
            }]
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &bind_entries,
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
                // dispatch with integer division rounding up
                let nx = (self.cols + wx - 1) / wx;
                let ny = (self.rows + wy - 1) / wy;
                cpass.dispatch_workgroups(nx, ny, wz);
            }
            self.queue.submit(Some(encoder.finish()));
        }

        // 4) Readback
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

        {
            let slice = readback.slice(..);
            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
            self.device.poll(wgpu::Maintain::Wait);
            rx.receive().await.unwrap().unwrap();
        }

        let data = readback.slice(..).get_mapped_range();
        let result_vec = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        readback.unmap();

        Array2::from_shape_vec((self.rows as usize, self.cols as usize), result_vec).unwrap()
    }

    // TWO BUFFER OP (with optional uniform)
    async fn run_two_buffer_op(
        &self,
        input_bytes: &[f32],
        output_bytes: &[f32],
        pipeline: &wgpu::ComputePipeline,
        layout: &wgpu::BindGroupLayout,
        bytes_per_input_item: usize,
        bytes_per_output_item: usize,
        optional_uniform: Option<&wgpu::Buffer>,
    ) -> Vec<f32> {
        let size_in = (input_bytes.len() * bytes_per_input_item) as wgpu::BufferAddress;
        let size_out = (output_bytes.len() * bytes_per_output_item) as wgpu::BufferAddress;

        // 1) Create staging buffers
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

        // 2) Create GPU buffers
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

        // 3) Copy CPU → GPU
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

        // 4) Bind group
        let bind_entries = if let Some(u_buf) = optional_uniform {
            vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(gpu_in.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(u_buf.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(gpu_out.as_entire_buffer_binding()),
                },
            ]
        } else {
            vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(gpu_in.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(gpu_out.as_entire_buffer_binding()),
                },
            ]
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &bind_entries,
            label: Some("Two-Buffers BindGroup"),
        });

        // 5) Dispatch
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
                let (wx, wy, wz) = self.workgroup_size;
                let nx = (self.cols + wx - 1) / wx;
                let ny = (self.rows + wy - 1) / wy;
                cpass.dispatch_workgroups(nx, ny, wz);
            }
            self.queue.submit(Some(enc.finish()));
        }

        // 6) Read back
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

    // THREE BUFFER OP (with optional uniform)
    async fn run_three_buffer_op(
        &self,
        input_a_bytes: &[f32],
        input_b_bytes: &[f32],
        output_bytes: &[f32],
        pipeline: &wgpu::ComputePipeline,
        layout: &wgpu::BindGroupLayout,
        bytes_per_a_item: usize,
        bytes_per_b_item: usize,
        bytes_per_output_item: usize,
        optional_uniform: Option<&wgpu::Buffer>,
    ) -> Vec<f32> {
        let size_a = (input_a_bytes.len() * bytes_per_a_item) as wgpu::BufferAddress;
        let size_b = (input_b_bytes.len() * bytes_per_b_item) as wgpu::BufferAddress;
        let size_out = (output_bytes.len() * bytes_per_output_item) as wgpu::BufferAddress;

        // 1) Create staging buffers
        let staging_a = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Staging A"),
                contents: bytemuck::cast_slice(input_a_bytes),
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        let staging_b = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Staging B"),
                contents: bytemuck::cast_slice(input_b_bytes),
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        let staging_out = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Staging Out"),
                contents: bytemuck::cast_slice(output_bytes),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // 2) Create GPU buffers
        let gpu_a = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU A"),
            size: size_a,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gpu_b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU B"),
            size: size_b,
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

        // 3) Copy CPU → GPU
        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Init Three-Buffers"),
                });
            enc.copy_buffer_to_buffer(&staging_a, 0, &gpu_a, 0, size_a);
            enc.copy_buffer_to_buffer(&staging_b, 0, &gpu_b, 0, size_b);
            enc.copy_buffer_to_buffer(&staging_out, 0, &gpu_out, 0, size_out);
            self.queue.submit(Some(enc.finish()));
        }

        // 4) Bind group
        let bind_entries = if let Some(u_buf) = optional_uniform {
            vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(gpu_a.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(gpu_b.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(u_buf.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(gpu_out.as_entire_buffer_binding()),
                },
            ]
        } else {
            vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(gpu_a.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(gpu_b.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(gpu_out.as_entire_buffer_binding()),
                },
            ]
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &bind_entries,
            label: Some("Three-Buffers BindGroup"),
        });

        // 5) Dispatch
        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compute Three-Buffers"),
                });
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Three-Buffers Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                let (wx, wy, wz) = self.workgroup_size;
                let nx = (self.cols + wx - 1) / wx;
                let ny = (self.rows + wy - 1) / wy;
                cpass.dispatch_workgroups(nx, ny, wz);
            }
            self.queue.submit(Some(enc.finish()));
        }

        // 6) Read back
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Three-Buffers Readback"),
            size: size_out,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy Three-Buffers"),
                });
            enc.copy_buffer_to_buffer(&gpu_out, 0, &readback, 0, size_out);
            self.queue.submit(Some(enc.finish()));
        }

        {
            let slice = readback.slice(..);
            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
            self.device.poll(wgpu::Maintain::Wait);
            rx.receive().await.unwrap().unwrap();
        }

        let data = readback.slice(..).get_mapped_range();
        let result_floats = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        readback.unmap();
        result_floats
    }

    // Single-buffer examples

    pub async fn invert(&self, input: &Array2<f32>) -> Array2<f32> {
        let (pipeline, layout) = self
            .pipelines
            .get(INVERT_LABEL)
            .expect(&format!("{} pipeline not found", INVERT_LABEL));
        self.run_single_buffer_op(input, pipeline, layout, None)
            .await
    }

    pub async fn multiply(&self, input: &Array2<f32>, scale: f32) -> Array2<f32> {
        let scale_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Multiply Uniform"),
                contents: bytemuck::cast_slice(&[scale]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let (pipeline, layout) = self
            .pipelines
            .get(MULTIPLY_LABEL)
            .expect(&format!("{} pipeline not found", MULTIPLY_LABEL));
        self.run_single_buffer_op(input, pipeline, layout, Some(&scale_buf))
            .await
    }

    pub async fn add(&self, input: &Array2<f32>, offset: f32) -> Array2<f32> {
        let offset_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Add Offset Uniform"),
                contents: bytemuck::cast_slice(&[offset]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let (pipeline, layout) = self
            .pipelines
            .get(ADD_LABEL)
            .expect(&format!("{} pipeline not found", ADD_LABEL));
        self.run_single_buffer_op(input, pipeline, layout, Some(&offset_buf))
            .await
    }

    pub async fn normalise(&self, input: &Array2<f32>) -> Array2<f32> {
        let min_val = input.fold(f32::MAX, |a, &b| a.min(b));
        let max_val = input.fold(f32::MIN, |a, &b| a.max(b));
        if (max_val - min_val).abs() < f32::EPSILON {
            return input.clone();
        }
        let range = max_val - min_val;
        let param_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Normalise Param Uniform"),
                contents: bytemuck::cast_slice(&[min_val, range]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let (pipeline, layout) = self
            .pipelines
            .get(NORMALISE_LABEL)
            .expect(&format!("{} pipeline not found", NORMALISE_LABEL));
        self.run_single_buffer_op(input, pipeline, layout, Some(&param_buf))
            .await
    }

    // Two-buffer no uniform

    pub async fn gradient(&self, heightmap: &Array2<f32>) -> Array3<f32> {
        let in_data = heightmap.as_slice().unwrap();
        let out_len = (self.rows * self.cols * 2) as usize;
        let out_data = vec![0.0; out_len];

        let (pipeline, layout) = self
            .pipelines
            .get(GRADIENT_LABEL)
            .expect(&format!("{} pipeline not found", GRADIENT_LABEL));
        let result = self
            .run_two_buffer_op(in_data, &out_data, pipeline, layout, 4, 4, None)
            .await;
        Array3::from_shape_vec((self.rows as usize, self.cols as usize, 2), result).unwrap()
    }

    pub async fn magnitude(&self, gradient: &Array3<f32>) -> Array2<f32> {
        let grad_data = gradient.as_slice().unwrap();
        let mut mags = Array2::<f32>::zeros((self.rows as usize, self.cols as usize));
        let mag_slice = mags.as_slice().unwrap();

        let (pipeline, layout) = self
            .pipelines
            .get(MAGNITUDE_LABEL)
            .expect(&format!("{} pipeline not found", MAGNITUDE_LABEL));
        let result = self
            .run_two_buffer_op(grad_data, mag_slice, pipeline, layout, 4, 4, None)
            .await;
        mags.as_slice_mut().unwrap().copy_from_slice(&result);
        mags
    }

    pub async fn smooth(&self, input: &Array2<f32>) -> Array2<f32> {
        let in_data = input.as_slice().unwrap();
        let mut out_arr = Array2::<f32>::zeros((self.rows as usize, self.cols as usize));
        let out_data = out_arr.as_slice().unwrap();

        let (pipeline, layout) = self
            .pipelines
            .get(SMOOTH_LABEL)
            .expect(&format!("{} pipeline not found", SMOOTH_LABEL));
        let result = self
            .run_two_buffer_op(in_data, out_data, pipeline, layout, 4, 4, None)
            .await;
        out_arr.as_slice_mut().unwrap().copy_from_slice(&result);
        out_arr
    }

    pub async fn contour(&self, input: &Array2<f32>) -> Array2<f32> {
        let in_data = input.as_slice().unwrap();
        let mut out_arr = Array2::<f32>::zeros((self.rows as usize, self.cols as usize));
        let out_data = out_arr.as_slice().unwrap();

        let (pipeline, layout) = self
            .pipelines
            .get(CONTOUR_LABEL)
            .expect(&format!("{} pipeline not found", CONTOUR_LABEL));
        let result = self
            .run_two_buffer_op(in_data, out_data, pipeline, layout, 4, 4, None)
            .await;
        out_arr.as_slice_mut().unwrap().copy_from_slice(&result);
        out_arr
    }

    // Two-buffer with uniform

    pub async fn quantise(&self, input: &Array2<f32>, num_steps: u32) -> Array2<f32> {
        let in_data = input.as_slice().unwrap();
        let mut out_arr = Array2::<f32>::zeros((self.rows as usize, self.cols as usize));
        let out_data = out_arr.as_slice().unwrap();

        let steps_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Num Steps Uniform"),
                contents: bytemuck::cast_slice(&[num_steps]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let (pipeline, layout) = self
            .pipelines
            .get(QUANTISE_LABEL)
            .expect(&format!("{} pipeline not found", QUANTISE_LABEL));
        let result = self
            .run_two_buffer_op(in_data, out_data, pipeline, layout, 4, 4, Some(&steps_buf))
            .await;
        out_arr.as_slice_mut().unwrap().copy_from_slice(&result);
        out_arr
    }

    // Three-buffer examples

    pub async fn flow(&self, heightmap: &Array2<f32>, gradient_map: &Array3<f32>) -> Array3<f32> {
        let h_slice = heightmap.as_slice().unwrap();
        let g_slice = gradient_map.as_slice().unwrap();
        let out_len = (self.rows * self.cols * 2) as usize;
        let out_data = vec![0.0f32; out_len];

        let (pipeline, layout) = self
            .pipelines
            .get(FLOW_LABEL)
            .expect(&format!("{} pipeline not found", FLOW_LABEL));
        let result = self
            .run_three_buffer_op(h_slice, g_slice, &out_data, pipeline, layout, 4, 4, 4, None)
            .await;
        Array3::from_shape_vec((self.rows as usize, self.cols as usize, 2), result).unwrap()
    }

    pub async fn colour(&self, heightmap: &Array2<f32>, colours: &[[f32; 4]]) -> Array3<f32> {
        let height_slice = heightmap.as_slice().unwrap();

        // Flatten the colours
        let mut colour_data = Vec::new();
        for c in colours {
            colour_data.extend_from_slice(c);
        }
        let out_len = (self.rows * self.cols * 4) as usize;
        let out_data = vec![0.0f32; out_len];

        let (pipeline, layout) = self
            .pipelines
            .get(COLOUR_LABEL)
            .expect(&format!("{} pipeline not found", COLOUR_LABEL));
        let result = self
            .run_three_buffer_op(
                height_slice,
                &colour_data,
                &out_data,
                pipeline,
                layout,
                4,
                4,
                4,
                None, // no uniform used here
            )
            .await;
        Array3::from_shape_vec((self.rows as usize, self.cols as usize, 4), result).unwrap()
    }
}
