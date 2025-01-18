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

struct PipelineInfo {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    num_bind_entries: usize, // The total number of entries this pipeline expects
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: HashMap<&'static str, PipelineInfo>,
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
        let (invert_pipeline, invert_layout) = Self::create_pipeline(
            &device,
            &INVERT_SHADER_WGSL,
            INVERT_SHADER_LAYOUT,
            &format!("{} Pipeline", INVERT_LABEL),
            rows,
            cols,
        );
        pipelines.insert(
            "invert",
            PipelineInfo {
                pipeline: invert_pipeline,
                bind_group_layout: invert_layout,
                num_bind_entries: 1, // Just one read-write buffer
            },
        );

        // Multiply pipeline
        let (multiply_pipeline, multiply_layout) = Self::create_pipeline(
            &device,
            MULTIPLY_SHADER_WGSL,
            MULTIPLY_SHADER_LAYOUT,
            &format!("{} Pipeline", MULTIPLY_LABEL),
            rows,
            cols,
        );
        pipelines.insert(
            MULTIPLY_LABEL,
            PipelineInfo {
                pipeline: multiply_pipeline,
                bind_group_layout: multiply_layout,
                num_bind_entries: 2,
            },
        );

        // Add pipeline
        let (add_pipeline, add_layout) = Self::create_pipeline(
            &device,
            ADD_SHADER_WGSL,
            ADD_SHADER_LAYOUT,
            &format!("{} Pipeline", ADD_LABEL),
            rows,
            cols,
        );
        pipelines.insert(
            ADD_LABEL,
            PipelineInfo {
                pipeline: add_pipeline,
                bind_group_layout: add_layout,
                num_bind_entries: 2,
            },
        );

        // Normalise pipeline
        let (normalise_pipeline, normalise_layout) = Self::create_pipeline(
            &device,
            NORMALISE_SHADER_WGSL,
            NORMALISE_SHADER_LAYOUT,
            &format!("{} Pipeline", NORMALISE_LABEL),
            rows,
            cols,
        );
        pipelines.insert(
            NORMALISE_LABEL,
            PipelineInfo {
                pipeline: normalise_pipeline,
                bind_group_layout: normalise_layout,
                num_bind_entries: 2,
            },
        );

        // Gradient pipeline
        let (gradient_pipeline, gradient_layout) = Self::create_pipeline(
            &device,
            GRADIENT_SHADER_WGSL,
            GRADIENT_SHADER_LAYOUT,
            &format!("{} Pipeline", GRADIENT_LABEL),
            rows,
            cols,
        );
        pipelines.insert(
            GRADIENT_LABEL,
            PipelineInfo {
                pipeline: gradient_pipeline,
                bind_group_layout: gradient_layout,
                num_bind_entries: 2,
            },
        );

        // Gradient magnitude pipeline
        let (magnitude_pipeline, magnitude_layout) = Self::create_pipeline(
            &device,
            MAGNITUDE_SHADER_WGSL,
            MAGNITUDE_SHADER_LAYOUT,
            &format!("{} Pipeline", MAGNITUDE_LABEL),
            rows,
            cols,
        );
        pipelines.insert(
            MAGNITUDE_LABEL,
            PipelineInfo {
                pipeline: magnitude_pipeline,
                bind_group_layout: magnitude_layout,
                num_bind_entries: 2,
            },
        );

        // Flow pipeline
        let (flow_pipeline, flow_layout) = Self::create_pipeline(
            &device,
            FLOW_SHADER_WGSL,
            FLOW_SHADER_LAYOUT,
            &format!("{} Pipeline", FLOW_LABEL),
            rows,
            cols,
        );
        pipelines.insert(
            FLOW_LABEL,
            PipelineInfo {
                pipeline: flow_pipeline,
                bind_group_layout: flow_layout,
                num_bind_entries: 3,
            },
        );

        // Smooth pipeline
        let (smooth_pipeline, smooth_layout) = Self::create_pipeline(
            &device,
            SMOOTH_SHADER_WGSL,
            SMOOTH_SHADER_LAYOUT,
            &format!("{} Pipeline", SMOOTH_LABEL),
            rows,
            cols,
        );
        pipelines.insert(
            SMOOTH_LABEL,
            PipelineInfo {
                pipeline: smooth_pipeline,
                bind_group_layout: smooth_layout,
                num_bind_entries: 2,
            },
        );

        // Contour pipeline
        let (contour_pipeline, contour_layout) = Self::create_pipeline(
            &device,
            CONTOUR_SHADER_WGSL,
            CONTOUR_SHADER_LAYOUT,
            &format!("{} Pipeline", CONTOUR_LABEL),
            rows,
            cols,
        );
        pipelines.insert(
            CONTOUR_LABEL,
            PipelineInfo {
                pipeline: contour_pipeline,
                bind_group_layout: contour_layout,
                num_bind_entries: 2,
            },
        );

        // Colour pipeline
        let (colour_pipeline, colour_layout) = Self::create_pipeline(
            &device,
            COLOUR_SHADER_WGSL,
            COLOUR_SHADER_LAYOUT,
            &format!("{} Pipeline", COLOUR_LABEL),
            rows,
            cols,
        );
        pipelines.insert(
            COLOUR_LABEL,
            PipelineInfo {
                pipeline: colour_pipeline,
                bind_group_layout: colour_layout,
                num_bind_entries: 3,
            },
        );

        // Quantise pipeline
        let (quantize_pipeline, quantize_layout) = Self::create_pipeline(
            &device,
            QUANTISE_SHADER_WGSL,
            QUANTISE_SHADER_LAYOUT,
            &format!("{} Pipeline", QUANTISE_LABEL),
            rows,
            cols,
        );
        pipelines.insert(
            QUANTISE_LABEL,
            PipelineInfo {
                pipeline: quantize_pipeline,
                bind_group_layout: quantize_layout,
                num_bind_entries: 3,
            },
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
            // optional
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        (pipeline, bind_group_layout)
    }

    /// Our single dispatch function that decides whether to do “in-place” or
    /// “separate-output” based on the pipeline's declared number of bind entries.
    pub async fn run_op(
        &self,
        pipeline_name: &str,
        input_slices: &[&[f32]],
        output_len: usize,
        optional_uniform: Option<&[f32]>,
    ) -> Vec<f32> {
        let info = self
            .pipelines
            .get(pipeline_name)
            .expect("No pipeline found with that name");

        let pipeline = &info.pipeline;
        let layout = &info.bind_group_layout;
        let expected = info.num_bind_entries;

        // Count how many “binding slots” we’ll fill with data buffers + uniform
        let uniform_count = if optional_uniform.is_some() { 1 } else { 0 };
        let total_inputs = input_slices.len() + uniform_count;

        // If the pipeline’s total expected binding entries == total inputs,
        // we assume it’s an “in-place” pipeline. Otherwise, we assume there's
        // a dedicated output buffer.
        let is_in_place = expected == total_inputs;

        // 1) Create buffers for each input slice
        //    Each is read-write, since some pipelines do in-place updates.
        //    We might do multi-buffer input if your pipeline expects it.
        let mut staging_in = Vec::new();
        let mut gpu_in = Vec::new();
        for (i, slice) in input_slices.iter().enumerate() {
            let size_bytes = (slice.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
            let sbuf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Staging In {i}")),
                    contents: bytemuck::cast_slice(slice),
                    usage: wgpu::BufferUsages::COPY_SRC,
                });
            let gbuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("GPU In {i}")),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            staging_in.push(sbuf);
            gpu_in.push(gbuf);
        }

        // If we have a uniform, make a uniform buffer for it:
        let uniform_buf = if let Some(u_data) = optional_uniform {
            Some(
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Uniform Buffer"),
                        contents: bytemuck::cast_slice(u_data),
                        usage: wgpu::BufferUsages::UNIFORM,
                    }),
            )
        } else {
            None
        };

        // 2) If the pipeline is in-place, we do NOT create a separate “output” buffer.
        //    Instead, we treat the last GPU buffer as the “output” after the shader modifies it.
        //    If the pipeline expects a separate output buffer, we create one.
        let (gpu_out, staging_out) = if is_in_place {
            // No separate output, we’ll just read back from gpu_in[0] or whichever
            // buffer is relevant. In simpler pipelines, there's only one buffer.
            (None, None)
        } else {
            // We do the multi-buffer approach
            let size_out_bytes = (output_len * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
            let zeroes = vec![0.0_f32; output_len];
            let sbuf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Staging Out"),
                    contents: bytemuck::cast_slice(&zeroes),
                    usage: wgpu::BufferUsages::COPY_SRC,
                });
            let gbuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GPU Out"),
                size: size_out_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            (Some(gbuf), Some(sbuf))
        };

        // 3) Copy all staging input slices → GPU
        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Init Op Encoder"),
                });
            for (sbuf, gbuf) in staging_in.iter().zip(gpu_in.iter()) {
                enc.copy_buffer_to_buffer(sbuf, 0, gbuf, 0, sbuf.size());
            }
            // If we have a separate output, zero it too:
            if let (Some(staging_out), Some(gpu_out)) = (&staging_out, &gpu_out) {
                enc.copy_buffer_to_buffer(staging_out, 0, gpu_out, 0, staging_out.size());
            }
            self.queue.submit(Some(enc.finish()));
        }

        // 4) Create the bind group with exactly the correct number of entries
        let mut entries = Vec::new();
        // a) Each input buffer (treated as read-write)
        for (i, gbuf) in gpu_in.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::Buffer(gbuf.as_entire_buffer_binding()),
            });
        }
        // b) Uniform, if present
        let mut next_binding = gpu_in.len() as u32;
        if let Some(ubuf) = &uniform_buf {
            entries.push(wgpu::BindGroupEntry {
                binding: next_binding,
                resource: wgpu::BindingResource::Buffer(ubuf.as_entire_buffer_binding()),
            });
            next_binding += 1;
        }
        // c) If not in-place, add the separate output buffer
        if let Some(gpu_out) = &gpu_out {
            entries.push(wgpu::BindGroupEntry {
                binding: next_binding,
                resource: wgpu::BindingResource::Buffer(gpu_out.as_entire_buffer_binding()),
            });
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &entries,
            label: Some("Op BindGroup"),
        });

        // 5) Dispatch
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compute Op Encoder"),
                });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Op Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                let (wx, wy, wz) = self.workgroup_size;
                let nx = (self.cols + wx - 1) / wx;
                let ny = (self.rows + wy - 1) / wy;
                cpass.dispatch_workgroups(nx, ny, wz);
            }
            self.queue.submit(Some(encoder.finish()));
        }

        // 6) Read back
        //    - If in-place, read from the buffer you wrote in
        //    - If separate, read from that dedicated “gpu_out”
        let (read_src, read_size) = if is_in_place {
            // We’ll just read from the “first” input buffer in this example.
            // Real code might read from a specific buffer if there are multiple.
            (
                gpu_in[0].clone(),
                gpu_in[0].size(), // same as input length in bytes
            )
        } else {
            let actual_out = gpu_out.expect("We have a dedicated output in multi-buffer path");
            let size_out = actual_out.size();
            (actual_out, size_out)
        };

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback"),
            size: read_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy Back"),
                });
            enc.copy_buffer_to_buffer(&read_src, 0, &readback, 0, read_size);
            self.queue.submit(Some(enc.finish()));
        }

        // Wait & map
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

        result
    }

    pub async fn invert(&self, input: &Array2<f32>) -> Array2<f32> {
        let in_data = input.as_slice().unwrap();
        let out_len = (self.rows * self.cols) as usize;

        // We call run_op("invert", ...)
        let result = self.run_op(INVERT_LABEL, &[in_data], out_len, None).await;

        Array2::from_shape_vec((self.rows as usize, self.cols as usize), result).unwrap()
    }

    pub async fn multiply(&self, input: &Array2<f32>, scale: f32) -> Array2<f32> {
        let in_data = input.as_slice().unwrap();
        let out_len = (self.rows * self.cols) as usize;

        // Pass the uniform as a slice of floats
        let scale_data = [scale];

        let result = self
            .run_op(MULTIPLY_LABEL, &[in_data], out_len, Some(&scale_data))
            .await;

        Array2::from_shape_vec((self.rows as usize, self.cols as usize), result).unwrap()
    }

    pub async fn add(&self, input: &Array2<f32>, offset: f32) -> Array2<f32> {
        let in_data = input.as_slice().unwrap();
        let out_len = (self.rows * self.cols) as usize;

        let offset_data = [offset];

        let result = self
            .run_op(ADD_LABEL, &[in_data], out_len, Some(&offset_data))
            .await;

        Array2::from_shape_vec((self.rows as usize, self.cols as usize), result).unwrap()
    }

    pub async fn normalise(&self, input: &Array2<f32>) -> Array2<f32> {
        let min_val = input.fold(f32::MAX, |a, &b| a.min(b));
        let max_val = input.fold(f32::MIN, |a, &b| a.max(b));

        if (max_val - min_val).abs() < f32::EPSILON {
            return input.clone();
        }
        let range = max_val - min_val;
        let param_data = [min_val, range];

        let in_data = input.as_slice().unwrap();
        let out_len = (self.rows * self.cols) as usize;

        let result = self
            .run_op(NORMALISE_LABEL, &[in_data], out_len, Some(&param_data))
            .await;

        Array2::from_shape_vec((self.rows as usize, self.cols as usize), result).unwrap()
    }

    pub async fn gradient(&self, heightmap: &Array2<f32>) -> Array3<f32> {
        let in_data = heightmap.as_slice().unwrap();
        let out_len = (self.rows * self.cols * 2) as usize;

        // No uniform
        let result = self.run_op(GRADIENT_LABEL, &[in_data], out_len, None).await;

        Array3::from_shape_vec((self.rows as usize, self.cols as usize, 2), result).unwrap()
    }

    pub async fn magnitude(&self, gradient: &Array3<f32>) -> Array2<f32> {
        let in_data = gradient.as_slice().unwrap();
        let out_len = (self.rows * self.cols) as usize;

        let result = self
            .run_op(MAGNITUDE_LABEL, &[in_data], out_len, None)
            .await;

        Array2::from_shape_vec((self.rows as usize, self.cols as usize), result).unwrap()
    }

    pub async fn smooth(&self, input: &Array2<f32>) -> Array2<f32> {
        let in_data = input.as_slice().unwrap();
        let out_len = (self.rows * self.cols) as usize;

        let result = self.run_op(SMOOTH_LABEL, &[in_data], out_len, None).await;

        Array2::from_shape_vec((self.rows as usize, self.cols as usize), result).unwrap()
    }

    pub async fn contour(&self, input: &Array2<f32>) -> Array2<f32> {
        let in_data = input.as_slice().unwrap();
        let out_len = (self.rows * self.cols) as usize;

        let result = self.run_op(CONTOUR_LABEL, &[in_data], out_len, None).await;

        Array2::from_shape_vec((self.rows as usize, self.cols as usize), result).unwrap()
    }

    pub async fn quantise(&self, input: &Array2<f32>, num_steps: u32) -> Array2<f32> {
        let in_data = input.as_slice().unwrap();
        let out_len = (self.rows * self.cols) as usize;

        let steps_data = [num_steps as f32];
        let result = self
            .run_op(QUANTISE_LABEL, &[in_data], out_len, Some(&steps_data))
            .await;

        Array2::from_shape_vec((self.rows as usize, self.cols as usize), result).unwrap()
    }

    pub async fn flow(&self, heightmap: &Array2<f32>, gradient_map: &Array3<f32>) -> Array3<f32> {
        let h_slice = heightmap.as_slice().unwrap();
        let g_slice = gradient_map.as_slice().unwrap();
        let out_len = (self.rows * self.cols * 2) as usize;

        // Two input slices, no uniform
        let result = self
            .run_op(FLOW_LABEL, &[h_slice, g_slice], out_len, None)
            .await;

        Array3::from_shape_vec((self.rows as usize, self.cols as usize, 2), result).unwrap()
    }

    pub async fn colour(&self, heightmap: &Array2<f32>, colours: &[[f32; 4]]) -> Array3<f32> {
        let height_slice = heightmap.as_slice().unwrap();
        let mut colour_data = Vec::new();
        for c in colours {
            colour_data.extend_from_slice(c);
        }
        let out_len = (self.rows * self.cols * 4) as usize;

        // Two inputs: height slice + flattened RGBA array
        let result = self
            .run_op(COLOUR_LABEL, &[height_slice, &colour_data], out_len, None)
            .await;

        Array3::from_shape_vec((self.rows as usize, self.cols as usize, 4), result).unwrap()
    }
}
