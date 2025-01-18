@group(0) @binding(0)
var<storage, read> input_data : array<f32>;

@group(0) @binding(1)
var<uniform> num_steps : f32;

@group(0) @binding(2)
var<storage, read_write> output_data : array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3 < u32>)
{
    let rows = i32(@rows);
    let cols = i32(@cols);

    let x = i32(gid.x);
    let y = i32(gid.y);
    let i = y * cols + x;

    if (x >= cols || y >= rows)
    {
        return;
    }

    //Read input value
    let value = input_data[u32(i)];

    //Compute the step size
    let step_size = 1.0 / num_steps;
    let delta = 1.0 / (num_steps - 1.0);

    //Quantise to the nearest lower step value
    let quantised_value = floor(min(value, 0.999999) / step_size) * delta;

    //Store the result
    output_data[u32(i)] = quantised_value;
}
