@group(0) @binding(0)
var<storage, read> input_data : array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data : array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3 < u32>)
{
    let rows = i32(@rows);
    let cols = i32(@cols);

    let x = i32(gid.x);
    let y = i32(gid.y);
    let i = y * cols + x;

    //Guard against out-of-bounds
    if (x >= cols || y >= rows)
    {
        return;
    }

    //Wrap-around indices for toroidal layout
    let left = (x - 1 + cols) % cols;
    let right = (x + 1) % cols;
    let up = (y - 1 + rows) % rows;
    let down = (y + 1) % rows;

    //Fetch neighbouring values using wrapped indices
    let left_value = input_data[u32(y * cols + left)];
    let right_value = input_data[u32(y * cols + right)];
    let up_value = input_data[u32(up * cols + x)];
    let down_value = input_data[u32(down * cols + x)];

    //Compute the average of the surrounding values
    let avg_surrounding = (left_value + right_value + up_value + down_value) / 4.0;

    //Compute the output value as the absolute difference
    let value = input_data[u32(i)];
    let output_value = abs(value - avg_surrounding);

    //Store the result
    output_data[u32(i)] = output_value;
}
