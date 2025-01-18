@group(0) @binding(0)
var<storage, read> heightmap : array<f32>;

@group(0) @binding(1)
var<storage, read> gradient_map : array<vec2 < f32>>;

@group(0) @binding(2)
var<storage, read_write> gradient_out : array<f32>;

fn wrap_index(coord : i32, max : i32) -> i32 {
    return ((coord % max) + max) % max;
}

fn calculate_gradient(x : i32, y : i32) -> vec2 < f32> {
    let x_left = wrap_index(x - 1, @cols);
    let x_right = wrap_index(x + 1, @cols);
    let y_up = wrap_index(y - 1, @rows);
    let y_down = wrap_index(y + 1, @rows);

    let left_idx = y * @cols + x_left;
    let right_idx = y * @cols + x_right;
    let up_idx = y_up * @cols + x;
    let down_idx = y_down * @cols + x;

    let grad_x = (heightmap[right_idx] - heightmap[left_idx]) * 0.5;
    let grad_y = (heightmap[down_idx] - heightmap[up_idx]) * 0.5;

    return vec2 < f32 > (grad_x, grad_y);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3 < u32>)
{
    let rows = i32(@rows);
    let cols = i32(@cols);
    let x = i32(gid.x);
    let y = i32(gid.y);

    //Guard against out-of-bounds
    if (x >= cols || y >= rows)
    {
        return;
    }

    //Flattened index of the current pixel
    let i = y * cols + x;

    //Compute gradient
    let grad = calculate_gradient(x, y);

    //Write both components to the output array.
    //Each pixel i uses two consecutive floats: [grad_x, grad_y].
    let out_index = u32(i) * 2u;
    gradient_out[out_index + 0u] = grad.x;
    gradient_out[out_index + 1u] = grad.y;
}
