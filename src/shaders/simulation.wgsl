@group(0) @binding(0)
var<storage, read> heightmap : array<f32>;

@group(0) @binding(1)
var<storage, read_write> distance : array<f32>;

//Helper function to wrap indices for toroidal access
fn wrap_index(coord : i32, max : i32) -> i32 {
    return ((coord % max) + max) % max;
}

fn calculate_gradient(
x : i32, y : i32,
) -> vec2 < f32> {
    let x_left = wrap_index(x - 1, @cols);
    let x_right = wrap_index(x + 1, @cols);
    let y_up = wrap_index(y - 1, @rows);
    let y_down = wrap_index(y + 1, @rows);

    let left = y * @cols + x_left;
    let right = y * @cols + x_right;
    let up = y_up * @cols + x;
    let down = y_down * @cols + x;

    let grad_x = (heightmap[right] - heightmap[left]) * 0.5;
    let grad_y = (heightmap[down] - heightmap[up]) * 0.5;

    return vec2 < f32 > (grad_x, grad_y);
}


@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3 < u32>)
{
    let rows = i32(@rows);
    let cols = i32(@cols);
    let x = i32(gid.x);
    let y = i32(gid.y);
    let i = y * cols + x;

    //Ensure the index is within bounds
    if (x >= cols || y >= rows)
    {
        return;
    }

    //Fetch the height at the current position
    var height = heightmap[i];

    //Calculate gradient length
    let gradient = calculate_gradient(x, y);
    let len = length(gradient);

    //Store the total distance
    distance[i] = len;
}
