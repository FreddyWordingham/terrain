@group(0) @binding(0)
var<storage, read> data : array<f32>;

@group(0) @binding(1)
var<storage, read_write> output : array<f32>;

//Helper to wrap coordinates for toroidal access
fn wrap_index(coord : i32, max : i32) -> i32 {
    return ((coord % max) + max) % max;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3 < u32>)
{
    let rows = i32(@rows);
    let cols = i32(@cols);

    let x = i32(gid.x);
    let y = i32(gid.y);

    //Out-of-bounds guard
    if (x >= cols || y >= rows)
    {
        return;
    }

    //Sum the 3×3 region about (x, y), wrapping edges
    var sum = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1)
    {
        for (var dx = -1; dx <= 1; dx = dx + 1)
        {
            let nx = wrap_index(x + dx, cols);
            let ny = wrap_index(y + dy, rows);
            let idx = ny * cols + nx;
            sum = sum + data[u32(idx)];
        }
    }

    //Write average to output
    let i = y * cols + x;
    output[u32(i)] = sum / 9.0;
}
