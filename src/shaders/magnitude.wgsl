@group(0) @binding(0)
var<storage, read> gradient_data : array<f32>;  //2 floats per pixel: [gx, gy, gx, gy, ...]

@group(0) @binding(1)
var<storage, read_write> magnitude_out : array<f32>; //1 float per pixel

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3 < u32>)
{
    let rows = i32(@rows);
    let cols = i32(@cols);

    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= cols || y >= rows)
    {
        return;
    }

    //Flattened pixel index in 2D
    let i = y * cols + x;

    //Each gradient pixel has two floats: [gx, gy].
    let base = u32(i) * 2u;
    let gx = gradient_data[base + 0u];
    let gy = gradient_data[base + 1u];

    //Compute the magnitude
    let mag = sqrt(gx * gx + gy * gy);

    //Write it to the output array
    magnitude_out[u32(i)] = mag;
}
