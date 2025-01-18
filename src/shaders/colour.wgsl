@group(0) @binding(0)
var<storage, read> heightmap : array<f32>;

@group(0) @binding(1)
var<storage, read> colourmap : array<vec4 < f32>>;

@group(0) @binding(2)
var<storage, read_write> rgb_out : array<f32>;

//Returns arrayLength(&colourmap)
fn get_colour_count() -> u32 {
    return arrayLength(&colourmap);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3 < u32>)
{
    let rows = i32(@rows);
    let cols = i32(@cols);
    let colour_count = f32(get_colour_count());

    let x = i32(gid.x);
    let y = i32(gid.y);
    let i = y * cols + x;

    //Guard out-of-bounds
    if (x >= cols || y >= rows)
    {
        return;
    }

    //1) Read the height
    var h = heightmap[u32(i)];

    //2) Scale h into [0, colour_count - 1]
    let scaled = h * (colour_count - 1.0);
    let i0 = u32(floor(scaled));
    let i1 = min(u32(i0 + 1u), u32(colour_count - 1.0));

    let alpha = fract(scaled);

    //3) Fetch the two stop colours
    let c0 = colourmap[i0];
    let c1 = colourmap[i1];

    //4) Interpolate
    let c = mix(c0, c1, alpha);

    //Store result in rgb_out
    let out_index = u32(i) * 4u;
    rgb_out[out_index + 0u] = c.r;
    rgb_out[out_index + 1u] = c.g;
    rgb_out[out_index + 2u] = c.b;
    rgb_out[out_index + 3u] = c.a;
}
