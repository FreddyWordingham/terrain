struct ErosionParams {
    width : u32,
    height : u32,
    radius : f32,
    water_level : f32,
    gravity : f32,
    friction : f32,
    erosion : f32,
    min_velocity : f32,
    max_velocity : f32,
    max_steps : u32,
}

@group(0) @binding(0)
var<storage, read> heightmap : array<f32>;
@group(0) @binding(1)
var<storage, read_write> erosionBuffer : array<f32>;
@group(0) @binding(2)
var<storage, read_write> waterBuffer : array<f32>;
@group(0) @binding(3)
var<storage, read> seeds : array<u32>;
@group(0) @binding(4)
var<uniform> params : ErosionParams;

fn wrap(v : f32, size : f32) -> f32 {
    return (v + size) % size;
}

fn get_seed(idx : u32) -> f32 {
    return f32(seeds[idx]) / 4294967296.0;
}

fn calculate_gradient_from_heightmap(pos : vec2 < f32>, cols : i32, rows : i32) -> vec2 < f32> {
    let x0 = i32(floor(pos.x));
    let x1 = (x0 + 1) % cols;
    let y0 = i32(floor(pos.y));
    let y1 = (y0 + 1) % rows;

    let fx = fract(pos.x);
    let fy = fract(pos.y);

    let h00 = heightmap[u32(y0 * cols + x0)];
    let h10 = heightmap[u32(y0 * cols + x1)];
    let h01 = heightmap[u32(y1 * cols + x0)];
    let h11 = heightmap[u32(y1 * cols + x1)];

    //Bilinear interpolation (for reference):
    let h0 = mix(h00, h10, fx);
    let h1 = mix(h01, h11, fx);
    let _h = mix(h0, h1, fy);

    //Central difference for the gradient:
    let h_left = mix(h00, h01, fy);
    let h_right = mix(h10, h11, fy);
    let h_top = mix(h00, h10, fx);
    let h_bottom = mix(h01, h11, fx);

    let dx = h_right - h_left;
    let dy = h_bottom - h_top;

    return vec2 < f32 > (-dx, -dy);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3 < u32>)
{
    let drop_id = gid.x;
    if (drop_id >= params.width * params.height)
    {
        return;
    }

    //Convert drop_id to a seed index
    let seed_x = get_seed(drop_id * 2u + 0u);
    let seed_y = get_seed(drop_id * 2u + 1u);

    //Random initial position
    var pos = vec2 < f32 > (
    (seed_x * f32(params.width)) + 0.5,
    (seed_y * f32(params.height)) + 0.5
    );
    var vel = vec2 < f32 > (0.0, 0.0);

    for (var step = 0u; step < params.max_steps; step++)
    {
        let xi = (u32(floor(pos.x)) + params.width) % params.width;
        let yi = (u32(floor(pos.y)) + params.height) % params.height;
        let index = yi * params.width + xi;

        //Check water level
        if (heightmap[index] < params.water_level)
        {
            break;
        }

        //Gradient and velocity update
        let grad = calculate_gradient_from_heightmap(pos, i32(params.width), i32(params.height));
        if (length(grad) < 0.0001)
        {
            break;
        }

        vel = vel + grad - (vel * params.friction);
        let speed = length(vel);
        if (speed > params.max_velocity)
        {
            vel = normalize(vel) * params.max_velocity;
        } else if (speed < params.min_velocity)
        {
            break;
        }

        //Move and wrap
        pos = pos + vel;
        pos.x = wrap(pos.x, f32(params.width));
        pos.y = wrap(pos.y, f32(params.height));

        //Radial erosion around the drop
        let drop_xi = (u32(floor(pos.x)) + params.width) % params.width;
        let drop_yi = (u32(floor(pos.y)) + params.height) % params.height;

        let r = i32(ceil(params.radius));
        //for (var yd = -r; yd <= r; yd++)
        //{
        //for (var xd = -r; xd <= r; xd++)
        //{
        //let distSq = f32(xd * xd + yd * yd);
        //let rSq = params.radius * params.radius;
        //if (distSq <= rSq)
        //{
        //let falloff = 1.0 - (distSq / rSq);
        //let nx = (i32(drop_xi) + xd + i32(params.width)) % i32(params.width);
        //let ny = (i32(drop_yi) + yd + i32(params.height)) % i32(params.height);
        //let nIndex = u32(ny) * params.width + u32(nx);

        //let erosionVal = speed * falloff * params.erosion;
        //erosionBuffer[nIndex] = erosionBuffer[nIndex] + erosionVal;
        //waterBuffer[nIndex] = waterBuffer[nIndex] + falloff;
        //}
        //}
        //}
    }
}
