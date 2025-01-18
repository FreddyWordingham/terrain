@group(0) @binding(0)
var<storage, read_write> data: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.y * u32(@cols) + gid.x;
    if (gid.x < u32(@cols) && gid.y < u32(@rows)) {
        data[index] = -data[index];
    }
}