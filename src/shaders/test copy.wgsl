// Bind groups for data storage
@group(0) @binding(0)
var<storage, read_write> heightmap_current: array<f32>;
@group(0) @binding(1)
var<storage, read_write> heightmap_next: array<f32>;

@group(0) @binding(2)
var<storage, read_write> watermap_current: array<f32>;
@group(0) @binding(3)
var<storage, read_write> watermap_next: array<f32>;

@group(0) @binding(4)
var<storage, read_write> sedimentmap_current: array<f32>;
@group(0) @binding(5)
var<storage, read_write> sedimentmap_next: array<f32>;

@group(0) @binding(6)
var<uniform> params: Params;

// Structure for simulation parameters
struct Params {
    rows: u32, // Number of rows in the grid
    cols: u32, // Number of columns in the grid
    dt: f32, // Time step
    erosion_rate: f32, // Rate of erosion
    deposition_rate: f32, // Rate of deposition
    sediment_capacity: f32, // Maximum sediment capacity
    evaporation_rate: f32, // Rate of evaporation
    iterations: u32, // Number of iterations to perform
}


// Utility function: Calculate neighbour indices with toroidal wrapping
fn get_neighbours(x: i32, y: i32, cols: i32, rows: i32) -> array<u32, 4> {
    return array<u32, 4>(u32(((y - 1 + rows) % rows) * cols + x), // Top
    u32(((y + 1) % rows) * cols + x), // Bottom
    u32(y * cols + ((x - 1 + cols) % cols)), // Left
    u32(y * cols + ((x + 1) % cols)) // Right
    );
}

// Utility function: Compute gradient (approximate slope)
fn gradient(h: f32, neighbours: array<f32, 4>) -> f32 {
    var max_diff = 0.0;
    for (var i = 0; i < 4; i = i + 1) {
        max_diff = max(max_diff, abs(h - neighbours[i]));
    }
    return max_diff;
}

// Function: Perform erosion, sediment transport, and evaporation for one iteration
fn simulate_step(x: i32, y: i32, i: u32, cols: i32, rows: i32) {
    // Read current state
    let h = heightmap_current[i];
    let w = watermap_current[i];
    let s = sedimentmap_current[i];

    // Get neighbours
    let neighbours = get_neighbours(x, y, cols, rows);
    var neighbour_heights = array<f32, 4>();
    for (var j = 0; j < 4; j = j + 1) {
        neighbour_heights[j] = heightmap_current[neighbours[j]];
    }

    // Compute gradient and sediment capacity
    let grad = gradient(h, neighbour_heights);
    let sediment_capacity = params.sediment_capacity * w * grad;

    // Erosion or deposition
    var new_h = h;
    var new_s = s;
    if (s < sediment_capacity) {
        let erosion = params.erosion_rate * (sediment_capacity - s);
        new_h -= erosion;
        new_s += erosion;
    } else {
        let deposition = params.deposition_rate * (s - sediment_capacity);
        new_h += deposition;
        new_s -= deposition;
    }

    // Evaporation
    let new_w = w * (1.0 - params.evaporation_rate);

    // Write updated state to the "next" buffers
    heightmap_next[i] = new_h;
    watermap_next[i] = new_w;
    sedimentmap_next[i] = new_s;
}

// Main compute function: Perform multiple iterations
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cols = i32(params.cols);
    let rows = i32(params.rows);

    let x = i32(gid.x);
    let y = i32(gid.y);
    let i = u32(y * cols + x);

    // Guard against out-of-bounds
    if (x >= cols || y >= rows) {
        return;
    }

    // Perform multiple iterations
    for (var step = 0u; step < params.iterations; step = step + 1u) {
        simulate_step(x, y, i, cols, rows);

        // Swap buffers: Make "next" the "current" for the next iteration
        if (step < params.iterations - 1u) {
            heightmap_current[i] = heightmap_next[i];
            watermap_current[i] = watermap_next[i];
            sedimentmap_current[i] = sedimentmap_next[i];
        }
    }
}
