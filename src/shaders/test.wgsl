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
// Compute the offset in the 1D array for a 2D grid
fn get_index(row: u32, col: u32, cols: u32) -> u32 {
    return row * cols + col;
}

// Wrap index to handle toroidal behaviour
fn wrap_index(index: i32, max: u32) -> u32 {
    if (index < 0) {
        return u32(index + i32(max));
    } else if (index >= i32(max)) {
        return u32(index - i32(max));
    }
    return u32(index);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    // Grid dimensions
    let rows = params.rows;
    let cols = params.cols;

    // Ensure we stay within bounds
    if (row >= rows || col >= cols) {
        return;
    }

    let current_idx = get_index(row, col, cols);

    // Retrieve current cell's height and water level
    let base_height = heightmap_current[current_idx];
    let water_height = watermap_current[current_idx];
    let total_height = base_height + water_height;

    // Initialise inflow and outflow
    var inflow = 0.0;
    var outflow = 0.0;

    // Loop through neighbours to calculate both inflow and outflow
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) {
                continue; // Skip the current cell
            }

            // Wrap neighbour indices for toroidal behaviour
            let neighbour_row = wrap_index(i32(row) + dy, rows);
            let neighbour_col = wrap_index(i32(col) + dx, cols);
            let neighbour_idx = get_index(neighbour_row, neighbour_col, cols);

            let neighbour_height = heightmap_current[neighbour_idx];
            let neighbour_water_height = watermap_current[neighbour_idx];
            let neighbour_total_height = neighbour_height + neighbour_water_height;

            // Calculate flow to and from the neighbour
            let height_difference = total_height - neighbour_total_height;

            if (height_difference > 0.0) {
                // Flow out of this cell to the neighbour
                let flow = min(height_difference * 0.5, water_height); // Control flow rate
                outflow += flow;
            } else {
                // Flow into this cell from the neighbour
                let flow = min(abs(height_difference) * 0.5, neighbour_water_height);
                inflow += flow;
            }
        }
    }

    // Calculate the new water height
    let net_change = inflow - outflow;
    let new_water_height = max(0.0, water_height + net_change);

    // Update buffers
    watermap_next[current_idx] = new_water_height;
    heightmap_next[current_idx] = base_height; // Keep base height unchanged
    sedimentmap_next[current_idx] = sedimentmap_current[current_idx]; // Copy sediment as is
}
