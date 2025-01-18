@group(0) @binding(0)
var<storage, read> heightmap: array<f32>;

@group(0) @binding(1)
var<storage, read_write> distance: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let rows = u32(@rows);
    let cols = u32(@cols);
    let i = gid.y * cols + gid.x;

    // Make sure it's a valid index
    if (gid.x >= cols || gid.y >= rows) {
        return;
    }

    // Start a point in the centre of this cell:
    var xPos = f32(gid.x) + 0.5;
    var yPos = f32(gid.y) + 0.5;
    
    // We'll track total distance travelled in 'dist'.
    var dist = 0.0;

    // For simplicity, we do e.g. 50 steps or until we can't go lower.
    // In a real simulation, you'd have a better termination condition.
    for (var step = 0u; step < 50u; step = step + 1u) {
        // Discretely sample integer coords to find neighbours.
        let baseX = i32(xPos);
        let baseY = i32(yPos);

        // Convert to index in 'heightmap'
        // We'll clamp to valid range so we don't go out of bounds:
        let clampedX = max(0, min(i32(cols)-1, baseX));
        let clampedY = max(0, min(i32(rows)-1, baseY));
        let idx = clampedY * i32(cols) + clampedX;
        let hereHeight = heightmap[idx];

        // We'll find the best (lowest) neighbour among the 8 directions
        // and compute a small step in that direction.
        var bestX = clampedX;
        var bestY = clampedY;
        var bestHeight = hereHeight;

        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                if (dx == 0 && dy == 0) {
                    continue;
                }
                let nx = max(0, min(i32(cols)-1, clampedX + dx));
                let ny = max(0, min(i32(rows)-1, clampedY + dy));
                let nIdx = ny * i32(cols) + nx;
                let nh = heightmap[nIdx];
                if (nh < bestHeight) {
                    bestHeight = nh;
                    bestX = nx;
                    bestY = ny;
                }
            }
        }

        // If we didn't find a lower neighbour, we're done rolling.
        if (bestHeight >= hereHeight) {
            break;
        }

        // Move our "point" to the centre of that neighbour cell
        let oldXPos = xPos;
        let oldYPos = yPos;
        xPos = f32(bestX) + 0.5;
        yPos = f32(bestY) + 0.5;

        // Accumulate the distance
        let dx = xPos - oldXPos;
        let dy = yPos - oldYPos;
        dist = dist + sqrt(dx*dx + dy*dy);
    }

    // Finally, store the total distance
    distance[i] = dist;
}
