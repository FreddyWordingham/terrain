use nalgebra::{Unit, Vector2};
use ndarray::Array2;
use ndarray_images::Image; // For saving images
use rand::thread_rng;
use rand::Rng;
use terrain::{noise, utils}; // Custom modules assumed to handle noise and utilities

fn generate_slope_map(height_map: &Array2<f32>) -> Array2<Vector2<f32>> {
    let (rows, cols) = height_map.dim();
    let mut slope_map = Array2::from_elem((rows, cols), Vector2::new(0.0, 0.0));

    for i in 0..rows {
        for j in 0..cols {
            let dz_dx = {
                let left = height_map[[i, utils::wrap_index(j, -1, cols)]];
                let right = height_map[[i, utils::wrap_index(j, 1, cols)]];
                (right - left) / 2.0
            };

            let dz_dy = {
                let up = height_map[[utils::wrap_index(i, -1, rows), j]];
                let down = height_map[[utils::wrap_index(i, 1, rows), j]];
                (down - up) / 2.0
            };

            slope_map[[i, j]] = Vector2::new(dz_dx, dz_dy);
        }
    }

    slope_map
}

fn generate_slope_map_with_radius(height_map: &Array2<f32>, radius: usize) -> Array2<Vector2<f32>> {
    let (rows, cols) = height_map.dim();
    let mut slope_map = Array2::from_elem((rows, cols), Vector2::new(0.0, 0.0));

    for i in 0..rows {
        for j in 0..cols {
            let mut dz_dx = 0.0;
            let mut dz_dy = 0.0;
            let mut weight_sum = 0.0;

            for di in -(radius as isize)..=(radius as isize) {
                for dj in -(radius as isize)..=(radius as isize) {
                    let dist = ((di * di + dj * dj) as f32).sqrt();
                    if dist > 0.0 && dist <= radius as f32 {
                        let weight = 1.0 / dist;

                        let neighbour_x = utils::wrap_index(j, dj, cols);
                        let neighbour_y = utils::wrap_index(i, di, rows);

                        let dz = height_map[[neighbour_y, neighbour_x]] - height_map[[i, j]];

                        dz_dx += dz * (dj as f32) * weight;
                        dz_dy += dz * (di as f32) * weight;
                        weight_sum += weight;
                    }
                }
            }

            if weight_sum > 0.0 {
                dz_dx /= weight_sum;
                dz_dy /= weight_sum;
            }

            slope_map[[i, j]] = Vector2::new(dz_dx, dz_dy);
        }
    }

    slope_map
}

fn position_to_indices(position: Vector2<f32>, resolution: (usize, usize)) -> (usize, usize) {
    let (rows, cols) = resolution;
    let x = (position.x * cols as f32).round() as usize % cols;
    let y = (position.y * rows as f32).round() as usize % rows;
    (y, x)
}

fn distance_to_border(position: Vector2<f32>, direction: Unit<Vector2<f32>>) -> f32 {
    // Calculate distances to the next integer lines along x and y directions
    let t_x = if direction.x > 0.0 {
        (position.x.ceil() - position.x) / direction.x
    } else if direction.x < 0.0 {
        (position.x.floor() - position.x) / direction.x
    } else {
        f32::INFINITY // No movement in the x direction
    };

    let t_y = if direction.y > 0.0 {
        (position.y.ceil() - position.y) / direction.y
    } else if direction.y < 0.0 {
        (position.y.floor() - position.y) / direction.y
    } else {
        f32::INFINITY // No movement in the y direction
    };

    // Return the minimum positive distance to the border
    t_x.min(t_y)
}

#[tokio::main]
async fn main() {
    // Map settings
    let resolution = (256, 256);

    // Initialize random number generator
    let mut rng = thread_rng();

    // Generate a height map using Perlin noise
    let mut height_map = noise::sample_perlin(
        resolution,
        &[
            ((1, 1), 1.0),
            ((2, 2), 1.0),
            ((3, 3), 0.5),
            ((5, 5), 0.25),
            ((7, 7), 0.125),
            ((11, 11), 0.0625),
            ((13, 13), 0.03125),
            ((17, 17), 0.015625),
        ],
        &mut rng,
    );

    // Save the height map as an image
    height_map.save("output/height_map.png").expect("Failed to save height_map");

    // Simulate the particle moving downhill
    let mut heat_map = Array2::from_elem(resolution, 0.0);
    let mut erosion_map = Array2::from_elem(resolution, 0.0);
    let mut sediment_map = Array2::from_elem(resolution, 0.0);

    // Simulation parameters
    let erosion_rate = 0.1; // Amount of sediment picked up per step
    let deposition_rate = 0.1; // Amount of sediment deposited per step
    let max_sediment_capacity = 1.0; // Maximum sediment a particle can carry

    let outer_steps = 100;
    let erosion_factor = 1.0 / outer_steps as f32;
    let sediment_factor = 1.0 / outer_steps as f32;

    for n in 0..outer_steps {
        println!("Step: {}", n);

        // Calculate the slope map
        let slope_map = generate_slope_map_with_radius(&height_map, 5);

        for _ in 0..1000 {
            // Place a particle at the center of the map
            let init_x = rng.gen_range(0.0..1.0);
            let init_y = rng.gen_range(0.0..1.0);
            let mut pos = Vector2::new(init_x, init_y);

            let mut sediment = 0.0; // Sediment carried by the particle

            for _ in 0..1000 {
                let (i, j) = position_to_indices(pos, resolution);
                heat_map[[i, j]] += 1.0;

                let slope = slope_map[[i, j]];
                let slope_magnitude = slope.norm();

                // Check if the slope magnitude is close to zero (local minima)
                if slope_magnitude < 1e-6 {
                    sediment_map[[i, j]] += sediment;
                    break; // Kill the particle
                }

                // Erosion: Pick up sediment based on slope magnitude
                let sediment_capacity = slope_magnitude * max_sediment_capacity;
                let erosion_amount = (sediment_capacity - sediment).min(erosion_rate).max(0.0);
                sediment += erosion_amount;
                erosion_map[[i, j]] += erosion_amount;

                // Deposition: Deposit sediment if carrying more than capacity
                if sediment > sediment_capacity {
                    let deposit_amount = (sediment - sediment_capacity).min(deposition_rate);
                    sediment -= deposit_amount;
                    sediment_map[[i, j]] += deposit_amount;
                }

                let direction = Unit::new_normalize(Vector2::new(-slope.x, -slope.y));
                pos += direction.into_inner() * 0.001;

                // Wrap the particle around the map
                pos.x = pos.x.rem_euclid(1.0);
                pos.y = pos.y.rem_euclid(1.0);
            }
        }

        erosion_map = fix_map(erosion_map);
        sediment_map = fix_map(sediment_map);
        heat_map = fix_map(heat_map);

        // Update the height map
        height_map = height_map - (&erosion_map * erosion_factor) + (&sediment_map * sediment_factor);
        height_map = utils::clamp(height_map, 0.0, 1.0);

        if n % 10 == 0 {
            // Save the height map
            height_map.save(&format!("output/height_map_{}.png", n)).expect("Failed to save height_map");

            // Normalize the erosion map
            erosion_map.save(&format!("output/erosion_map_{}.png", n)).expect("Failed to save erosion_map");

            // Normalize the sediment map
            sediment_map.save(&format!("output/sediment_map_{}.png", n)).expect("Failed to save sediment_map");

            // Normalize the heat map
            let normalized_heat_map = utils::normalize(heat_map.clone());
            normalized_heat_map.save(&format!("output/heat_map_{}.png", n)).expect("Failed to save heat_map");
        }

        // height_map = utils::gaussian_blur(&height_map, 0.5);

        erosion_map.fill(0.0);
        sediment_map.fill(0.0);
    }

    println!("Simulation complete!");
}

fn fix_map(mut map: Array2<f32>) -> Array2<f32> {
    map = utils::normalize(map);
    for _ in 0..3 {
        map = utils::square_root(map);
    }
    map = utils::clamp(map, 0.0, 0.4);
    map = utils::normalize(map);
    utils::gaussian_blur(&map, 1.4)
}
