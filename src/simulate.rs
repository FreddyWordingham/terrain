use nalgebra::Vector2;
use ndarray::Array2;
use rand::Rng;

use crate::utils::generate_gradient_with_radius;

/// Simulate hydraulic erosion on a heightmap.
pub fn erode<R: Rng>(
    num_drops: usize,
    mut map: Array2<f32>,
    rng: &mut R,
) -> (Array2<f32>, Array2<f32>) {
    let water_level = 0.2;
    let gravity = 0.5;
    let friction = 0.1;
    let erosion = 0.00001;
    let min_velocity = 0.5;
    let max_velocity = 1.0;
    let radius = 5.0; // Radius of erosion sphere

    let gradient = generate_gradient_with_radius(&map, 5);

    let width = map.ncols();
    let height = map.nrows();
    let mut erosion_buffer = Array2::<f32>::zeros((height, width));
    let mut water = Array2::<f32>::zeros((height, width));
    for _ in 0..num_drops {
        // Initialize the drop at a random location
        let mut xi = rng.gen_range(0..width);
        let mut yi = rng.gen_range(0..height);
        let mut pos = Vector2::new(xi as f32 + 0.5, yi as f32 + 0.5);
        let mut vel: Vector2<f32> = Vector2::zeros();

        // Simulate the drop
        for _ in 0..100 {
            if map[(yi, xi)] < water_level {
                break;
            }

            // Update the velocity
            let grad = -gradient[(yi, xi)];
            vel = vel + (grad.into_inner() * gravity) - (vel * friction);

            let speed = vel.norm();
            if speed > max_velocity {
                vel = vel.normalize() * max_velocity;
            } else if speed < min_velocity {
                break;
            }

            let mut new_pos = pos + vel;
            new_pos.x = new_pos.x % width as f32;
            new_pos.y = new_pos.y % height as f32;

            // Update the position
            pos = new_pos;
            xi = pos.x as usize;
            yi = pos.y as usize;

            // Radial erosion
            for dy in -(radius as isize)..=(radius as isize) {
                for dx in -(radius as isize)..=(radius as isize) {
                    let nx = (xi as isize + dx).rem_euclid(width as isize) as usize;
                    let ny = (yi as isize + dy).rem_euclid(height as isize) as usize;

                    let distance = ((dx * dx + dy * dy) as f32).sqrt();
                    if distance <= radius {
                        // Calculate erosion weight based on distance
                        let weight = (1.0 - (distance / radius)).max(0.0).powf(2.0);
                        erosion_buffer[(ny, nx)] += speed * weight * erosion; // Scale erosion by velocity
                        water[(ny, nx)] += weight;
                    }
                }
            }
        }
    }

    // Apply the erosion buffer
    map -= &erosion_buffer;
    map.mapv_inplace(|v| v.max(0.0));

    (map, water)
}
