use nalgebra::{Unit, Vector2};
use ndarray::Array2;

pub fn normalize(mut samples: Array2<f32>) -> Array2<f32> {
    let min = samples.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = samples.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = max - min;
    samples.mapv_inplace(|v| (v - min) / range);
    samples
}

pub fn invert(mut samples: Array2<f32>) -> Array2<f32> {
    samples.mapv_inplace(|v| 1.0 - v);
    samples
}

pub fn square(mut samples: Array2<f32>) -> Array2<f32> {
    samples.mapv_inplace(|v| v * v);
    samples
}

pub fn band(mut samples: Array2<f32>, levels: usize) -> Array2<f32> {
    let l = levels as f32;
    let delta = 1.0 / l;
    samples.mapv_inplace(|v| ((v.min(1.0 - f32::EPSILON)) * l).floor() * delta);
    samples
}

pub fn flatten_below(mut samples: Array2<f32>, value: f32) -> Array2<f32> {
    samples.mapv_inplace(|v| if v < value { value } else { v });
    samples
}

pub fn flatten_above(mut samples: Array2<f32>, value: f32) -> Array2<f32> {
    samples.mapv_inplace(|v| if v > value { value } else { v });
    samples
}

pub fn smooth(map: &Array2<f32>) -> Array2<f32> {
    let mut smoothed_map = map.clone();
    let width = map.ncols();
    let height = map.nrows();
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            let mut count = 0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = (x as isize + dx).rem_euclid(width as isize) as usize;
                    let ny = (y as isize + dy).rem_euclid(height as isize) as usize;
                    sum += map[(ny, nx)];
                    count += 1;
                }
            }
            smoothed_map[(y, x)] = sum / count as f32;
        }
    }
    smoothed_map
}

pub fn generate_gradient(samples: &Array2<f32>) -> Array2<Unit<Vector2<f32>>> {
    let width = samples.ncols();
    let height = samples.nrows();
    let mut gradient = Array2::from_elem((height, width), Vector2::x_axis());
    for y in 0..height {
        for x in 0..width {
            let dx = samples[(y, (x + 1) % width)] - samples[(y, (x + width - 1) % width)];
            let dy = samples[((y + 1) % height, x)] - samples[((y + height - 1) % height, x)];
            gradient[(y, x)] = Unit::new_normalize(Vector2::new(dx, dy));
        }
    }
    gradient
}

pub fn generate_gradient_with_radius(
    samples: &Array2<f32>,
    radius: usize,
) -> Array2<Unit<Vector2<f32>>> {
    let width = samples.ncols();
    let height = samples.nrows();
    let mut gradient = Array2::from_elem((height, width), Vector2::x_axis());

    for y in 0..height {
        for x in 0..width {
            let mut sum_dx = 0.0;
            let mut sum_dy = 0.0;
            let mut weight_sum = 0.0;

            for dy in -(radius as isize)..=(radius as isize) {
                for dx in -(radius as isize)..=(radius as isize) {
                    let distance = ((dx * dx + dy * dy) as f32).sqrt();
                    if distance <= radius as f32 {
                        // Weight decreases with distance (e.g., linear decay)
                        let weight = (1.0 - (distance / radius as f32)).max(0.0).powf(2.0);

                        // Calculate height differences
                        let dx_height =
                            samples[(y, (x + 1) % width)] - samples[(y, (x + width - 1) % width)];
                        let dy_height = samples[((y + 1) % height, x)]
                            - samples[((y + height - 1) % height, x)];

                        // Accumulate weighted differences
                        sum_dx += weight * dx_height;
                        sum_dy += weight * dy_height;
                        weight_sum += weight;
                    }
                }
            }

            // Normalize gradient direction and store it
            let avg_dx = if weight_sum > 0.0 {
                sum_dx / weight_sum
            } else {
                0.0
            };
            let avg_dy = if weight_sum > 0.0 {
                sum_dy / weight_sum
            } else {
                0.0
            };
            gradient[(y, x)] = Unit::new_normalize(Vector2::new(avg_dx, avg_dy));
        }
    }

    gradient
}
