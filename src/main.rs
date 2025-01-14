use nalgebra::{Unit, Vector2};
use ndarray::Array2;
use ndarray_images::Image;
use noisette::{Noise, Perlin, Stack, Worley};
use rand::Rng;

fn normalize_samples(mut samples: Array2<f32>) -> Array2<f32> {
    let min = samples.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = samples.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = max - min;
    samples.mapv_inplace(|v| (v - min) / range);
    samples
}

fn invert_samples(mut samples: Array2<f32>) -> Array2<f32> {
    samples.mapv_inplace(|v| 1.0 - v);
    samples
}

fn square_samples(mut samples: Array2<f32>) -> Array2<f32> {
    samples.mapv_inplace(|v| v * v);
    samples
}

fn band_samples(mut samples: Array2<f32>, levels: usize) -> Array2<f32> {
    let delta = 1.0 / levels as f32;
    samples.mapv_inplace(|v| (v / delta).floor() * delta);
    samples
}

fn flatten_below(mut samples: Array2<f32>, value: f32) -> Array2<f32> {
    samples.mapv_inplace(|v| if v < value { value } else { v });
    samples
}

fn smooth(map: &Array2<f32>) -> Array2<f32> {
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

fn generate_gradient(samples: &Array2<f32>) -> Array2<Unit<Vector2<f32>>> {
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

fn generate_gradient_with_radius(
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
                    let nx = (x as isize + dx).rem_euclid(width as isize) as usize;
                    let ny = (y as isize + dy).rem_euclid(height as isize) as usize;

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

fn sample_noise<N: Noise>(resolution: (usize, usize), noise: N) -> Array2<f32> {
    let mut samples = Array2::zeros(resolution);
    let width = samples.ncols();
    let height = samples.nrows();
    for ((xi, yi), value) in samples.indexed_iter_mut() {
        let x = xi as f32 / width as f32;
        let y = yi as f32 / height as f32;
        *value = noise.sample(x, y);
    }
    samples
}

fn sample_perlin<R: Rng>(
    resolution: (usize, usize),
    layers: &[((usize, usize), f32)],
    mut rng: &mut R,
) -> Array2<f32> {
    // Generate the noise function
    let mut stack = Vec::with_capacity(layers.len());
    for (shape, weight) in layers {
        let noise_box: Box<dyn Noise> = Box::new(Perlin::new(*shape, &mut rng));
        stack.push((noise_box, *weight));
    }
    let noise = Stack::new(stack);

    // Sample the noise function
    let samples = sample_noise(resolution, noise);

    // Normalize the samples
    normalize_samples(samples)
}

fn sample_worley<R: Rng>(
    resolution: (usize, usize),
    num_points: usize,
    mut rng: &mut R,
) -> Array2<f32> {
    // Generate the noise function
    let noise = Worley::new(num_points, &mut rng);

    // Sample the noise function
    let samples = sample_noise(resolution, noise);

    // Normalize the samples
    normalize_samples(samples)
}

fn main() {
    let resolution = (1024, 1024);

    let mut rng = rand::thread_rng();

    let height_map = sample_perlin(
        resolution,
        &[
            ((5, 5), 1.0),
            ((9, 9), 0.5),
            ((17, 17), 0.25),
            ((33, 33), 0.125),
            ((65, 65), 0.0625),
            ((129, 129), 0.03125),
        ],
        &mut rng,
    );
    height_map.save("height_map.png").unwrap();

    let mut continent_map = invert_samples(sample_worley(resolution, 17, &mut rng));
    continent_map = square_samples(continent_map);
    continent_map.save("continent_map.png").unwrap();

    let mut altitude_map = height_map * continent_map.clone();
    altitude_map = normalize_samples(altitude_map);
    altitude_map = flatten_below(altitude_map, 0.1);
    altitude_map.save("altitude.png").unwrap();

    // let altitude_map = band_samples(altitude_map, 16);
    // altitude_map.save("map.png").unwrap();

    let mut water = Array2::zeros(resolution);
    for n in 0..100 {
        println!("Eroding map {}", n);
        let result = erode(1000000, altitude_map.clone(), &mut rng);
        altitude_map = result.0;
        water += &result.1;
        altitude_map
            .save(&format!("continent_map_{}.png", n))
            .unwrap();
    }
    altitude_map = smooth(&altitude_map);
    altitude_map.save("final.png").unwrap();

    let water = normalize_samples(water);
    water.save("water.png").unwrap();
}

fn erode<R: Rng>(
    num_drops: usize,
    mut map: Array2<f32>,
    rng: &mut R,
) -> (Array2<f32>, Array2<f32>) {
    let water_level = 0.3;
    let gravity = 0.5;
    let friction = 0.1;
    let erosion = 0.000001;
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
