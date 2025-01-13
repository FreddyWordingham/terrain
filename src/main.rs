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

fn gradient(samples: &Array2<f32>) -> Array2<Unit<Vector2<f32>>> {
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
    let mut rng = rand::thread_rng();

    let height_map = sample_perlin(
        (512, 512),
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

    let mut continent_map = invert_samples(sample_worley((512, 512), 5, &mut rng));
    continent_map = square_samples(continent_map);
    continent_map.save("continent_map.png").unwrap();

    let mut altitude_map = height_map * continent_map.clone();
    altitude_map = flatten_below(altitude_map, 0.1);
    altitude_map.save("altitude.png").unwrap();

    let altitude_map = band_samples(altitude_map, 16);
    altitude_map.save("map.png").unwrap();

    for n in 0..5 {
        continent_map = erode(100000, continent_map.clone(), &mut rng);
        continent_map
            .save(&format!("continent_map_{}.png", n))
            .unwrap();
    }
}

fn erode<R: Rng>(num_drops: usize, mut map: Array2<f32>, rng: &mut R) -> Array2<f32> {
    let _gradient = gradient(&map);

    let width = map.ncols();
    let height = map.nrows();
    for _ in 0..num_drops {
        let xi = rng.gen_range(0..width);
        let yi = rng.gen_range(0..height);

        map[(yi, xi)] -= 0.1;
        map[(yi, xi)] = map[(yi, xi)].max(0.0);
    }
    map
}
