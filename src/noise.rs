use ndarray::Array2;
use noisette::{Noise, Perlin, Simplex, Stack, Worley};
use rand::Rng;

use crate::utils::normalize;

/// Uniformly sample a noise function in a 2D grid.
pub fn sample_noise<N: Noise>(resolution: (usize, usize), noise: N) -> Array2<f32> {
    let (height, width) = resolution;
    let mut samples = Array2::zeros((height, width));
    let inv_width = 1.0 / width as f32;
    let inv_height = 1.0 / height as f32;

    for ((yi, xi), value) in samples.indexed_iter_mut() {
        let x = xi as f32 * inv_width;
        let y = yi as f32 * inv_height;
        *value = noise.sample(x, y);
    }

    samples
}

/// Uniformly sample a weighted stack of Perlin noise functions.
pub fn sample_perlin<R: Rng>(
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
    normalize(samples)
}

/// Uniformly sample a weighted stack of Simplex noise functions.
pub fn sample_simplex<R: Rng>(
    resolution: (usize, usize),
    layers: &[(f32, f32)],

    mut rng: &mut R,
) -> Array2<f32> {
    // Generate the noise function
    let mut stack = Vec::with_capacity(layers.len());
    for (scale, weight) in layers {
        let noise_box: Box<dyn Noise> = Box::new(Simplex::new(*scale, &mut rng));
        stack.push((noise_box, *weight));
    }
    let noise = Stack::new(stack);

    // Sample the noise function
    let samples = sample_noise(resolution, noise);

    // Normalize the samples
    normalize(samples)
}

/// Uniformly sample a Worley noise function.
pub fn sample_worley<R: Rng>(
    resolution: (usize, usize),
    num_points: usize,
    mut rng: &mut R,
) -> Array2<f32> {
    // Generate the noise function
    let noise = Worley::new(num_points, &mut rng);

    // Sample the noise function
    let samples = sample_noise(resolution, noise);

    // Normalize the samples
    normalize(samples)
}
