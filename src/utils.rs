use nalgebra::{Unit, Vector2};
use ndarray::parallel::prelude::*;
use ndarray::{Array2, Axis};
use rayon::prelude::*;
use rustfft::{num_complex::Complex, num_traits::Zero, FftPlanner};

// Helper for torus wrapping.
pub fn wrap_index(index: usize, delta: isize, max: usize) -> usize {
    let new_index = (index as isize + delta).rem_euclid(max as isize);
    new_index as usize
}

pub fn normalize(mut samples: Array2<f32>) -> Array2<f32> {
    // Compute min and max in parallel.
    let min = samples.par_iter().cloned().reduce(|| f32::INFINITY, f32::min);
    let max = samples.par_iter().cloned().reduce(|| f32::NEG_INFINITY, f32::max);
    let range = max - min;

    // Update in parallel.
    samples.as_slice_mut().expect("Array is not contiguous").par_iter_mut().for_each(|v| *v = (*v - min) / range);
    samples
}

pub fn invert(mut samples: Array2<f32>) -> Array2<f32> {
    samples.as_slice_mut().expect("Array is not contiguous").par_iter_mut().for_each(|v| *v = 1.0 - *v);
    samples
}

pub fn double(mut samples: Array2<f32>) -> Array2<f32> {
    samples.as_slice_mut().expect("Array is not contiguous").par_iter_mut().for_each(|v| *v *= 2.0);
    samples
}

pub fn clamp(mut samples: Array2<f32>, lower: f32, upper: f32) -> Array2<f32> {
    samples.as_slice_mut().expect("Array is not contiguous").par_iter_mut().for_each(|v| *v = v.clamp(lower, upper));
    samples
}

pub fn square(mut samples: Array2<f32>) -> Array2<f32> {
    samples.as_slice_mut().expect("Array is not contiguous").par_iter_mut().for_each(|v| *v = *v * *v);
    samples
}

pub fn square_root(mut samples: Array2<f32>) -> Array2<f32> {
    samples.as_slice_mut().expect("Array is not contiguous").par_iter_mut().for_each(|v| *v = v.sqrt());
    samples
}

pub fn band(mut samples: Array2<f32>, levels: usize) -> Array2<f32> {
    let l = levels as f32;
    let delta = 1.0 / l;
    samples
        .as_slice_mut()
        .expect("Array is not contiguous")
        .par_iter_mut()
        .for_each(|v| *v = ((v.min(1.0 - f32::EPSILON)) * l).floor() * delta);
    samples
}

pub fn flatten_below(mut samples: Array2<f32>, value: f32) -> Array2<f32> {
    samples.as_slice_mut().expect("Array is not contiguous").par_iter_mut().for_each(|v| {
        if *v < value {
            *v = value;
        }
    });
    samples
}

pub fn flatten_above(mut samples: Array2<f32>, value: f32) -> Array2<f32> {
    samples.as_slice_mut().expect("Array is not contiguous").par_iter_mut().for_each(|v| {
        if *v > value {
            *v = value;
        }
    });
    samples
}

pub fn smooth(map: &Array2<f32>) -> Array2<f32> {
    let (height, width) = (map.nrows(), map.ncols());
    let mut smoothed_map = map.clone();
    // Parallelise over rows.
    smoothed_map.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(y, mut row)| {
        for x in 0..width {
            let mut sum = 0.0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let ny = wrap_index(y, dy, height);
                    let nx = wrap_index(x, dx, width);
                    sum += map[(ny, nx)];
                }
            }
            row[x] = sum / 9.0;
        }
    });
    smoothed_map
}

/// Applies Gaussian blur with torus wrapping using parallel row processing.
pub fn gaussian_blur(input: &Array2<f32>, sigma: f32) -> Array2<f32> {
    let kernel_size = ((6.0 * sigma).ceil() as usize) | 1;
    let half_size = kernel_size / 2;
    let (height, width) = (input.nrows(), input.ncols());

    // Create and normalise Gaussian kernel.
    let mut kernel = vec![0.0; kernel_size];
    let mut sum = 0.0;
    for i in 0..kernel_size {
        let x = (i as isize - half_size as isize) as f32;
        kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
        sum += kernel[i];
    }
    kernel.iter_mut().for_each(|v| *v /= sum);

    // Temporary buffers.
    let mut temp = Array2::zeros((height, width));
    let mut output = Array2::zeros((height, width));

    // Horizontal blur – parallelise over rows.
    temp.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(y, mut row)| {
        for x in 0..width {
            let mut value = 0.0;
            for k in 0..kernel_size {
                let dx = k as isize - half_size as isize;
                let nx = wrap_index(x, dx, width);
                value += input[(y, nx)] * kernel[k];
            }
            row[x] = value;
        }
    });

    // Vertical blur – parallelise over columns via rows.
    output.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(y, mut row)| {
        for x in 0..width {
            let mut value = 0.0;
            for k in 0..kernel_size {
                let dy = k as isize - half_size as isize;
                let ny = wrap_index(y, dy, height);
                value += temp[(ny, x)] * kernel[k];
            }
            row[x] = value;
        }
    });

    output
}

/// Applies a median filter with torus wrapping.
pub fn median_filter(input: &Array2<f32>, window_size: usize) -> Array2<f32> {
    let (height, width) = (input.nrows(), input.ncols());
    let half_window = window_size as isize / 2;
    let mut output = input.clone();

    output.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(y, mut row)| {
        for x in 0..width {
            let mut neighbourhood = Vec::with_capacity(window_size * window_size);
            for wy in -half_window..=half_window {
                for wx in -half_window..=half_window {
                    let ny = wrap_index(y, wy, height);
                    let nx = wrap_index(x, wx, width);
                    neighbourhood.push(input[(ny, nx)]);
                }
            }
            neighbourhood.sort_by(|a, b| a.partial_cmp(b).unwrap());
            row[x] = neighbourhood[neighbourhood.len() / 2];
        }
    });
    output
}

/// Applies a bilateral filter with torus wrapping.
pub fn bilateral_filter(input: &Array2<f32>, sigma_spatial: f32, sigma_intensity: f32) -> Array2<f32> {
    let (height, width) = (input.nrows(), input.ncols());
    let kernel_size = ((6.0 * sigma_spatial).ceil() as usize) | 1;
    let half_size = kernel_size as isize / 2;
    let mut output = Array2::zeros((height, width));

    output.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(y, mut row)| {
        for x in 0..width {
            let mut weight_sum = 0.0;
            let mut value_sum = 0.0;
            for wy in -half_size..=half_size {
                for wx in -half_size..=half_size {
                    let ny = wrap_index(y, wy, height);
                    let nx = wrap_index(x, wx, width);
                    let spatial_dist = ((wy * wy + wx * wx) as f32).sqrt();
                    let intensity_dist = (input[(ny, nx)] - input[(y, x)]).abs();

                    let spatial_weight = (-spatial_dist.powi(2) / (2.0 * sigma_spatial.powi(2))).exp();
                    let intensity_weight = (-intensity_dist.powi(2) / (2.0 * sigma_intensity.powi(2))).exp();
                    let weight = spatial_weight * intensity_weight;

                    weight_sum += weight;
                    value_sum += input[(ny, nx)] * weight;
                }
            }
            row[x] = value_sum / weight_sum;
        }
    });
    output
}

pub fn frequency_filter(input: &Array2<f32>, threshold: f32) -> Array2<f32> {
    let (height, width) = (input.nrows(), input.ncols());
    let size = height * width;

    let mut complex_input: Vec<Complex<f32>> = input.iter().map(|&val| Complex::new(val, 0.0)).collect();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(size);
    fft.process(&mut complex_input);

    let mut filtered = complex_input.clone();
    for y in 0..height {
        for x in 0..width {
            let freq_dist = ((x as f32 / width as f32).powi(2) + (y as f32 / height as f32).powi(2)).sqrt();
            if freq_dist > threshold {
                filtered[y * width + x] = Complex::zero();
            }
        }
    }

    let ifft = planner.plan_fft_inverse(size);
    ifft.process(&mut filtered);

    let output_real: Vec<f32> = filtered.iter().map(|c| c.re / size as f32).collect();
    Array2::from_shape_vec((height, width), output_real).unwrap()
}

pub fn generate_gradient(samples: &Array2<f32>) -> Array2<Unit<Vector2<f32>>> {
    let (height, width) = (samples.nrows(), samples.ncols());
    let mut gradient = Array2::from_elem((height, width), Vector2::x_axis());

    gradient.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(y, mut row)| {
        for x in 0..width {
            let dx = samples[(y, wrap_index(x, 1, width))] - samples[(y, wrap_index(x, -1, width))];
            let dy = samples[(wrap_index(y, 1, height), x)] - samples[(wrap_index(y, -1, height), x)];
            row[x] = Unit::new_normalize(Vector2::new(dx, dy));
        }
    });
    gradient
}

pub fn generate_gradient_with_radius(samples: &Array2<f32>, radius: usize) -> Array2<Unit<Vector2<f32>>> {
    let (height, width) = (samples.nrows(), samples.ncols());
    let mut gradient = Array2::from_elem((height, width), Vector2::x_axis());

    gradient.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(y, mut row)| {
        for x in 0..width {
            let mut sum_dx = 0.0;
            let mut sum_dy = 0.0;
            let mut weight_sum = 0.0;
            for dy in -(radius as isize)..=(radius as isize) {
                for dx in -(radius as isize)..=(radius as isize) {
                    let distance = ((dx * dx + dy * dy) as f32).sqrt();
                    if distance <= radius as f32 {
                        let weight = (1.0 - (distance / radius as f32)).max(0.0).powf(2.0);
                        let dx_height = samples[(y, wrap_index(x, 1, width))] - samples[(y, wrap_index(x, -1, width))];
                        let dy_height = samples[(wrap_index(y, 1, height), x)] - samples[(wrap_index(y, -1, height), x)];
                        sum_dx += weight * dx_height;
                        sum_dy += weight * dy_height;
                        weight_sum += weight;
                    }
                }
            }
            let avg_dx = if weight_sum > 0.0 { sum_dx / weight_sum } else { 0.0 };
            let avg_dy = if weight_sum > 0.0 { sum_dy / weight_sum } else { 0.0 };
            row[x] = Unit::new_normalize(Vector2::new(avg_dx, avg_dy));
        }
    });
    gradient
}

pub fn histogram(samples: &Array2<f32>, bins: usize) -> Vec<usize> {
    let min = samples.par_iter().cloned().reduce(|| f32::INFINITY, f32::min);
    let max = samples.par_iter().cloned().reduce(|| f32::NEG_INFINITY, f32::max);
    let range = max - min;
    let bin_width = range / bins as f32;

    // Use a parallel fold to count bins.
    samples
        .par_iter()
        .fold(
            || vec![0usize; bins],
            |mut hist, &sample| {
                let bin = ((sample - min) / bin_width).min(bins as f32 - 1.0) as usize;
                hist[bin] += 1;
                hist
            },
        )
        .reduce(
            || vec![0usize; bins],
            |mut a, b| {
                a.iter_mut().zip(b.into_iter()).for_each(|(x, y)| *x += y);
                a
            },
        )
}
