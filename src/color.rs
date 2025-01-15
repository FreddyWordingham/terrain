use itertools::izip;
use ndarray::{Array2, Array3};

const DEEP_BLUE: [f32; 3] = [0.22, 0.40, 0.69];
const MID_BLUE: [f32; 3] = [0.25, 0.49, 0.79];
const LIGHT_BLUE: [f32; 3] = [0.40, 0.67, 0.84];
const PALE_YELLOW: [f32; 3] = [0.95, 0.87, 0.60];
const OLIVE_GREEN: [f32; 3] = [0.72, 0.72, 0.28];
const GRASS_GREEN: [f32; 3] = [0.48, 0.65, 0.28];
const MUTED_GREEN: [f32; 3] = [0.40, 0.53, 0.23];
const TEAL_GREEN: [f32; 3] = [0.26, 0.45, 0.37];
const BURNT_ORANGE: [f32; 3] = [0.71, 0.44, 0.28];
const SLATE_GREY: [f32; 3] = [0.29, 0.35, 0.39];
const DARK_GREY: [f32; 3] = [0.25, 0.29, 0.29];
pub const ISLAND_CMAP: [[f32; 3]; 11] = [
    DEEP_BLUE,
    MID_BLUE,
    LIGHT_BLUE,
    PALE_YELLOW,
    OLIVE_GREEN,
    GRASS_GREEN,
    MUTED_GREEN,
    TEAL_GREEN,
    BURNT_ORANGE,
    SLATE_GREY,
    DARK_GREY,
];

/// Colorize a heightmap using a list of colors.
pub fn colorize(samples: &Array2<f32>, colors: &[[f32; 3]]) -> Array3<f32> {
    let num_cols = colors.len();

    let mut image = Array3::zeros((samples.nrows(), samples.ncols(), 3));
    for (sample, mut pixel) in izip!(samples.iter(), image.outer_iter_mut()) {
        let index = ((sample * (num_cols - 1) as f32).floor() as usize).min(num_cols - 1);
        for (px, &color) in pixel.iter_mut().zip(colors[index].iter()) {
            *px = color;
        }
    }

    image
}
