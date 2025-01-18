use ndarray::Zip;
use ndarray_images::Image;
use terrain::{noise, Gpu};

#[tokio::main]
async fn main() {
    // Map settings
    let resolution = (128, 128);

    // Initialize the random number generator
    let mut rng = rand::thread_rng();

    // Generate a height map
    let mut height = noise::sample_perlin(
        resolution,
        &[
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
    // Save the height map
    height
        .save("output/height.png")
        .expect("Failed to save height map");

    let gpu = Gpu::new(resolution.0 as u32, resolution.1 as u32).await;

    height = gpu.invert(&height).await;
    height = gpu.add(&height, 1.0).await;
    height = gpu.normalise(&height).await;

    let gradient = gpu.gradient(&height).await;
    let mut gradient_x = gradient.index_axis(ndarray::Axis(2), 0).to_owned();
    let mut gradient_y = gradient.index_axis(ndarray::Axis(2), 1).to_owned();
    let mut gradient_mag = Zip::from(&gradient_x)
        .and(&gradient_y)
        .map_collect(|&x, &y| (x.powi(2) + y.powi(2)).sqrt());
    // Save the gradient magnitude map
    gradient_mag = gpu.normalise(&gradient_mag).await;
    gradient_mag
        .save("output/gradient_mag.png")
        .expect("Failed to save gradient magnitude map");

    // Save the gradient map
    gradient_x = gpu.normalise(&gradient_x).await;
    gradient_x
        .save("output/gradient_x.png")
        .expect("Failed to save gradient map");
    gradient_y = gpu.normalise(&gradient_y).await;
    gradient_y
        .save("output/gradient_y.png")
        .expect("Failed to save gradient map");
}
