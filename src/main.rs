use ndarray_images::Image;
use terrain::{noise, Gpu};

#[tokio::main]
async fn main() {
    // Map settings
    let resolution = (1024, 1024);

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

    let gpu = Gpu::new(resolution.0 as u32, resolution.1 as u32).await;

    height = gpu.invert(&height).await;
    height = gpu.add(&height, 1.0).await;
    height = gpu.normalise(&height).await;

    let mut distances = gpu.simulate(&height).await;
    distances = gpu.normalise(&distances).await;
    // Save the distance map
    distances
        .save("output/distances.png")
        .expect("Failed to save distance map");

    // Save the height map
    height
        .save("output/height.png")
        .expect("Failed to save height map");
}
