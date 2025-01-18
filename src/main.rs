use ndarray_images::Image;
use terrain::{noise, Gpu};

#[tokio::main]
async fn main() {
    // Map settings
    let resolution = (2048, 2048);

    // Initialize the random number generator
    let mut rng = rand::thread_rng();

    // Generate a height map
    let height = noise::sample_perlin(
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

    let gradient = gpu.gradient(&height).await;

    let flow = gpu.flow(&height, &gradient).await;
    let mut magnitude = gpu.magnitude(&flow).await;
    magnitude = gpu.normalise(&magnitude).await;
    magnitude
        .save("output/flow.png")
        .expect("Failed to save flow map");
}
