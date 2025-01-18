use ndarray_images::Image;
use terrain::{noise, utils, Gpu};

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

    // height = gpu.invert(&height).await;
    // height = gpu.add(&height, 1.0).await;
    // height = gpu.normalise(&height).await;

    // let gradient = gpu.gradient(&height).await;
    // let mut gradient_mag = gpu.magnitude(&gradient).await;

    // // Save the gradient magnitude map
    // gradient_mag = gpu.normalise(&gradient_mag).await;
    // gradient_mag
    //     .save("output/gradient_mag.png")
    //     .expect("Failed to save gradient magnitude map");

    // Find the min and max
    let min = height.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = height.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    println!("Min: {}, Max: {}", min, max);
    let histogram = utils::histogram(&height, 20);
    println!("{:?}", histogram);

    // Colour
    let colour_stops: [[f32; 4]; 3] = [
        [1.0, 0.0, 0.0, 1.0], // red
        [0.0, 1.0, 0.0, 1.0], // green
        [0.0, 0.0, 1.0, 1.0], // blue
    ];
    let colour = gpu.colour(&height, &colour_stops).await;

    // Save the colour map
    colour
        .save("output/colour.png")
        .expect("Failed to save colour map");
}
