use ndarray::Array2;
use ndarray_images::Image;
use rand::thread_rng;
// use rand::Rng;
// use rand::{rngs::StdRng, SeedableRng};
use terrain::{gpu::TestParams, noise, Gpu};

#[tokio::main]
async fn main() {
    // Map settings
    let resolution = (512, 512);

    // // Initialize the random number generator
    // let seed: [u8; 32] = [42; 32];
    // let mut rng = StdRng::from_seed(seed);
    let mut rng = thread_rng();

    // Generate a height map
    let mut height = noise::sample_perlin(
        resolution,
        &[
            ((1, 1), 1.0),
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
    height.save("output/height.png").expect("Failed to save height map");

    let gpu = Gpu::new(resolution.0 as u32, resolution.1 as u32).await;

    // Calculate gradient
    let _gradient = gpu.gradient(&height).await;

    let params = TestParams {
        rows: resolution.1 as u32,
        cols: resolution.0 as u32,
        dt: 0.1,
        erosion_rate: 0.01,
        deposition_rate: 0.01,
        sediment_capacity: 1.0,
        evaporation_rate: 0.1,
        iterations: 100,
    };

    let mut watermap = Array2::from_elem(resolution, 0.01); // Initial water map
    let mut sedimentmap = Array2::zeros(resolution); // Initial sediment map

    for n in 0..1 {
        // Run the simulation with the current state of the height, water, and sediment maps
        let (new_height, new_water, new_sediment) = gpu.test(&height, &watermap, &sedimentmap, params).await;

        // Update the maps for the next iteration
        height = new_height;
        watermap = new_water;
        sedimentmap = new_sediment;

        height.iter_mut().for_each(|x| *x = x.max(0.0));

        let min_height = height.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_height = height.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_water = watermap.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_water = watermap.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_sediment = sedimentmap.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_sediment = sedimentmap.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        println!("Height: min={}, max={}", min_height, max_height);
        println!("Water: min={}, max={}", min_water, max_water);
        println!("Sediment: min={}, max={}", min_sediment, max_sediment);

        // Save the final results
        if (max_height - min_height).abs() > 1.0e-6 {
            height = gpu.normalise(&height).await;
        }
        height.save(&format!("output/height-{}.png", n)).expect("Failed to save final height map");

        if (max_water - min_water).abs() > 1.0e-6 {
            watermap = gpu.normalise(&watermap).await;
        }
        let water_min = watermap.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let water_max = watermap.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        println!("Water: min={}, max={}", water_min, water_max);
        watermap.save(&format!("output/water-{}.png", n)).expect("Failed to save final water map");

        // if (max_sediment - min_sediment).abs() > 1.0e-6 {
        //     sedimentmap = gpu.normalise(&sedimentmap).await;
        // }
        // sedimentmap.save(&format!("output/sediment-{}.png", n)).expect("Failed to save final sediment map");
    }

    println!("Simulation complete. Results saved in the output directory.");
}
