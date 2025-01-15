# terrain

Terrain generation experiments

```rust
use nalgebra::{Unit, Vector2};
use ndarray::{Array2, Array3};
use ndarray_images::Image;
use noisette::{Noise, Perlin, Simplex, Stack, Worley};
use rand::Rng;

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

    let mut continent_map = invert_samples(sample_worley(resolution, 1, &mut rng));
    continent_map = square_samples(continent_map);
    continent_map.save("continent_map.png").unwrap();

    let mut altitude_map = height_map * continent_map.clone();
    altitude_map = normalize_samples(altitude_map);
    altitude_map = flatten_below(altitude_map, 0.1);
    altitude_map.save("altitude.png").unwrap();

    // let altitude_map = band_samples(altitude_map, 16);
    // altitude_map.save("map.png").unwrap();

    let mut water = Array2::zeros(resolution);
    for n in 0..2 {
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

    // Colour by altitude
    let deep_blue = [0.22, 0.40, 0.69];
    let mid_blue = [0.25, 0.49, 0.79];
    let light_blue = [0.40, 0.67, 0.84];
    let pale_yellow = [0.95, 0.87, 0.60];
    let olive_green = [0.72, 0.72, 0.28];
    let grass_green = [0.48, 0.65, 0.28];
    let muted_green = [0.40, 0.53, 0.23];
    let teal_green = [0.26, 0.45, 0.37];
    let burnt_orange = [0.71, 0.44, 0.28];
    let slate_grey = [0.29, 0.35, 0.39];
    let dark_grey = [0.25, 0.29, 0.29];

    let mut image = Array3::zeros((resolution.1, resolution.0, 3));
    for yi in 0..resolution.1 {
        for xi in 0..resolution.0 {
            let altitude = altitude_map[(yi, xi)];
            let water = water[(yi, xi)];

            let mut colour = [1.0, 0.0, 1.0];
            if altitude < 0.1 {
                colour = deep_blue;
            } else if altitude < 0.2 {
                colour = mid_blue;
            } else if altitude < 0.3 {
                colour = light_blue;
            } else if altitude < 0.4 {
                colour = pale_yellow;
            } else if altitude < 0.5 {
                colour = olive_green;
            } else if altitude < 0.6 {
                colour = grass_green;
            } else if altitude < 0.7 {
                colour = muted_green;
            } else if altitude < 0.8 {
                colour = teal_green;
            } else if altitude < 0.9 {
                colour = slate_grey;
            } else {
                colour = burnt_orange;
            }

            if water > 0.04 {
                colour = light_blue;
            }

            image[(yi, xi, 0)] = colour[0];
            image[(yi, xi, 1)] = colour[1];
            image[(yi, xi, 2)] = colour[2];
        }
    }

    // Create simplex noise
    let mut noise_map = sample_simplex(
        resolution,
        &[(4.0, 1.0), (8.0, 0.5), (19.0, 0.25), (43.0, 0.125)],
        &mut rng,
    );
    noise_map.save("noise.png").unwrap();
    noise_map = band_samples(noise_map, 16);
    noise_map.save("noise_banded.png").unwrap();

    // Apply noise to the image
    for yi in 0..resolution.1 {
        for xi in 0..resolution.0 {
            let noise = noise_map[(yi, xi)];
            image[(yi, xi, 0)] *= (noise + 1.0) / 2.0;
            image[(yi, xi, 1)] *= (noise + 1.0) / 2.0;
            image[(yi, xi, 2)] *= (noise + 1.0) / 2.0;
        }
    }

    // Save the image
    image.save("image.png").unwrap();
}
```
