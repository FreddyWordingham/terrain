use ndarray::{s, Array2, Array3};
use ndarray_images::Image;
use terrain::{pathing, utils};

fn convert_3d_to_2d(map3d: &Array3<f32>) -> Array2<f32> {
    let height = map3d.shape()[0];
    let width = map3d.shape()[1];

    let mut map2d = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            map2d[(y, x)] = map3d[(y, x, 0)] + map3d[(y, x, 1)] + map3d[(y, x, 2)];
        }
    }

    map2d / 3.0
}

fn find_pixel(image: &Array3<f32>, color: [f32; 4]) -> Option<(usize, usize)> {
    // Ensure the array has the correct shape
    let (height, width, channels) = image.dim();
    assert_eq!(channels, 4, "Image must have 4 channels (RGBA).");

    // Iterate over each pixel
    for h in 0..height {
        for w in 0..width {
            // Extract RGBA values for the current pixel and convert to a Vec
            let pixel = image.slice(s![h, w, ..]).to_vec();
            if pixel == color {
                return Some((h, w)); // Return the position if it matches the green pixel
            }
        }
    }
    None // Return None if no green pixel is found
}

fn main() {
    // Load the travel map
    let mut map: Array3<f32> =
        Array3::load("input/travel_map.png").expect("Failed to load travel map");

    // Locate the start and the target
    let start = find_pixel(&map, [0.0, 0.0, 1.0, 1.0]).expect("Failed to find start (green) pixel");
    let target =
        find_pixel(&map, [0.0, 1.0, 0.0, 1.0]).expect("Failed to find target (blue) pixel");

    // Construct the travel map
    let travel_map = utils::invert(convert_3d_to_2d(&map));
    let travel_map = travel_map.mapv(|x| x as i32) * 100;

    // Calculate the path
    let path = pathing::find_path(&travel_map, start, target).unwrap().0;

    // Draw path on original map
    for (y, x) in path {
        map[(y, x, 0)] = 1.0;
        map[(y, x, 1)] = 0.0;
        map[(y, x, 2)] = 0.0;
    }

    // Draw start and target on original map
    map[(start.0, start.1, 0)] = 0.0;
    map[(start.0, start.1, 1)] = 0.0;
    map[(start.0, start.1, 2)] = 1.0;
    map[(target.0, target.1, 0)] = 0.0;
    map[(target.0, target.1, 1)] = 1.0;
    map[(target.0, target.1, 2)] = 0.0;

    // Save the map with the path
    map.save("output/path.png")
        .expect("Failed to save path map");
}
