use ndarray::Array2;
use pathfinding::prelude::astar;

/// Wraps an index to simulate toroidal movement.
fn wrap(idx: isize, max: usize) -> usize {
    ((idx % max as isize) + max as isize) as usize % max
}

/// Return neighboring positions plus their cost (from the given map).
fn neighbors(pos: (usize, usize), map: &Array2<i32>) -> Vec<((usize, usize), i32)> {
    let (row, col) = pos;
    let rows = map.nrows();
    let cols = map.ncols();

    // Define offsets for 4-directional movement (can add diagonals if needed).
    let moves = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    moves
        .iter()
        .map(|(dr, dc)| {
            let nr = wrap(row as isize + dr, rows);
            let nc = wrap(col as isize + dc, cols);
            ((nr, nc), map[[nr, nc]])
        })
        .collect()
}

/// A basic toroidal distance heuristic: measure “wrapped” row and column differences.
fn toroidal_distance(
    (r1, c1): (usize, usize),
    (r2, c2): (usize, usize),
    rows: usize,
    cols: usize,
) -> i32 {
    let dr = (r1 as isize - r2 as isize).abs() as i32;
    let dc = (c1 as isize - c2 as isize).abs() as i32;
    // Minimal distance considering wrap-around
    let min_dr = dr.min(rows as i32 - dr);
    let min_dc = dc.min(cols as i32 - dc);
    min_dr + min_dc
}

pub fn find_path(
    map: &Array2<i32>,
    start: (usize, usize),
    goal: (usize, usize),
) -> Option<(Vec<(usize, usize)>, i32)> {
    let (rows, cols) = (map.nrows(), map.ncols());

    astar(
        &start,
        |&p| neighbors(p, map),
        |&p| toroidal_distance(p, goal, rows, cols),
        |&p| p == goal,
    )
}
