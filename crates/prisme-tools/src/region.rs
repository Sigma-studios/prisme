//! Colour-similarity regions: global colour range selection and contiguous flood
//! ("blob"/magic wand, fill). Distances are measured in OKLab plus alpha.

use crate::flatten::flatten_tile;
use prisme_color::working_to_oklab;
use prisme_core::mask::new_mask_grid;
use prisme_core::tile::{TileCoord, TileGrid, TILE_SIZE};
use prisme_core::{Document, IRect};
use std::collections::VecDeque;

const TS: i32 = TILE_SIZE as i32;

pub enum Sample<'a> {
    Layer(&'a TileGrid),
    Merged(&'a Document),
}

impl Sample<'_> {
    fn tile(&self, c: TileCoord) -> Option<Vec<f32>> {
        match self {
            Sample::Layer(g) => g.get_tile(c).map(|t| t.f32_view().into_owned()),
            Sample::Merged(d) => Some(flatten_tile(d, c)),
        }
    }

    pub fn pixel(&self, x: i32, y: i32) -> [f32; 4] {
        match self {
            Sample::Layer(g) => g.get_pixel(x, y),
            Sample::Merged(d) => crate::flatten::flatten_pixel(d, x, y),
        }
    }
}

fn distance(a_lab: [f32; 3], a_alpha: f32, p: &[f32]) -> f32 {
    if p[3] <= 0.0 && a_alpha <= 0.0 {
        return 0.0;
    }
    let lab = working_to_oklab([p[0], p[1], p[2]]);
    let d = ((lab[0] - a_lab[0]).powi(2) + (lab[1] - a_lab[1]).powi(2) + (lab[2] - a_lab[2]).powi(2)).sqrt();
    // Blend colour distance with alpha difference.
    let da = (p[3] - a_alpha).abs();
    d * p[3].min(a_alpha).max(0.0) + da.max(d * (1.0 - p[3].min(a_alpha)))
}

/// Similarity in [0,1] (1 = same colour) for every pixel of `bounds`, as u8.
fn similarity(sample: &Sample, seed: [f32; 4], tolerance: f32, bounds: IRect, antialias: bool) -> Vec<u8> {
    let w = bounds.width() as usize;
    let h = bounds.height() as usize;
    let mut out = vec![0u8; w * h];
    let lab = working_to_oklab([seed[0], seed[1], seed[2]]);
    let thr = (tolerance.clamp(0.0, 1.0) * 0.6).max(1e-4);
    let coords: Vec<TileCoord> = TileGrid::coords_in(&bounds).collect();
    let compute = |c: &TileCoord| -> (TileCoord, Vec<u8>) {
        let mut v = vec![0u8; (TS * TS) as usize];
        let data = sample.tile(*c);
        for i in 0..(TS * TS) as usize {
            let px: [f32; 4] = match &data {
                Some(d) => [d[i * 4], d[i * 4 + 1], d[i * 4 + 2], d[i * 4 + 3]],
                None => [0.0; 4],
            };
            let d = distance(lab, seed[3], &px);
            let s = if antialias {
                let t = ((d - thr * 0.7) / (thr * 0.3)).clamp(0.0, 1.0);
                1.0 - t * t * (3.0 - 2.0 * t)
            } else if d <= thr {
                1.0
            } else {
                0.0
            };
            v[i] = (s * 255.0).round() as u8;
        }
        (*c, v)
    };
    #[cfg(not(target_arch = "wasm32"))]
    let tiles: Vec<(TileCoord, Vec<u8>)> = {
        use rayon::prelude::*;
        coords.par_iter().map(compute).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let tiles: Vec<(TileCoord, Vec<u8>)> = coords.iter().map(compute).collect();
    for (c, v) in tiles {
        let r = c.rect().intersect(&bounds);
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                out[(y - bounds.y0) as usize * w + (x - bounds.x0) as usize] = v[((y - c.y * TS) * TS + (x - c.x * TS)) as usize];
            }
        }
    }
    out
}

fn to_grid(values: &[f32], bounds: IRect) -> TileGrid {
    let mut g = new_mask_grid();
    g.write_rect(&bounds, values);
    g
}

/// Select every pixel similar to the colour at `seed` (non-contiguous).
pub fn color_range(sample: &Sample, seed: [i32; 2], tolerance: f32, bounds: IRect, antialias: bool) -> TileGrid {
    let c = sample.pixel(seed[0], seed[1]);
    let sim = similarity(sample, c, tolerance, bounds, antialias);
    let v: Vec<f32> = sim.iter().map(|&s| s as f32 / 255.0).collect();
    to_grid(&v, bounds)
}

/// Contiguous region similar to the seed pixel (4-connected), with soft edges.
pub fn flood(sample: &Sample, seed: [i32; 2], tolerance: f32, bounds: IRect, antialias: bool) -> TileGrid {
    if !bounds.contains(seed[0], seed[1]) {
        return new_mask_grid();
    }
    let c = sample.pixel(seed[0], seed[1]);
    let sim = similarity(sample, c, tolerance, bounds, antialias);
    let w = bounds.width() as usize;
    let h = bounds.height() as usize;
    let mut out = vec![0.0f32; w * h];
    let mut visited = vec![false; w * h];
    let start = (seed[1] - bounds.y0) as usize * w + (seed[0] - bounds.x0) as usize;
    let mut q = VecDeque::new();
    q.push_back(start);
    visited[start] = true;
    while let Some(i) = q.pop_front() {
        let s = sim[i];
        out[i] = s as f32 / 255.0;
        if s < 128 {
            continue; // soft fringe: included, but doesn't propagate
        }
        let (x, y) = (i % w, i / w);
        let mut push = |j: usize| {
            if !visited[j] && sim[j] > 0 {
                visited[j] = true;
                q.push_back(j);
            }
        };
        if x > 0 {
            push(i - 1);
        }
        if x + 1 < w {
            push(i + 1);
        }
        if y > 0 {
            push(i - w);
        }
        if y + 1 < h {
            push(i + w);
        }
    }
    to_grid(&out, bounds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prisme_core::PixelFormat;

    fn two_regions() -> TileGrid {
        let mut g = TileGrid::rgba(PixelFormat::F32);
        g.map_rect(&IRect::new(0, 0, 300, 300), |x, _, p| {
            let v = if x < 100 || x >= 200 { [1.0, 0.0, 0.0, 1.0] } else { [0.0, 0.0, 1.0, 1.0] };
            p.copy_from_slice(&v);
        });
        g
    }

    #[test]
    fn flood_is_contiguous() {
        let g = two_regions();
        let m = flood(&Sample::Layer(&g), [10, 10], 0.1, IRect::new(0, 0, 300, 300), false);
        assert_eq!(m.get_pixel(50, 150)[0], 1.0);
        assert_eq!(m.get_pixel(150, 150)[0], 0.0);
        assert_eq!(m.get_pixel(250, 150)[0], 0.0);
    }

    #[test]
    fn color_range_is_global() {
        let g = two_regions();
        let m = color_range(&Sample::Layer(&g), [10, 10], 0.1, IRect::new(0, 0, 300, 300), false);
        assert_eq!(m.get_pixel(50, 150)[0], 1.0);
        assert_eq!(m.get_pixel(150, 150)[0], 0.0);
        assert_eq!(m.get_pixel(250, 150)[0], 1.0);
    }
}
