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
    to_grid(&flood_values(sample, seed, tolerance, bounds, antialias), bounds)
}

fn flood_values(sample: &Sample, seed: [i32; 2], tolerance: f32, bounds: IRect, antialias: bool) -> Vec<f32> {
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
    out
}

/// Paint-bucket region: the flood core plus its anti-aliased fringe.
pub struct FillRegion {
    /// Seed colour (straight alpha).
    pub seed: [f32; 4],
    /// Flood mask (as `flood`).
    pub core: TileGrid,
    /// Per-pixel share of the seed colour in the edge pixels just outside the core
    /// (blends of seed and barrier colour the tolerance left out).
    pub fringe: TileGrid,
}

/// Fringe rings explored outside the flood core.
const FRINGE_RINGS: usize = 2;

/// `flood`, plus an estimate of how much seed colour the barrier's anti-aliased edge
/// pixels contain, so a fill can replace exactly that share instead of leaving a halo.
pub fn fill_flood(sample: &Sample, seed: [i32; 2], tolerance: f32, bounds: IRect, antialias: bool) -> FillRegion {
    let c = sample.pixel(seed[0], seed[1]);
    if !bounds.contains(seed[0], seed[1]) {
        return FillRegion { seed: c, core: new_mask_grid(), fringe: new_mask_grid() };
    }
    let mut out = flood_values(sample, seed, tolerance, bounds, antialias);
    if !antialias {
        return FillRegion { seed: c, core: to_grid(&out, bounds), fringe: new_mask_grid() };
    }
    let w = bounds.width() as usize;
    let h = bounds.height() as usize;
    let mut px = PixelCache::new(sample);
    let s = premul(c);
    // Seed share of pixel `i`: project it onto the seed→barrier segment, the barrier
    // being the least seed-like colour nearby.
    let mut share = |i: usize| -> f32 {
        let (x, y) = ((i % w) as i32 + bounds.x0, (i / w) as i32 + bounds.y0);
        let p = premul(px.get(x, y));
        let mut b = s;
        let mut bd = 0.0;
        for dy in -2..=2 {
            for dx in -2..=2 {
                let q = premul(px.get(x + dx, y + dy));
                let d = dist2(q, s);
                if d > bd {
                    bd = d;
                    b = q;
                }
            }
        }
        if bd < 1e-6 {
            return 0.0;
        }
        let a = (0..4).map(|k| (p[k] - b[k]) * (s[k] - b[k])).sum::<f32>() / bd;
        let a = a.clamp(0.0, 1.0);
        // Not a blend of the two (some third colour): leave it alone.
        let resid: f32 = (0..4).map(|k| (p[k] - (b[k] + a * (s[k] - b[k]))).powi(2)).sum();
        if resid > 0.25 * bd {
            0.0
        } else {
            a
        }
    };
    let mut fringe = vec![0.0f32; w * h];
    let mut seen = vec![false; w * h];
    // Soft flood edges are re-evaluated as fringe (the share is the better estimate);
    // the outer ones don't propagate, the inner ones still seed the rings.
    let mut ring: Vec<(usize, f32)> = Vec::new();
    for i in 0..w * h {
        if out[i] >= 0.5 {
            seen[i] = true;
            ring.push((i, 1.0));
            if out[i] < 1.0 {
                let a = share(i);
                if a > 0.0 {
                    fringe[i] = a;
                    out[i] = 0.0;
                }
            }
        } else {
            out[i] = 0.0;
        }
    }
    for _ in 0..FRINGE_RINGS {
        let mut next = Vec::new();
        for &(i, from) in &ring {
            let (x, y) = (i % w, i / w);
            let mut visit = |j: usize| {
                if seen[j] {
                    return;
                }
                let a = share(j);
                // Shares must fall moving outward, or we'd leak past a thin line.
                if a > 0.02 && a < from {
                    seen[j] = true;
                    fringe[j] = a;
                    next.push((j, a));
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < w {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - w);
            }
            if y + 1 < h {
                visit(i + w);
            }
        }
        ring = next;
    }
    FillRegion { seed: c, core: to_grid(&out, bounds), fringe: to_grid(&fringe, bounds) }
}

/// Gamma-encoded, premultiplied: anti-aliasing is usually blended in encoded values.
pub(crate) fn premul(p: [f32; 4]) -> [f32; 4] {
    let a = p[3].clamp(0.0, 1.0);
    let e = prisme_color::linear_to_srgb;
    [e(p[0]) * a, e(p[1]) * a, e(p[2]) * a, a]
}

fn dist2(a: [f32; 4], b: [f32; 4]) -> f32 {
    (0..4).map(|k| (a[k] - b[k]).powi(2)).sum()
}

/// Per-pixel reads, flattening each merged tile once.
struct PixelCache<'a> {
    sample: &'a Sample<'a>,
    tiles: std::collections::HashMap<TileCoord, Option<Vec<f32>>>,
}

impl<'a> PixelCache<'a> {
    fn new(sample: &'a Sample<'a>) -> Self {
        Self { sample, tiles: Default::default() }
    }

    fn get(&mut self, x: i32, y: i32) -> [f32; 4] {
        if let Sample::Layer(g) = self.sample {
            return g.get_pixel(x, y);
        }
        let c = TileCoord::of_pixel(x, y);
        let sample = self.sample;
        let t = self.tiles.entry(c).or_insert_with(|| sample.tile(c));
        match t {
            Some(d) => {
                let i = (((y - c.y * TS) * TS + (x - c.x * TS)) * 4) as usize;
                [d[i], d[i + 1], d[i + 2], d[i + 3]]
            }
            None => [0.0; 4],
        }
    }
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

    /// White, an anti-aliased black line at x = 100..103 (sRGB grey ramps either side), white.
    fn aa_line() -> TileGrid {
        let mut g = TileGrid::rgba(PixelFormat::F32);
        g.map_rect(&IRect::new(0, 0, 200, 50), |x, _, p| {
            let v = match x {
                99 => 0.75,
                100 => 0.3,
                101 => 0.0,
                102 => 0.3,
                103 => 0.75,
                _ => 1.0,
            };
            let v = prisme_color::srgb_to_linear(v);
            p.copy_from_slice(&[v, v, v, 1.0]);
        });
        g
    }

    #[test]
    fn fill_fringe_covers_aa_edge_without_leaking() {
        let g = aa_line();
        let r = fill_flood(&Sample::Layer(&g), [10, 10], 0.12, IRect::new(0, 0, 200, 50), true);
        assert_eq!(r.core.get_pixel(98, 10)[0], 1.0);
        // The grey ramp is outside tolerance but carries its seed share.
        assert!((r.fringe.get_pixel(99, 10)[0] - 0.75).abs() < 0.05);
        assert!((r.fringe.get_pixel(100, 10)[0] - 0.3).abs() < 0.05);
        assert_eq!(r.fringe.get_pixel(101, 10)[0], 0.0);
        // Nothing past the line.
        for x in 102..200 {
            assert_eq!(r.core.get_pixel(x, 10)[0] + r.fringe.get_pixel(x, 10)[0], 0.0, "x {x}");
        }
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
