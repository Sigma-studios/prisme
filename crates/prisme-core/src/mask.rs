//! Masks and selections: single-channel coverage grids plus the operations on them.

use crate::geom::IRect;
use crate::ids::MaskId;
use crate::tile::{PixelFormat, TileGrid};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A mask is a shared single-channel grid.
pub type Mask = Arc<TileGrid>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedMask {
    pub id: MaskId,
    pub name: String,
    pub mask: Mask,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SelectionOp {
    #[default]
    Replace,
    Add,
    Subtract,
    Intersect,
}

impl SelectionOp {
    pub fn name(self) -> &'static str {
        match self {
            SelectionOp::Replace => "Replace",
            SelectionOp::Add => "Add",
            SelectionOp::Subtract => "Subtract",
            SelectionOp::Intersect => "Intersect",
        }
    }
}

pub fn new_mask_grid() -> TileGrid {
    TileGrid::new(PixelFormat::F16, 1)
}

/// Combine an existing selection with a new shape.
pub fn combine(existing: Option<&TileGrid>, shape: &TileGrid, op: SelectionOp) -> TileGrid {
    let Some(a) = existing else {
        return match op {
            SelectionOp::Replace | SelectionOp::Add => shape.clone(),
            SelectionOp::Subtract | SelectionOp::Intersect => new_mask_grid(),
        };
    };
    match op {
        SelectionOp::Replace => shape.clone(),
        SelectionOp::Add | SelectionOp::Subtract | SelectionOp::Intersect => {
            let mut out = a.clone();
            let region = match op {
                SelectionOp::Add | SelectionOp::Subtract => shape.tile_bounds(),
                _ => a.tile_bounds(),
            };
            if region.is_empty() {
                return if op == SelectionOp::Intersect { new_mask_grid() } else { out };
            }
            let sv = shape.read_rect(&region);
            let av = a.read_rect(&region);
            let res: Vec<f32> = av
                .iter()
                .zip(&sv)
                .map(|(&x, &y)| match op {
                    SelectionOp::Add => x.max(y),
                    SelectionOp::Subtract => x * (1.0 - y),
                    _ => x.min(y),
                })
                .collect();
            if op == SelectionOp::Intersect {
                out = new_mask_grid();
            }
            out.write_rect(&region, &res);
            out.remove_clear_tiles();
            out
        }
    }
}

pub fn invert(sel: Option<&TileGrid>, canvas: IRect) -> TileGrid {
    let v = sel.map(|s| s.read_rect(&canvas)).unwrap_or_else(|| vec![0.0; canvas.area() as usize]);
    let inv: Vec<f32> = v.iter().map(|x| 1.0 - x).collect();
    let mut g = new_mask_grid();
    g.write_rect(&canvas, &inv);
    g
}

pub fn select_all(canvas: IRect) -> TileGrid {
    let mut g = new_mask_grid();
    g.write_rect(&canvas, &vec![1.0; canvas.area() as usize]);
    g
}

/// Rasterize a closed polygon (even-odd rule) into a mask with anti-aliasing.
pub fn polygon(points: &[[f32; 2]], antialias: bool) -> TileGrid {
    let mut g = new_mask_grid();
    if points.len() < 3 {
        return g;
    }
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in points {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    let rect = IRect::from_f32(x0, y0, x1, y1);
    if rect.is_empty() || rect.area() > 400_000_000 {
        return g;
    }
    let w = rect.width() as usize;
    let mut buf = vec![0.0f32; rect.area() as usize];
    let sub = if antialias { 4 } else { 1 };
    let mut xs: Vec<f32> = Vec::new();
    for row in 0..rect.height() {
        let line = &mut buf[row as usize * w..(row as usize + 1) * w];
        for s in 0..sub {
            let sy = rect.y0 as f32 + row as f32 + (s as f32 + 0.5) / sub as f32;
            xs.clear();
            for i in 0..points.len() {
                let a = points[i];
                let b = points[(i + 1) % points.len()];
                if (a[1] <= sy && b[1] > sy) || (b[1] <= sy && a[1] > sy) {
                    let t = (sy - a[1]) / (b[1] - a[1]);
                    xs.push(a[0] + t * (b[0] - a[0]));
                }
            }
            xs.sort_by(|a, b| a.total_cmp(b));
            for pair in xs.chunks_exact(2) {
                let (xa, xb) = (pair[0] - rect.x0 as f32, pair[1] - rect.x0 as f32);
                if antialias {
                    accumulate_span(line, xa, xb, 1.0 / sub as f32);
                } else {
                    // pixel centres inside the span
                    let s = (xa - 0.5).ceil().max(0.0) as usize;
                    let e = ((xb - 0.5).floor() as isize).min(w as isize - 1);
                    if e >= s as isize {
                        for v in &mut line[s..=e as usize] {
                            *v = 1.0;
                        }
                    }
                }
            }
        }
    }
    for v in &mut buf {
        *v = v.min(1.0);
    }
    g.write_rect(&rect, &buf);
    g
}

/// Add exact horizontal coverage of [xa, xb) to a row, scaled by `weight`.
fn accumulate_span(line: &mut [f32], xa: f32, xb: f32, weight: f32) {
    let w = line.len() as f32;
    let xa = xa.clamp(0.0, w);
    let xb = xb.clamp(0.0, w);
    if xb <= xa {
        return;
    }
    let ia = xa.floor() as usize;
    let ib = xb.floor() as usize;
    if ia == ib {
        line[ia.min(line.len() - 1)] += (xb - xa) * weight;
        return;
    }
    line[ia] += (ia as f32 + 1.0 - xa) * weight;
    let end = ib.min(line.len());
    for v in &mut line[ia + 1..end] {
        *v += weight;
    }
    if ib < line.len() {
        line[ib] += (xb - ib as f32) * weight;
    }
}

pub fn rect(r: [f32; 4], antialias: bool) -> TileGrid {
    polygon(&[[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]], antialias)
}

pub fn ellipse(r: [f32; 4], antialias: bool) -> TileGrid {
    let (cx, cy) = ((r[0] + r[2]) * 0.5, (r[1] + r[3]) * 0.5);
    let (rx, ry) = (((r[2] - r[0]) * 0.5).abs(), ((r[3] - r[1]) * 0.5).abs());
    let n = ((rx.max(ry) * 0.75) as usize).clamp(32, 2048);
    let pts: Vec<[f32; 2]> = (0..n)
        .map(|i| {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            [cx + rx * a.cos(), cy + ry * a.sin()]
        })
        .collect();
    polygon(&pts, antialias)
}

/// Gaussian-ish feather (three box blur passes) of radius `r` pixels.
pub fn feather(sel: &TileGrid, r: f32) -> TileGrid {
    if r <= 0.0 {
        return sel.clone();
    }
    let b = sel.content_bounds();
    if b.is_empty() {
        return sel.clone();
    }
    let pad = (r * 2.0).ceil() as i32 + 2;
    let region = b.expand(pad);
    let (w, h) = (region.width() as usize, region.height() as usize);
    let mut buf = sel.read_rect(&region);
    // box radius for 3 passes approximating a gaussian of sigma ~ r/2
    let sigma = r / 2.0;
    let br = (((12.0 * sigma * sigma / 3.0) + 1.0).sqrt() / 2.0).round().max(1.0) as usize;
    for _ in 0..3 {
        box_blur_h(&mut buf, w, h, br);
        box_blur_v(&mut buf, w, h, br);
    }
    let mut out = new_mask_grid();
    out.write_rect(&region, &buf);
    out
}

pub fn box_blur_h(buf: &mut [f32], w: usize, h: usize, r: usize) {
    let mut tmp = vec![0.0f32; w];
    let inv = 1.0 / (2 * r + 1) as f32;
    for y in 0..h {
        let row = &mut buf[y * w..(y + 1) * w];
        let mut acc = 0.0;
        for i in 0..=(2 * r) {
            acc += row[i.saturating_sub(r).min(w - 1)];
        }
        for x in 0..w {
            tmp[x] = acc * inv;
            let add = (x + r + 1).min(w - 1);
            let sub = x.saturating_sub(r);
            acc += row[add] - row[sub];
        }
        row.copy_from_slice(&tmp);
    }
}

pub fn box_blur_v(buf: &mut [f32], w: usize, h: usize, r: usize) {
    let mut tmp = vec![0.0f32; h];
    let inv = 1.0 / (2 * r + 1) as f32;
    for x in 0..w {
        let mut acc = 0.0;
        for i in 0..=(2 * r) {
            acc += buf[i.saturating_sub(r).min(h - 1) * w + x];
        }
        for y in 0..h {
            tmp[y] = acc * inv;
            let add = (y + r + 1).min(h - 1);
            let sub = y.saturating_sub(r);
            acc += buf[add * w + x] - buf[sub * w + x];
        }
        for y in 0..h {
            buf[y * w + x] = tmp[y];
        }
    }
}

/// 1D squared Euclidean distance transform (Felzenszwalb & Huttenlocher).
fn edt_1d(f: &[f32], d: &mut [f32], v: &mut [usize], z: &mut [f32]) {
    let n = f.len();
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;
    for q in 1..n {
        loop {
            let p = v[k];
            let s = ((f[q] + (q * q) as f32) - (f[p] + (p * p) as f32)) / (2.0 * q as f32 - 2.0 * p as f32);
            if s <= z[k] && k > 0 {
                k -= 1;
                continue;
            }
            if s <= z[k] {
                // k == 0
                v[0] = q;
                z[0] = f32::NEG_INFINITY;
                z[1] = f32::INFINITY;
                break;
            }
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = f32::INFINITY;
            break;
        }
    }
    k = 0;
    for q in 0..n {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let p = v[k];
        d[q] = (q as f32 - p as f32).powi(2) + f[p];
    }
}

/// Euclidean distance (pixels) from each pixel to the nearest pixel where `inside` is true.
pub fn distance_field(inside: &[bool], w: usize, h: usize) -> Vec<f32> {
    const INF: f32 = 1e20;
    let mut grid: Vec<f32> = inside.iter().map(|&b| if b { 0.0 } else { INF }).collect();
    let n = w.max(h);
    let mut f = vec![0.0f32; n];
    let mut d = vec![0.0f32; n];
    let mut v = vec![0usize; n];
    let mut z = vec![0.0f32; n + 1];
    for x in 0..w {
        for y in 0..h {
            f[y] = grid[y * w + x];
        }
        edt_1d(&f[..h], &mut d[..h], &mut v[..h], &mut z[..h + 1]);
        for y in 0..h {
            grid[y * w + x] = d[y];
        }
    }
    for y in 0..h {
        f[..w].copy_from_slice(&grid[y * w..(y + 1) * w]);
        edt_1d(&f[..w], &mut d[..w], &mut v[..w], &mut z[..w + 1]);
        for x in 0..w {
            grid[y * w + x] = d[x].sqrt();
        }
    }
    grid
}

/// Grow (`by > 0`) or shrink (`by < 0`) a selection by a Euclidean distance with a soft edge.
pub fn grow(sel: &TileGrid, by: f32) -> TileGrid {
    if by == 0.0 {
        return sel.clone();
    }
    let b = sel.content_bounds();
    if b.is_empty() {
        return sel.clone();
    }
    let pad = by.abs().ceil() as i32 + 2;
    let region = b.expand(pad);
    let (w, h) = (region.width() as usize, region.height() as usize);
    let buf = sel.read_rect(&region);
    let out: Vec<f32> = if by > 0.0 {
        let inside: Vec<bool> = buf.iter().map(|&v| v >= 0.5).collect();
        let d = distance_field(&inside, w, h);
        d.iter().zip(&buf).map(|(&d, &v)| v.max((by + 0.5 - d).clamp(0.0, 1.0))).collect()
    } else {
        let outside: Vec<bool> = buf.iter().map(|&v| v < 0.5).collect();
        let d = distance_field(&outside, w, h);
        let s = -by;
        d.iter().zip(&buf).map(|(&d, &v)| v.min((d - s - 0.5 + 1.0).clamp(0.0, 1.0))).collect()
    };
    let mut g = new_mask_grid();
    g.write_rect(&region, &out);
    g
}

/// Border selection of width `w` around the current edge.
pub fn border(sel: &TileGrid, width: f32) -> TileGrid {
    let outer = grow(sel, width / 2.0);
    let inner = grow(sel, -width / 2.0);
    combine(Some(&outer), &inner, SelectionOp::Subtract)
}

/// Threshold the mask to a hard edge.
pub fn harden(sel: &TileGrid) -> TileGrid {
    let mut g = sel.clone();
    let b = g.tile_bounds();
    g.map_rect(&b, |_, _, v| v[0] = if v[0] >= 0.5 { 1.0 } else { 0.0 });
    g.remove_clear_tiles();
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_area_is_exact() {
        let g = rect([10.25, 10.0, 20.75, 30.0], true);
        let b = g.tile_bounds();
        let sum: f32 = g.read_rect(&b).iter().sum();
        assert!((sum - 10.5 * 20.0).abs() < 0.01, "{sum}");
    }

    #[test]
    fn ellipse_area_close() {
        let g = ellipse([0.0, 0.0, 200.0, 100.0], true);
        let sum: f32 = g.read_rect(&g.tile_bounds()).iter().sum();
        let expect = std::f32::consts::PI * 100.0 * 50.0;
        assert!((sum - expect).abs() / expect < 0.005, "{sum} vs {expect}");
    }

    #[test]
    fn grow_then_shrink() {
        let g = rect([50.0, 50.0, 100.0, 100.0], false);
        let grown = grow(&g, 10.0);
        assert!(grown.get_pixel(45, 75)[0] > 0.9);
        assert!(grown.get_pixel(35, 75)[0] < 0.1);
        let shrunk = grow(&g, -10.0);
        assert!(shrunk.get_pixel(55, 75)[0] < 0.1);
        assert!(shrunk.get_pixel(65, 75)[0] > 0.9);
    }

    #[test]
    fn combine_ops() {
        let a = rect([0.0, 0.0, 10.0, 10.0], false);
        let b = rect([5.0, 0.0, 15.0, 10.0], false);
        let add = combine(Some(&a), &b, SelectionOp::Add);
        assert_eq!(add.get_pixel(12, 5)[0], 1.0);
        let sub = combine(Some(&a), &b, SelectionOp::Subtract);
        assert_eq!(sub.get_pixel(7, 5)[0], 0.0);
        assert_eq!(sub.get_pixel(2, 5)[0], 1.0);
        let int = combine(Some(&a), &b, SelectionOp::Intersect);
        assert_eq!(int.get_pixel(2, 5)[0], 0.0);
        assert_eq!(int.get_pixel(7, 5)[0], 1.0);
    }
}
