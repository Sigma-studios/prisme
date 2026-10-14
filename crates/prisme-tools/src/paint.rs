//! Pixel operations used by fill, gradient and retouching tools.

use prisme_brush::Dab;
use prisme_core::gradient::{dither_noise, Gradient, GradientGeometry};
use prisme_core::tile::TileGrid;
use prisme_core::IRect;
use std::sync::Arc;

fn over(dst: &mut [f32], col: [f32; 4], a: f32) {
    let a = a * col[3];
    if a <= 0.0 {
        return;
    }
    let ab = dst[3];
    let ao = a + ab * (1.0 - a);
    for k in 0..3 {
        dst[k] = (col[k] * a + dst[k] * ab * (1.0 - a)) / ao.max(1e-7);
    }
    dst[3] = ao;
}

fn sel_at(sel: Option<&TileGrid>, x: i32, y: i32) -> f32 {
    sel.map(|s| s.get_pixel(x, y)[0]).unwrap_or(1.0)
}

/// Paint `color` over `grid` weighted by `region` (and the selection).
pub fn fill_region(grid: &mut TileGrid, region: &TileGrid, color: [f32; 4], opacity: f32, sel: Option<&TileGrid>) {
    let b = region.tile_bounds();
    let rv = region.read_rect(&b);
    let w = b.width() as usize;
    let ch = grid.channels as usize;
    grid.map_rect(&b, |x, y, p| {
        let a = rv[(y - b.y0) as usize * w + (x - b.x0) as usize] * opacity * sel_at(sel, x, y);
        if a > 0.0 {
            if ch == 4 {
                over(p, color, a);
            } else {
                p[0] += (color[0] - p[0]) * a;
            }
        }
    });
    grid.remove_clear_tiles();
}

/// Render a gradient into `grid` over `bounds` (masked by the selection).
pub fn draw_gradient(grid: &mut TileGrid, bounds: IRect, g: &Gradient, geo: &GradientGeometry, opacity: f32, sel: Option<&TileGrid>) {
    let lut = g.to_lut(1024);
    let ch = grid.channels as usize;
    grid.map_rect(&bounds, |x, y, p| {
        let a = opacity * sel_at(sel, x, y);
        if a <= 0.0 {
            return;
        }
        let mut t = geo.t_at([x as f32 + 0.5, y as f32 + 0.5]);
        if geo.dither {
            t += dither_noise(x, y) / 512.0;
        }
        let c = lut[((t.clamp(0.0, 1.0) * 1023.0).round() as usize).min(1023)];
        if ch == 4 {
            over(p, c, a);
        } else {
            p[0] += (prisme_color::working_luminance([c[0], c[1], c[2]]) - p[0]) * a;
        }
    });
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RetouchMode {
    Smudge { strength: f32 },
    Blur { strength: f32 },
    Clone { offset: [f32; 2] },
    Heal { offset: [f32; 2] },
}

/// Per-dab retouching that reads pixels (smudge, blur, clone, heal).
pub struct RetouchCanvas {
    pub mode: RetouchMode,
    /// Snapshot of the layer at stroke start (clone/heal source; avoids feedback).
    base: Arc<TileGrid>,
    carried: Option<[f32; 4]>,
    selection: Option<Arc<TileGrid>>,
    pub dirty: IRect,
}

impl RetouchCanvas {
    pub fn new(mode: RetouchMode, base: Arc<TileGrid>, selection: Option<Arc<TileGrid>>) -> Self {
        Self { mode, base, carried: None, selection, dirty: IRect::EMPTY }
    }

    pub fn apply(&mut self, dabs: &[Dab], out: &mut TileGrid) {
        for d in dabs {
            self.apply_dab(d, out);
        }
    }

    fn apply_dab(&mut self, d: &Dab, out: &mut TileGrid) {
        let b = d.bounds();
        let r = IRect::new(b[0], b[1], b[2], b[3]);
        if out.channels != 4 {
            return;
        }
        self.dirty = self.dirty.union(&r);
        let w = r.width() as usize;
        let h = r.height() as usize;
        let mut px = out.read_rect(&r);
        let cov: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = (r.x0 + (i % w) as i32, r.y0 + (i / w) as i32);
                d.coverage(x as f32 + 0.5, y as f32 + 0.5) * d.alpha * sel_at(self.selection.as_deref(), x, y)
            })
            .collect();
        let weighted_mean = |buf: &[f32]| -> [f32; 4] {
            let mut acc = [0.0f32; 4];
            let mut wsum = 0.0;
            for i in 0..w * h {
                let c = cov[i];
                if c <= 0.0 {
                    continue;
                }
                let a = buf[i * 4 + 3];
                for k in 0..3 {
                    acc[k] += buf[i * 4 + k] * a * c;
                }
                acc[3] += a * c;
                wsum += c;
            }
            if acc[3] > 1e-6 {
                [acc[0] / acc[3], acc[1] / acc[3], acc[2] / acc[3], acc[3] / wsum.max(1e-6)]
            } else {
                [0.0; 4]
            }
        };
        match self.mode {
            RetouchMode::Smudge { strength } => {
                let local = weighted_mean(&px);
                let carried = self.carried.unwrap_or(local);
                for i in 0..w * h {
                    let c = cov[i] * strength;
                    if c > 0.0 {
                        let p = &mut px[i * 4..i * 4 + 4];
                        for k in 0..4 {
                            p[k] += (carried[k] - p[k]) * c;
                        }
                    }
                }
                let mut next = [0.0; 4];
                for k in 0..4 {
                    next[k] = carried[k] + (local[k] - carried[k]) * (1.0 - strength);
                }
                self.carried = Some(next);
            }
            RetouchMode::Blur { strength } => {
                let rad = (d.radius * 0.15).clamp(1.0, 8.0) as i32;
                let src = out.read_rect(&r.expand(rad));
                let sw = w + 2 * rad as usize;
                for y in 0..h {
                    for x in 0..w {
                        let c = cov[y * w + x] * strength;
                        if c <= 0.0 {
                            continue;
                        }
                        let mut acc = [0.0f32; 4];
                        let mut n = 0.0;
                        for oy in -rad..=rad {
                            for ox in -rad..=rad {
                                let sx = (x as i32 + rad + ox) as usize;
                                let sy = (y as i32 + rad + oy) as usize;
                                let s = &src[(sy * sw + sx) * 4..][..4];
                                acc[0] += s[0] * s[3];
                                acc[1] += s[1] * s[3];
                                acc[2] += s[2] * s[3];
                                acc[3] += s[3];
                                n += 1.0;
                            }
                        }
                        let blurred = if acc[3] > 1e-6 {
                            [acc[0] / acc[3], acc[1] / acc[3], acc[2] / acc[3], acc[3] / n]
                        } else {
                            [0.0; 4]
                        };
                        let p = &mut px[(y * w + x) * 4..][..4];
                        for k in 0..4 {
                            p[k] += (blurred[k] - p[k]) * c;
                        }
                    }
                }
            }
            RetouchMode::Clone { offset } | RetouchMode::Heal { offset } => {
                let (ox, oy) = (offset[0].round() as i32, offset[1].round() as i32);
                let src = self.base.read_rect(&r.translate(ox, oy));
                let shift = if let RetouchMode::Heal { .. } = self.mode {
                    // Match low frequencies: add (mean(dest) - mean(src)) inside the dab.
                    let md = weighted_mean(&px);
                    let ms = weighted_mean(&src);
                    [md[0] - ms[0], md[1] - ms[1], md[2] - ms[2]]
                } else {
                    [0.0; 3]
                };
                for i in 0..w * h {
                    let c = cov[i];
                    if c <= 0.0 {
                        continue;
                    }
                    let s = &src[i * 4..i * 4 + 4];
                    let target = [s[0] + shift[0], s[1] + shift[1], s[2] + shift[2], s[3]];
                    let p = &mut px[i * 4..i * 4 + 4];
                    for k in 0..4 {
                        p[k] += (target[k] - p[k]) * c;
                    }
                }
            }
        }
        out.write_rect(&r, &px);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prisme_core::PixelFormat;

    #[test]
    fn clone_copies_source() {
        let mut g = TileGrid::rgba(PixelFormat::F32);
        g.map_rect(&IRect::new(0, 0, 100, 100), |x, _, p| {
            let v = if x < 50 { 0.0 } else { 1.0 };
            p.copy_from_slice(&[v, v, v, 1.0]);
        });
        let base = Arc::new(g.clone());
        let mut c = RetouchCanvas::new(RetouchMode::Clone { offset: [60.0, 0.0] }, base, None);
        let d = Dab { center: [20.0, 50.0], radius: 6.0, angle: 0.0, roundness: 1.0, hardness: 1.0, alpha: 1.0, tip: prisme_brush::TipShape::Round, pixel: false, grain: 0.0, seed: 0 };
        c.apply(&[d], &mut g);
        assert!(g.get_pixel(20, 50)[0] > 0.99);
        assert!(g.get_pixel(5, 50)[0] < 0.01);
    }
}
