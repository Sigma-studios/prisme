//! Plane-level helpers: gaussian blur, distance-transform dilation, effect compositing.

use prisme_core::layer::{TextGlow, TextOutline, TextShadow};

/// Box sizes approximating a gaussian with three passes.
fn boxes_for_gauss(sigma: f32) -> [usize; 3] {
    let n = 3.0f32;
    let w_ideal = (12.0 * sigma * sigma / n + 1.0).sqrt();
    let mut wl = w_ideal.floor() as i32;
    if wl % 2 == 0 {
        wl -= 1;
    }
    let wl = wl.max(1);
    let wu = wl + 2;
    let m_ideal = (12.0 * sigma * sigma - n * (wl * wl) as f32 - 4.0 * n * wl as f32 - 3.0 * n)
        / (-4.0 * wl as f32 - 4.0);
    let m = m_ideal.round() as i32;
    std::array::from_fn(|i| {
        if (i as i32) < m {
            wl as usize
        } else {
            wu as usize
        }
    })
}

/// Reach (in pixels) of [`gaussian_blur`] for a given sigma.
pub(crate) fn blur_reach(sigma: f32) -> f32 {
    if sigma < 0.3 {
        0.0
    } else {
        (3.0 * sigma).ceil() + 1.0
    }
}

fn box_h(src: &[f32], dst: &mut [f32], w: usize, r: usize) {
    let inv = 1.0 / (2 * r + 1) as f32;
    let blur_row = |s: &[f32], d: &mut [f32]| {
        let mut acc = 0.0f32;
        for &v in s.iter().take(r.min(w)) {
            acc += v;
        }
        for x in 0..w {
            if x + r < w {
                acc += s[x + r];
            }
            d[x] = acc * inv;
            if x >= r {
                acc -= s[x - r];
            }
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        src.par_chunks(w)
            .zip(dst.par_chunks_mut(w))
            .for_each(|(s, d)| blur_row(s, d));
    }
    #[cfg(target_arch = "wasm32")]
    for (s, d) in src.chunks(w).zip(dst.chunks_mut(w)) {
        blur_row(s, d);
    }
}

fn box_v(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: usize) {
    let inv = 1.0 / (2 * r + 1) as f32;
    let mut acc = vec![0.0f32; w];
    for y in 0..r.min(h) {
        for (a, v) in acc.iter_mut().zip(&src[y * w..(y + 1) * w]) {
            *a += v;
        }
    }
    for y in 0..h {
        if y + r < h {
            for (a, v) in acc.iter_mut().zip(&src[(y + r) * w..(y + r + 1) * w]) {
                *a += v;
            }
        }
        for (d, a) in dst[y * w..(y + 1) * w].iter_mut().zip(&acc) {
            *d = a * inv;
        }
        if y >= r {
            for (a, v) in acc.iter_mut().zip(&src[(y - r) * w..(y - r + 1) * w]) {
                *a -= v;
            }
        }
    }
}

/// In-place gaussian blur (3 box passes) with zero outside the plane.
pub(crate) fn gaussian_blur(plane: &mut [f32], w: usize, h: usize, sigma: f32) {
    if sigma < 0.3 || w == 0 || h == 0 {
        return;
    }
    let mut tmp = vec![0.0f32; plane.len()];
    for b in boxes_for_gauss(sigma) {
        let r = (b - 1) / 2;
        if r == 0 {
            continue;
        }
        box_h(plane, &mut tmp, w, r);
        box_v(&tmp, plane, w, h, r);
    }
}

/// 1D squared euclidean distance transform (Felzenszwalb & Huttenlocher).
fn edt_1d(f: &[f32], d: &mut [f32], v: &mut [usize], z: &mut [f32]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;
    for q in 1..n {
        let fq = f[q] + (q * q) as f32;
        let mut s;
        loop {
            let p = v[k];
            s = (fq - (f[p] + (p * p) as f32)) / (2.0 * (q as f32 - p as f32));
            if s <= z[k] {
                // z[0] is -inf, so k never underflows.
                k -= 1;
            } else {
                break;
            }
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f32::INFINITY;
    }
    k = 0;
    for (q, dq) in d.iter_mut().enumerate() {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let p = v[k];
        let dx = q as f32 - p as f32;
        *dq = dx * dx + f[p];
    }
}

/// Dilate a coverage plane by `radius` pixels (round, anti-aliased edge). Returns the
/// dilated coverage (≥ input).
pub(crate) fn dilate(cov: &[f32], w: usize, h: usize, radius: f32) -> Vec<f32> {
    let inf = 2.0 * ((w + h) * (w + h)) as f32;
    let mut g: Vec<f32> = cov
        .iter()
        .map(|&a| if a >= 0.5 { 0.0 } else { inf })
        .collect();
    let n = w.max(h);
    let (mut f, mut d, mut v, mut z) = (
        vec![0.0f32; n],
        vec![0.0f32; n],
        vec![0usize; n],
        vec![0.0f32; n + 1],
    );
    // columns
    for x in 0..w {
        for y in 0..h {
            f[y] = g[y * w + x];
        }
        edt_1d(&f[..h], &mut d[..h], &mut v, &mut z);
        for y in 0..h {
            g[y * w + x] = d[y];
        }
    }
    // rows
    for y in 0..h {
        f[..w].copy_from_slice(&g[y * w..(y + 1) * w]);
        edt_1d(&f[..w], &mut d[..w], &mut v, &mut z);
        g[y * w..(y + 1) * w].copy_from_slice(&d[..w]);
    }
    g.iter()
        .zip(cov)
        .map(|(&d2, &a)| (radius + 0.5 - d2.sqrt()).clamp(0.0, 1.0).max(a))
        .collect()
}

/// Effect parameters converted to level pixels.
pub(crate) struct Effects {
    pub outline: Option<(f32, [f32; 4])>,
    pub glow: Option<(f32, [f32; 4], f32, bool)>,
    pub shadow: Option<([f32; 2], f32, [f32; 4])>,
}

impl Effects {
    pub fn new(
        o: &Option<TextOutline>,
        g: &Option<TextGlow>,
        s: &Option<TextShadow>,
        scale: f32,
    ) -> Self {
        Self {
            outline: o
                .as_ref()
                .filter(|o| o.width > 0.0 && o.color[3] > 0.0)
                .map(|o| (o.width * scale, o.color)),
            glow: g
                .as_ref()
                .filter(|g| g.radius > 0.0 && g.strength > 0.0 && g.color[3] > 0.0)
                .map(|g| (g.radius * scale * 0.5, g.color, g.strength, g.inner)),
            shadow: s.as_ref().filter(|s| s.color[3] > 0.0).map(|s| {
                (
                    [s.offset[0] * scale, s.offset[1] * scale],
                    s.blur * scale * 0.5,
                    s.color,
                )
            }),
        }
    }

    /// How far (level px) the effects extend beyond the shape, as (left, top, right, bottom).
    pub fn extents(&self) -> [f32; 4] {
        let ow = self.outline.map(|o| o.0).unwrap_or(0.0);
        let mut e = [ow; 4];
        if let Some((sigma, _, _, inner)) = self.glow
            && !inner
        {
            let r = ow + blur_reach(sigma);
            e = e.map(|v| v.max(r));
        }
        if let Some((off, sigma, _)) = self.shadow {
            let r = ow + blur_reach(sigma);
            e[0] = e[0].max(r - off[0]);
            e[1] = e[1].max(r - off[1]);
            e[2] = e[2].max(r + off[0]);
            e[3] = e[3].max(r + off[1]);
        }
        e
    }

    /// Context needed around an output region to compute the effects exactly.
    pub fn margin(&self) -> i32 {
        let ow = self.outline.map(|o| o.0).unwrap_or(0.0);
        let g = self.glow.map(|g| blur_reach(g.0)).unwrap_or(0.0);
        let s = self
            .shadow
            .map(|s| blur_reach(s.1) + s.0[0].abs().max(s.0[1].abs()))
            .unwrap_or(0.0);
        (ow + g.max(s)).ceil() as i32 + 2
    }
}

#[inline]
fn over(dst: &mut [f32; 4], c: [f32; 4], a: f32) {
    let a = a.clamp(0.0, 1.0) * c[3];
    if a <= 0.0 {
        return;
    }
    let k = 1.0 - a;
    dst[0] = c[0] * a + dst[0] * k;
    dst[1] = c[1] * a + dst[1] * k;
    dst[2] = c[2] * a + dst[2] * k;
    dst[3] = a + dst[3] * k;
}

/// Composite effects for the expanded region (`w`×`h`) and return straight-alpha RGBA for
/// the interior sub-rectangle starting at (`ix`, `iy`) of size `iw`×`ih`.
///
/// * `cov` — shape coverage, `fill` — premultiplied fill colour of the shape,
/// * `outline_cov` — coverage of shape ∪ outline (when an outline is enabled).
#[allow(clippy::too_many_arguments)]
pub(crate) fn composite(
    fx: &Effects,
    w: usize,
    h: usize,
    cov: &[f32],
    fill: &[[f32; 4]],
    outline_cov: Option<&[f32]>,
    (ix, iy, iw, ih): (usize, usize, usize, usize),
) -> Vec<f32> {
    let sil: &[f32] = outline_cov.unwrap_or(cov);
    let shadow = fx.shadow.map(|(off, sigma, col)| {
        let mut p = sil.to_vec();
        gaussian_blur(&mut p, w, h, sigma);
        (p, (off[0].round() as isize, off[1].round() as isize), col)
    });
    let glow = fx.glow.map(|(sigma, col, strength, inner)| {
        let mut p = if inner { cov.to_vec() } else { sil.to_vec() };
        gaussian_blur(&mut p, w, h, sigma);
        (p, col, strength, inner)
    });
    let mut out = vec![0.0f32; iw * ih * 4];
    for y in 0..ih {
        let gy = y + iy;
        for x in 0..iw {
            let gx = x + ix;
            let i = gy * w + gx;
            let mut d = [0.0f32; 4];
            if let Some((p, (ox, oy), col)) = &shadow {
                let sx = gx as isize - ox;
                let sy = gy as isize - oy;
                if sx >= 0 && sy >= 0 && (sx as usize) < w && (sy as usize) < h {
                    over(&mut d, *col, p[sy as usize * w + sx as usize]);
                }
            }
            if let Some((p, col, strength, false)) = &glow {
                over(&mut d, *col, (p[i] * 2.0 * strength).min(1.0));
            }
            if let (Some(oc), Some((_, col))) = (outline_cov, fx.outline) {
                over(&mut d, col, oc[i]);
            }
            // shape fill (premultiplied)
            let f = fill[i];
            let k = 1.0 - f[3];
            d = [
                f[0] + d[0] * k,
                f[1] + d[1] * k,
                f[2] + d[2] * k,
                f[3] + d[3] * k,
            ];
            if let Some((p, col, strength, true)) = &glow {
                over(
                    &mut d,
                    *col,
                    cov[i] * ((1.0 - p[i]) * 2.0 * strength).min(1.0),
                );
            }
            let o = (y * iw + x) * 4;
            if d[3] > 1e-6 {
                let inv = 1.0 / d[3];
                out[o] = d[0] * inv;
                out[o + 1] = d[1] * inv;
                out[o + 2] = d[2] * inv;
                out[o + 3] = d[3].min(1.0);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blur_preserves_mass() {
        let (w, h) = (64, 64);
        let mut p = vec![0.0f32; w * h];
        p[32 * w + 32] = 1.0;
        gaussian_blur(&mut p, w, h, 4.0);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-3, "{s}");
        assert!(p[32 * w + 32] < 0.05);
    }

    #[test]
    fn dilate_grows() {
        let (w, h) = (32, 32);
        let mut p = vec![0.0f32; w * h];
        p[16 * w + 16] = 1.0;
        let d = dilate(&p, w, h, 4.0);
        assert_eq!(d[16 * w + 19], 1.0);
        assert!((d[16 * w + 20] - 0.5).abs() < 1e-4);
        assert_eq!(d[16 * w + 21], 0.0);
        assert!((d[19 * w + 19] - (4.5 - 18f32.sqrt())).abs() < 1e-4);
    }
}
