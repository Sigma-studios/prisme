//! Separable resampling of straight-alpha RGBA f32 (premultiplied internally).

use crate::ResizeFilter;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// Run `f(row_index, row)` over every `row_len`-sized chunk of `out` (parallel on native).
pub(crate) fn for_each_row<F>(out: &mut [f32], row_len: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync + Send,
{
    if row_len == 0 {
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    out.par_chunks_mut(row_len).enumerate().for_each(|(y, row)| f(y, row));
    #[cfg(target_arch = "wasm32")]
    out.chunks_mut(row_len).enumerate().for_each(|(y, row)| f(y, row));
}

fn support(filter: ResizeFilter) -> f32 {
    match filter {
        ResizeFilter::Lanczos3 => 3.0,
        ResizeFilter::Mitchell => 2.0,
        ResizeFilter::Bilinear => 1.0,
        ResizeFilter::Nearest => 0.5,
    }
}

fn kernel(filter: ResizeFilter, x: f32) -> f32 {
    let x = x.abs();
    match filter {
        ResizeFilter::Lanczos3 => {
            if x < 1e-6 {
                1.0
            } else if x < 3.0 {
                let px = std::f32::consts::PI * x;
                3.0 * px.sin() * (px / 3.0).sin() / (px * px)
            } else {
                0.0
            }
        }
        ResizeFilter::Mitchell => {
            const B: f32 = 1.0 / 3.0;
            const C: f32 = 1.0 / 3.0;
            if x < 1.0 {
                ((12.0 - 9.0 * B - 6.0 * C) * x * x * x + (-18.0 + 12.0 * B + 6.0 * C) * x * x + (6.0 - 2.0 * B))
                    / 6.0
            } else if x < 2.0 {
                ((-B - 6.0 * C) * x * x * x
                    + (6.0 * B + 30.0 * C) * x * x
                    + (-12.0 * B - 48.0 * C) * x
                    + (8.0 * B + 24.0 * C))
                    / 6.0
            } else {
                0.0
            }
        }
        ResizeFilter::Bilinear => (1.0 - x).max(0.0),
        ResizeFilter::Nearest => 1.0,
    }
}

/// Per output sample: first source index and normalized weights.
struct Weights {
    start: Vec<usize>,
    len: Vec<usize>,
    w: Vec<f32>,
    stride: usize,
}

fn weights(src: usize, dst: usize, filter: ResizeFilter) -> Weights {
    let scale = src as f64 / dst as f64;
    if filter == ResizeFilter::Nearest {
        let start = (0..dst)
            .map(|x| (((x as f64 + 0.5) * scale).floor() as usize).min(src - 1))
            .collect();
        return Weights { start, len: vec![1; dst], w: vec![1.0; dst], stride: 1 };
    }
    let fscale = scale.max(1.0);
    let sup = support(filter) as f64 * fscale;
    let stride = (2.0 * sup).ceil() as usize + 2;
    let mut start = Vec::with_capacity(dst);
    let mut len = Vec::with_capacity(dst);
    let mut w = vec![0.0f32; dst * stride];
    for x in 0..dst {
        let center = (x as f64 + 0.5) * scale;
        let lo = ((center - sup).floor().max(0.0)) as usize;
        let hi = (((center + sup).ceil()) as usize).min(src);
        let hi = hi.max(lo + 1).min(lo + stride);
        let ws = &mut w[x * stride..x * stride + (hi - lo)];
        let mut sum = 0.0f64;
        for (k, i) in (lo..hi).enumerate() {
            let d = (i as f64 + 0.5 - center) / fscale;
            let v = kernel(filter, d as f32) as f64;
            ws[k] = v as f32;
            sum += v;
        }
        if sum.abs() < 1e-12 {
            // Degenerate: fall back to nearest.
            ws.iter_mut().for_each(|v| *v = 0.0);
            let n = ((center.floor() as usize).clamp(lo, hi - 1)) - lo;
            ws[n] = 1.0;
        } else {
            let inv = (1.0 / sum) as f32;
            ws.iter_mut().for_each(|v| *v *= inv);
        }
        start.push(lo);
        len.push(hi - lo);
    }
    Weights { start, len, w, stride }
}

/// Resample straight-alpha f32 RGBA from `w`×`h` to `nw`×`nh` (premultiplied internally to
/// avoid halos). Separable; parallel on native.
pub fn resize_rgba(pixels: &[f32], w: u32, h: u32, nw: u32, nh: u32, filter: ResizeFilter) -> Vec<f32> {
    let (w, h, nw, nh) = (w as usize, h as usize, nw as usize, nh as usize);
    if w == 0 || h == 0 || nw == 0 || nh == 0 || pixels.len() < w * h * 4 {
        return vec![0.0; nw * nh * 4];
    }
    if w == nw && h == nh {
        return pixels[..w * h * 4].to_vec();
    }
    // Premultiply.
    let mut pre = vec![0.0f32; w * h * 4];
    for_each_row(&mut pre, w * 4, |y, row| {
        let src = &pixels[y * w * 4..(y + 1) * w * 4];
        for (d, s) in row.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
            let a = if s[3].is_finite() { s[3].clamp(0.0, 1.0) } else { 0.0 };
            let f = |v: f32| if v.is_finite() { v * a } else { 0.0 };
            d[0] = f(s[0]);
            d[1] = f(s[1]);
            d[2] = f(s[2]);
            d[3] = a;
        }
    });

    // Horizontal pass: w×h -> nw×h.
    let tmp = if nw != w {
        let wx = weights(w, nw, filter);
        let mut tmp = vec![0.0f32; nw * h * 4];
        for_each_row(&mut tmp, nw * 4, |y, row| {
            let src = &pre[y * w * 4..(y + 1) * w * 4];
            for x in 0..nw {
                let (s0, n) = (wx.start[x], wx.len[x]);
                let ws = &wx.w[x * wx.stride..x * wx.stride + n];
                let mut acc = [0.0f32; 4];
                for (k, &wt) in ws.iter().enumerate() {
                    let p = &src[(s0 + k) * 4..(s0 + k) * 4 + 4];
                    acc[0] += p[0] * wt;
                    acc[1] += p[1] * wt;
                    acc[2] += p[2] * wt;
                    acc[3] += p[3] * wt;
                }
                row[x * 4..x * 4 + 4].copy_from_slice(&acc);
            }
        });
        tmp
    } else {
        pre
    };

    // Vertical pass: nw×h -> nw×nh, then unpremultiply.
    let mut out = vec![0.0f32; nw * nh * 4];
    let wy = if nh != h { Some(weights(h, nh, filter)) } else { None };
    for_each_row(&mut out, nw * 4, |y, row| {
        match &wy {
            Some(wy) => {
                let (s0, n) = (wy.start[y], wy.len[y]);
                let ws = &wy.w[y * wy.stride..y * wy.stride + n];
                row.iter_mut().for_each(|v| *v = 0.0);
                for (k, &wt) in ws.iter().enumerate() {
                    let src = &tmp[(s0 + k) * nw * 4..(s0 + k + 1) * nw * 4];
                    for (d, s) in row.iter_mut().zip(src) {
                        *d += s * wt;
                    }
                }
            }
            None => row.copy_from_slice(&tmp[y * nw * 4..(y + 1) * nw * 4]),
        }
        for p in row.chunks_exact_mut(4) {
            let a = p[3];
            if a > 1e-7 {
                let inv = 1.0 / a;
                p[0] *= inv;
                p[1] *= inv;
                p[2] *= inv;
                p[3] = a.min(1.0);
            } else {
                p.iter_mut().for_each(|v| *v = 0.0);
            }
        }
    });
    out
}
