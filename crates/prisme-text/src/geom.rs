//! Glyph outlines, flattening and warps.

use prisme_core::layer::TextWarp;
use rustybuzz::ttf_parser;
use std::f32::consts::PI;

pub(crate) type P2 = [f32; 2];

/// Outline segment in font units (y up).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Seg {
    Move(P2),
    Line(P2),
    Quad(P2, P2),
    Cubic(P2, P2, P2),
    Close,
}

#[derive(Default)]
pub(crate) struct OutlineCollector(pub Vec<Seg>);

impl ttf_parser::OutlineBuilder for OutlineCollector {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.push(Seg::Move([x, y]));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.0.push(Seg::Line([x, y]));
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.0.push(Seg::Quad([x1, y1], [x, y]));
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.0.push(Seg::Cubic([x1, y1], [x2, y2], [x, y]));
    }
    fn close(&mut self) {
        self.0.push(Seg::Close);
    }
}

#[inline]
fn len(a: P2) -> f32 {
    (a[0] * a[0] + a[1] * a[1]).sqrt()
}

/// Flatten an outline into closed polylines in text space.
/// `map` converts font units to text space; `tol` is the flattening tolerance in text
/// space and `max_len` the maximum segment length (for non-linear warps/3D).
pub(crate) fn flatten(
    segs: &[Seg],
    map: impl Fn(P2) -> P2,
    tol: f32,
    max_len: f32,
    out: &mut Vec<Vec<P2>>,
) {
    let mut cur: Vec<P2> = Vec::new();
    let mut last = [0.0f32; 2];
    let tol = tol.max(1e-4);
    let push_line = |cur: &mut Vec<P2>, a: P2, b: P2| {
        let l = len([b[0] - a[0], b[1] - a[1]]);
        let n = if max_len.is_finite() {
            ((l / max_len).ceil() as usize).clamp(1, 4096)
        } else {
            1
        };
        for k in 1..=n {
            let t = k as f32 / n as f32;
            cur.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
        }
    };
    let finish = |cur: &mut Vec<P2>, out: &mut Vec<Vec<P2>>| {
        if cur.len() >= 3 {
            out.push(std::mem::take(cur));
        } else {
            cur.clear();
        }
    };
    for s in segs {
        match *s {
            Seg::Move(p) => {
                finish(&mut cur, out);
                last = map(p);
                cur.push(last);
            }
            Seg::Line(p) => {
                let q = map(p);
                push_line(&mut cur, last, q);
                last = q;
            }
            Seg::Quad(c, p) => {
                let (p0, c1, p1) = (last, map(c), map(p));
                let dd = len([p0[0] - 2.0 * c1[0] + p1[0], p0[1] - 2.0 * c1[1] + p1[1]]);
                let n = ((dd / (8.0 * tol)).sqrt().ceil() as usize).clamp(1, 256);
                let mut prev = p0;
                for k in 1..=n {
                    let t = k as f32 / n as f32;
                    let u = 1.0 - t;
                    let q = [
                        u * u * p0[0] + 2.0 * u * t * c1[0] + t * t * p1[0],
                        u * u * p0[1] + 2.0 * u * t * c1[1] + t * t * p1[1],
                    ];
                    push_line(&mut cur, prev, q);
                    prev = q;
                }
                last = p1;
            }
            Seg::Cubic(c1, c2, p) => {
                let (p0, a, b, p1) = (last, map(c1), map(c2), map(p));
                let d1 = len([p0[0] - 2.0 * a[0] + b[0], p0[1] - 2.0 * a[1] + b[1]]);
                let d2 = len([a[0] - 2.0 * b[0] + p1[0], a[1] - 2.0 * b[1] + p1[1]]);
                let n = ((0.75 * d1.max(d2) / tol).sqrt().ceil() as usize).clamp(1, 256);
                let mut prev = p0;
                for k in 1..=n {
                    let t = k as f32 / n as f32;
                    let u = 1.0 - t;
                    let (w0, w1, w2, w3) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
                    let q = [
                        w0 * p0[0] + w1 * a[0] + w2 * b[0] + w3 * p1[0],
                        w0 * p0[1] + w1 * a[1] + w2 * b[1] + w3 * p1[1],
                    ];
                    push_line(&mut cur, prev, q);
                    prev = q;
                }
                last = p1;
            }
            Seg::Close => {
                if let Some(&first) = cur.first()
                    && (first[0] - last[0]).abs() + (first[1] - last[1]).abs() > 1e-6
                {
                    push_line(&mut cur, last, first);
                }
                // drop the duplicated closing point
                if cur.len() > 1 {
                    let (f, l) = (cur[0], cur[cur.len() - 1]);
                    if (f[0] - l[0]).abs() + (f[1] - l[1]).abs() < 1e-6 {
                        cur.pop();
                    }
                }
                finish(&mut cur, out);
            }
        }
    }
    finish(&mut cur, out);
}

/// Signed area (shoelace, y-down text space).
pub(crate) fn signed_area(c: &[P2]) -> f32 {
    let n = c.len();
    let mut a = 0.0;
    for i in 0..n {
        let (p, q) = (c[i], c[(i + 1) % n]);
        a += p[0] * q[1] - q[0] * p[1];
    }
    a * 0.5
}

/// Whether a warp is active (needs dense outlines).
pub(crate) fn warp_active(w: &TextWarp) -> bool {
    match *w {
        TextWarp::None => false,
        TextWarp::Arc { bend } => bend.abs() > 1e-4,
        TextWarp::Wave {
            amplitude,
            wavelength,
        }
        | TextWarp::Flag {
            amplitude,
            wavelength,
        } => amplitude.abs() > 1e-4 && wavelength.abs() > 1e-3,
        TextWarp::Bulge { amount } => amount.abs() > 1e-4,
    }
}

/// Apply a warp to a text-space point. `size` = text block size `[w, h]`.
pub(crate) fn warp_point(w: &TextWarp, size: [f32; 2], p: P2) -> P2 {
    let (bw, bh) = (size[0].max(1e-3), size[1].max(1e-3));
    let (cx, cy) = (bw * 0.5, bh * 0.5);
    match *w {
        TextWarp::None => p,
        TextWarp::Arc { bend } => {
            let b = bend.clamp(-1.0, 1.0);
            if b.abs() < 1e-4 {
                return p;
            }
            let theta = b.abs() * PI; // bend 1 → half circle
            let r = bw / theta;
            let s = b.signum();
            let phi = (p[0] - cx) / r;
            let rad = r - s * (p[1] - cy);
            [cx + rad * phi.sin(), cy + s * (r - rad * phi.cos())]
        }
        TextWarp::Wave {
            amplitude,
            wavelength,
        } => {
            if wavelength.abs() < 1e-3 {
                return p;
            }
            [
                p[0],
                p[1] + amplitude * (2.0 * PI * p[0] / wavelength).sin(),
            ]
        }
        TextWarp::Flag {
            amplitude,
            wavelength,
        } => {
            if wavelength.abs() < 1e-3 {
                return p;
            }
            let grow = (p[0] / bw).clamp(0.0, 1.0);
            [
                p[0],
                p[1] + amplitude * grow * (2.0 * PI * p[0] / wavelength).sin(),
            ]
        }
        TextWarp::Bulge { amount } => {
            let d = ((p[0] - cx) / cx.max(1e-3)).clamp(-1.0, 1.0);
            let bump = 0.5 * (1.0 + (PI * d).cos());
            let k = (1.0 + amount * bump).max(0.05);
            [p[0], cy + (p[1] - cy) * k]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arc_keeps_centre_and_bends() {
        let w = TextWarp::Arc { bend: 0.5 };
        let c = warp_point(&w, [200.0, 50.0], [100.0, 25.0]);
        assert!((c[0] - 100.0).abs() < 1e-3 && (c[1] - 25.0).abs() < 1e-3);
        let e = warp_point(&w, [200.0, 50.0], [0.0, 25.0]);
        assert!(e[1] > 30.0, "ends drop for a positive bend: {e:?}");
    }

    #[test]
    fn square_area_sign() {
        assert!(signed_area(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) > 0.0);
    }
}
