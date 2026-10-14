use serde::{Deserialize, Serialize};

/// Half-open integer rectangle `[x0, x1) x [y0, y1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct IRect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl IRect {
    pub const EMPTY: IRect = IRect { x0: 0, y0: 0, x1: 0, y1: 0 };

    pub fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        Self { x0, y0, x1, y1 }
    }

    pub fn from_size(w: u32, h: u32) -> Self {
        Self::new(0, 0, w as i32, h as i32)
    }

    pub fn width(&self) -> i32 {
        (self.x1 - self.x0).max(0)
    }

    pub fn height(&self) -> i32 {
        (self.y1 - self.y0).max(0)
    }

    pub fn is_empty(&self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }

    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x0 && x < self.x1 && y >= self.y0 && y < self.y1
    }

    pub fn intersect(&self, o: &IRect) -> IRect {
        let r = IRect::new(self.x0.max(o.x0), self.y0.max(o.y0), self.x1.min(o.x1), self.y1.min(o.y1));
        if r.is_empty() { IRect::EMPTY } else { r }
    }

    pub fn union(&self, o: &IRect) -> IRect {
        if self.is_empty() {
            return *o;
        }
        if o.is_empty() {
            return *self;
        }
        IRect::new(self.x0.min(o.x0), self.y0.min(o.y0), self.x1.max(o.x1), self.y1.max(o.y1))
    }

    pub fn expand(&self, by: i32) -> IRect {
        IRect::new(self.x0 - by, self.y0 - by, self.x1 + by, self.y1 + by)
    }

    pub fn translate(&self, dx: i32, dy: i32) -> IRect {
        IRect::new(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)
    }

    /// Smallest integer rect covering the float rect.
    pub fn from_f32(x0: f32, y0: f32, x1: f32, y1: f32) -> IRect {
        IRect::new(x0.floor() as i32, y0.floor() as i32, x1.ceil() as i32, y1.ceil() as i32)
    }

    pub fn area(&self) -> i64 {
        self.width() as i64 * self.height() as i64
    }
}

/// 2D affine transform stored as a 2x3 row-major matrix: `[a b tx; c d ty]`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Affine2 {
    pub m: [f32; 6],
}

impl Default for Affine2 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Affine2 {
    pub const IDENTITY: Affine2 = Affine2 { m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] };

    pub fn translate(x: f32, y: f32) -> Self {
        Affine2 { m: [1.0, 0.0, x, 0.0, 1.0, y] }
    }

    pub fn scale(sx: f32, sy: f32) -> Self {
        Affine2 { m: [sx, 0.0, 0.0, 0.0, sy, 0.0] }
    }

    pub fn rotate(rad: f32) -> Self {
        let (s, c) = rad.sin_cos();
        Affine2 { m: [c, -s, 0.0, s, c, 0.0] }
    }

    /// `self * other` (apply `other` first).
    pub fn mul(&self, o: &Affine2) -> Affine2 {
        let a = &self.m;
        let b = &o.m;
        Affine2 {
            m: [
                a[0] * b[0] + a[1] * b[3],
                a[0] * b[1] + a[1] * b[4],
                a[0] * b[2] + a[1] * b[5] + a[2],
                a[3] * b[0] + a[4] * b[3],
                a[3] * b[1] + a[4] * b[4],
                a[3] * b[2] + a[4] * b[5] + a[5],
            ],
        }
    }

    pub fn apply(&self, p: [f32; 2]) -> [f32; 2] {
        let m = &self.m;
        [m[0] * p[0] + m[1] * p[1] + m[2], m[3] * p[0] + m[4] * p[1] + m[5]]
    }

    pub fn inverse(&self) -> Option<Affine2> {
        let m = &self.m;
        let det = m[0] * m[4] - m[1] * m[3];
        if det.abs() < 1e-12 {
            return None;
        }
        let id = 1.0 / det;
        let a = m[4] * id;
        let b = -m[1] * id;
        let c = -m[3] * id;
        let d = m[0] * id;
        Some(Affine2 { m: [a, b, -(a * m[2] + b * m[5]), c, d, -(c * m[2] + d * m[5])] })
    }

    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }
}

/// 3x3 projective transform (row-major), used by perspective free-transform.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Homography {
    pub m: [f32; 9],
}

impl Homography {
    pub const IDENTITY: Homography = Homography { m: [1., 0., 0., 0., 1., 0., 0., 0., 1.] };

    pub fn apply(&self, p: [f32; 2]) -> [f32; 2] {
        let m = &self.m;
        let w = m[6] * p[0] + m[7] * p[1] + m[8];
        [(m[0] * p[0] + m[1] * p[1] + m[2]) / w, (m[3] * p[0] + m[4] * p[1] + m[5]) / w]
    }

    pub fn inverse(&self) -> Option<Homography> {
        let m: [f64; 9] = self.m.map(|v| v as f64);
        let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6]);
        if det.abs() < 1e-14 {
            return None;
        }
        let i = 1.0 / det;
        let r = [
            (m[4] * m[8] - m[5] * m[7]) * i,
            (m[2] * m[7] - m[1] * m[8]) * i,
            (m[1] * m[5] - m[2] * m[4]) * i,
            (m[5] * m[6] - m[3] * m[8]) * i,
            (m[0] * m[8] - m[2] * m[6]) * i,
            (m[2] * m[3] - m[0] * m[5]) * i,
            (m[3] * m[7] - m[4] * m[6]) * i,
            (m[1] * m[6] - m[0] * m[7]) * i,
            (m[0] * m[4] - m[1] * m[3]) * i,
        ];
        Some(Homography { m: r.map(|v| v as f32) })
    }

    /// Homography mapping the unit square corners (0,0),(1,0),(1,1),(0,1) to `q`.
    pub fn square_to_quad(q: [[f32; 2]; 4]) -> Homography {
        let [p0, p1, p2, p3] = q.map(|p| [p[0] as f64, p[1] as f64]);
        let dx1 = p1[0] - p2[0];
        let dx2 = p3[0] - p2[0];
        let dy1 = p1[1] - p2[1];
        let dy2 = p3[1] - p2[1];
        let sx = p0[0] - p1[0] + p2[0] - p3[0];
        let sy = p0[1] - p1[1] + p2[1] - p3[1];
        let (g, h) = if sx.abs() < 1e-12 && sy.abs() < 1e-12 {
            (0.0, 0.0)
        } else {
            let den = dx1 * dy2 - dx2 * dy1;
            ((sx * dy2 - dx2 * sy) / den, (dx1 * sy - sx * dy1) / den)
        };
        let a = p1[0] - p0[0] + g * p1[0];
        let b = p3[0] - p0[0] + h * p3[0];
        let c = p0[0];
        let d = p1[1] - p0[1] + g * p1[1];
        let e = p3[1] - p0[1] + h * p3[1];
        let f = p0[1];
        Homography { m: [a, b, c, d, e, f, g, h, 1.0].map(|v| v as f32) }
    }

    /// Homography mapping rect `r` onto quad `q` (corners in TL, TR, BR, BL order).
    pub fn rect_to_quad(r: [f32; 4], q: [[f32; 2]; 4]) -> Homography {
        let s2q = Self::square_to_quad(q);
        let (w, h) = (r[2] - r[0], r[3] - r[1]);
        // rect -> unit square
        let r2s = Homography { m: [1.0 / w, 0.0, -r[0] / w, 0.0, 1.0 / h, -r[1] / h, 0.0, 0.0, 1.0] };
        s2q.mul(&r2s)
    }

    pub fn mul(&self, o: &Homography) -> Homography {
        let a = &self.m;
        let b = &o.m;
        let mut r = [0.0f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
            }
        }
        Homography { m: r }
    }

    pub fn from_affine(a: &Affine2) -> Homography {
        let m = a.m;
        Homography { m: [m[0], m[1], m[2], m[3], m[4], m[5], 0.0, 0.0, 1.0] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_inverse_roundtrip() {
        let a = Affine2::translate(5.0, -3.0).mul(&Affine2::rotate(0.7)).mul(&Affine2::scale(2.0, 0.5));
        let inv = a.inverse().unwrap();
        let p = [3.0, 4.0];
        let q = inv.apply(a.apply(p));
        assert!((p[0] - q[0]).abs() < 1e-4 && (p[1] - q[1]).abs() < 1e-4);
    }

    #[test]
    fn quad_mapping_hits_corners() {
        let q = [[10.0, 10.0], [100.0, 20.0], [90.0, 120.0], [5.0, 80.0]];
        let h = Homography::rect_to_quad([0.0, 0.0, 50.0, 40.0], q);
        let c = [[0.0, 0.0], [50.0, 0.0], [50.0, 40.0], [0.0, 40.0]];
        for i in 0..4 {
            let p = h.apply(c[i]);
            assert!((p[0] - q[i][0]).abs() < 1e-2 && (p[1] - q[i][1]).abs() < 1e-2, "{p:?} vs {:?}", q[i]);
        }
    }
}
