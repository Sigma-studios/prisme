/// Row-major 3x3 matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3(pub [[f64; 3]; 3]);

impl Mat3 {
    pub const IDENTITY: Mat3 = Mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

    pub fn mul(&self, o: &Mat3) -> Mat3 {
        let mut r = [[0.0; 3]; 3];
        for (i, row) in r.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = (0..3).map(|k| self.0[i][k] * o.0[k][j]).sum();
            }
        }
        Mat3(r)
    }

    #[inline]
    pub fn apply(&self, v: [f32; 3]) -> [f32; 3] {
        let m = &self.0;
        let (x, y, z) = (v[0] as f64, v[1] as f64, v[2] as f64);
        [
            (m[0][0] * x + m[0][1] * y + m[0][2] * z) as f32,
            (m[1][0] * x + m[1][1] * y + m[1][2] * z) as f32,
            (m[2][0] * x + m[2][1] * y + m[2][2] * z) as f32,
        ]
    }

    pub fn inverse(&self) -> Mat3 {
        let m = &self.0;
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let i = 1.0 / det;
        Mat3([
            [
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * i,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * i,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * i,
            ],
            [
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * i,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * i,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * i,
            ],
            [
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * i,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * i,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * i,
            ],
        ])
    }

    /// Column-major f32 array padded to 3x vec4 (WGSL `mat3x3<f32>` uniform layout).
    pub fn to_wgsl_padded(&self) -> [f32; 12] {
        let m = &self.0;
        [
            m[0][0] as f32, m[1][0] as f32, m[2][0] as f32, 0.0,
            m[0][1] as f32, m[1][1] as f32, m[2][1] as f32, 0.0,
            m[0][2] as f32, m[1][2] as f32, m[2][2] as f32, 0.0,
        ]
    }

    /// RGB -> XYZ matrix from xy chromaticities of R, G, B and white.
    pub fn rgb_to_xyz(r: [f64; 2], g: [f64; 2], b: [f64; 2], w: [f64; 2]) -> Mat3 {
        let xyz = |p: [f64; 2]| [p[0] / p[1], 1.0, (1.0 - p[0] - p[1]) / p[1]];
        let (xr, xg, xb, xw) = (xyz(r), xyz(g), xyz(b), xyz(w));
        let m = Mat3([[xr[0], xg[0], xb[0]], [xr[1], xg[1], xb[1]], [xr[2], xg[2], xb[2]]]);
        let inv = m.inverse();
        let s = [
            inv.0[0][0] * xw[0] + inv.0[0][1] * xw[1] + inv.0[0][2] * xw[2],
            inv.0[1][0] * xw[0] + inv.0[1][1] * xw[1] + inv.0[1][2] * xw[2],
            inv.0[2][0] * xw[0] + inv.0[2][1] * xw[1] + inv.0[2][2] * xw[2],
        ];
        Mat3([
            [s[0] * xr[0], s[1] * xg[0], s[2] * xb[0]],
            [s[0] * xr[1], s[1] * xg[1], s[2] * xb[1]],
            [s[0] * xr[2], s[1] * xg[2], s[2] * xb[2]],
        ])
    }

    /// Bradford chromatic adaptation from white `src` to white `dst` (xy).
    pub fn bradford(src: [f64; 2], dst: [f64; 2]) -> Mat3 {
        let b = Mat3([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]]);
        let xyz = |p: [f64; 2]| [p[0] / p[1], 1.0, (1.0 - p[0] - p[1]) / p[1]];
        let lms = |p: [f64; 3]| {
            let m = &b.0;
            [
                m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
                m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
                m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2],
            ]
        };
        let s = lms(xyz(src));
        let d = lms(xyz(dst));
        let diag = Mat3([[d[0] / s[0], 0.0, 0.0], [0.0, d[1] / s[1], 0.0], [0.0, 0.0, d[2] / s[2]]]);
        b.inverse().mul(&diag).mul(&b)
    }
}
