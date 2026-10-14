use crate::math::Mat3;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

pub const D65: [f64; 2] = [0.3127, 0.3290];
pub const D50: [f64; 2] = [0.3457, 0.3585];

/// Transfer (EOTF inverse / OETF) functions.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Transfer {
    Linear,
    Srgb,
    /// Pure power function (e.g. 2.2 for Adobe RGB, 1.8 for ProPhoto).
    Gamma(f32),
    Rec709,
    /// SMPTE ST 2084, 1.0 linear == 203 nits (reference white) here.
    Pq,
    Hlg,
}

impl Transfer {
    /// Encoded -> linear.
    #[inline]
    pub fn to_linear(self, v: f32) -> f32 {
        match self {
            Transfer::Linear => v,
            Transfer::Srgb => srgb_to_linear(v),
            Transfer::Gamma(g) => v.signum() * v.abs().powf(g),
            Transfer::Rec709 => {
                let a = v.abs();
                let l = if a < 0.081 { a / 4.5 } else { ((a + 0.099) / 1.099).powf(1.0 / 0.45) };
                l.copysign(v)
            }
            Transfer::Pq => pq_to_linear(v) * (10000.0 / 203.0),
            Transfer::Hlg => hlg_to_linear(v),
        }
    }

    /// Linear -> encoded.
    #[inline]
    pub fn from_linear(self, v: f32) -> f32 {
        match self {
            Transfer::Linear => v,
            Transfer::Srgb => linear_to_srgb(v),
            Transfer::Gamma(g) => v.signum() * v.abs().powf(1.0 / g),
            Transfer::Rec709 => {
                let a = v.abs();
                let e = if a < 0.018 { a * 4.5 } else { 1.099 * a.powf(0.45) - 0.099 };
                e.copysign(v)
            }
            Transfer::Pq => linear_to_pq(v * (203.0 / 10000.0)),
            Transfer::Hlg => linear_to_hlg(v),
        }
    }
}

#[inline]
pub fn srgb_to_linear(v: f32) -> f32 {
    let a = v.abs();
    let l = if a <= 0.04045 { a / 12.92 } else { ((a + 0.055) / 1.055).powf(2.4) };
    l.copysign(v)
}

#[inline]
pub fn linear_to_srgb(v: f32) -> f32 {
    let a = v.abs();
    let e = if a <= 0.0031308 { a * 12.92 } else { 1.055 * a.powf(1.0 / 2.4) - 0.055 };
    e.copysign(v)
}

fn pq_to_linear(e: f32) -> f32 {
    let (m1, m2, c1, c2, c3) = (0.159_301_76_f32, 78.84375f32, 0.8359375f32, 18.851_563_f32, 18.6875f32);
    let p = e.max(0.0).powf(1.0 / m2);
    ((p - c1).max(0.0) / (c2 - c3 * p)).powf(1.0 / m1)
}

fn linear_to_pq(l: f32) -> f32 {
    let (m1, m2, c1, c2, c3) = (0.159_301_76_f32, 78.84375f32, 0.8359375f32, 18.851_563_f32, 18.6875f32);
    let y = l.max(0.0).powf(m1);
    ((c1 + c2 * y) / (1.0 + c3 * y)).powf(m2)
}

fn hlg_to_linear(e: f32) -> f32 {
    let (a, b, c) = (0.17883277f32, 0.28466892f32, 0.559_910_7_f32);
    if e <= 0.5 { e * e / 3.0 } else { (((e - c) / a).exp() + b) / 12.0 }
}

fn linear_to_hlg(l: f32) -> f32 {
    let (a, b, c) = (0.17883277f32, 0.28466892f32, 0.559_910_7_f32);
    let l = l.max(0.0);
    if l <= 1.0 / 12.0 { (3.0 * l).sqrt() } else { a * (12.0 * l - b).ln() + c }
}

/// Well known RGB colour spaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum RgbSpace {
    #[default]
    Srgb,
    LinearSrgb,
    DisplayP3,
    Rec2020,
    LinearRec2020,
    AdobeRgb,
    ProPhoto,
    Rec2100Pq,
    Rec2100Hlg,
}

impl RgbSpace {
    pub const ALL: [RgbSpace; 9] = [
        RgbSpace::Srgb,
        RgbSpace::LinearSrgb,
        RgbSpace::DisplayP3,
        RgbSpace::Rec2020,
        RgbSpace::LinearRec2020,
        RgbSpace::AdobeRgb,
        RgbSpace::ProPhoto,
        RgbSpace::Rec2100Pq,
        RgbSpace::Rec2100Hlg,
    ];

    pub fn name(self) -> &'static str {
        match self {
            RgbSpace::Srgb => "sRGB",
            RgbSpace::LinearSrgb => "Linear sRGB",
            RgbSpace::DisplayP3 => "Display P3",
            RgbSpace::Rec2020 => "Rec.2020",
            RgbSpace::LinearRec2020 => "Linear Rec.2020",
            RgbSpace::AdobeRgb => "Adobe RGB (1998)",
            RgbSpace::ProPhoto => "ProPhoto RGB",
            RgbSpace::Rec2100Pq => "Rec.2100 PQ",
            RgbSpace::Rec2100Hlg => "Rec.2100 HLG",
        }
    }

    pub fn transfer(self) -> Transfer {
        match self {
            RgbSpace::Srgb | RgbSpace::DisplayP3 => Transfer::Srgb,
            RgbSpace::LinearSrgb | RgbSpace::LinearRec2020 => Transfer::Linear,
            RgbSpace::Rec2020 => Transfer::Rec709,
            RgbSpace::AdobeRgb => Transfer::Gamma(563.0 / 256.0),
            RgbSpace::ProPhoto => Transfer::Gamma(1.8),
            RgbSpace::Rec2100Pq => Transfer::Pq,
            RgbSpace::Rec2100Hlg => Transfer::Hlg,
        }
    }

    /// (red, green, blue, white) xy chromaticities.
    pub fn primaries(self) -> ([f64; 2], [f64; 2], [f64; 2], [f64; 2]) {
        match self {
            RgbSpace::Srgb | RgbSpace::LinearSrgb => ([0.64, 0.33], [0.30, 0.60], [0.15, 0.06], D65),
            RgbSpace::DisplayP3 => ([0.680, 0.320], [0.265, 0.690], [0.150, 0.060], D65),
            RgbSpace::Rec2020 | RgbSpace::LinearRec2020 | RgbSpace::Rec2100Pq | RgbSpace::Rec2100Hlg => {
                ([0.708, 0.292], [0.170, 0.797], [0.131, 0.046], D65)
            }
            RgbSpace::AdobeRgb => ([0.64, 0.33], [0.21, 0.71], [0.15, 0.06], D65),
            RgbSpace::ProPhoto => ([0.7347, 0.2653], [0.1596, 0.8404], [0.0366, 0.0001], D50),
        }
    }

    /// Linear RGB -> XYZ (D65, adapted with Bradford when needed).
    pub fn to_xyz_d65(self) -> Mat3 {
        let (r, g, b, w) = self.primaries();
        let m = Mat3::rgb_to_xyz(r, g, b, w);
        if w == D65 { m } else { Mat3::bradford(w, D65).mul(&m) }
    }

    /// Linear RGB in `self` -> linear working space (Rec.2020).
    pub fn to_working_matrix(self) -> Mat3 {
        working_from_xyz().mul(&self.to_xyz_d65())
    }

    pub fn from_working_matrix(self) -> Mat3 {
        self.to_working_matrix().inverse()
    }
}

fn working_from_xyz() -> &'static Mat3 {
    static M: OnceLock<Mat3> = OnceLock::new();
    M.get_or_init(|| RgbSpace::LinearRec2020.to_xyz_d65().inverse())
}

pub fn working_to_xyz() -> &'static Mat3 {
    static M: OnceLock<Mat3> = OnceLock::new();
    M.get_or_init(|| RgbSpace::LinearRec2020.to_xyz_d65())
}

/// Precomputed analytic converter between an encoded RGB space and the working space.
#[derive(Clone, Copy, Debug)]
pub struct SpaceConverter {
    pub space: RgbSpace,
    to_working: Mat3,
    from_working: Mat3,
}

impl SpaceConverter {
    pub fn new(space: RgbSpace) -> Self {
        Self { space, to_working: space.to_working_matrix(), from_working: space.from_working_matrix() }
    }

    /// Encoded RGB -> linear working RGB.
    #[inline]
    pub fn to_working(&self, rgb: [f32; 3]) -> [f32; 3] {
        let t = self.space.transfer();
        self.to_working.apply([t.to_linear(rgb[0]), t.to_linear(rgb[1]), t.to_linear(rgb[2])])
    }

    /// Linear working RGB -> encoded RGB (unclamped).
    #[inline]
    pub fn from_working(&self, rgb: [f32; 3]) -> [f32; 3] {
        let t = self.space.transfer();
        let l = self.from_working.apply(rgb);
        [t.from_linear(l[0]), t.from_linear(l[1]), t.from_linear(l[2])]
    }

    pub fn matrix_to_working(&self) -> &Mat3 {
        &self.to_working
    }

    pub fn matrix_from_working(&self) -> &Mat3 {
        &self.from_working
    }
}

/// Shared sRGB converter (used heavily by the UI).
pub fn srgb() -> &'static SpaceConverter {
    static C: OnceLock<SpaceConverter> = OnceLock::new();
    C.get_or_init(|| SpaceConverter::new(RgbSpace::Srgb))
}

// --------------------------------------------------------------------------------
// OKLab / OKLCh (defined from linear sRGB in the reference implementation; we go via
// XYZ so it works from the Rec.2020 working space).
// --------------------------------------------------------------------------------

const XYZ_TO_LMS: Mat3 = Mat3([
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715, 0.0361456387],
    [0.0482003018, 0.2643662691, 0.6338517070],
]);
const LMS_TO_LAB: Mat3 = Mat3([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
]);

fn working_to_lms() -> &'static Mat3 {
    static M: OnceLock<Mat3> = OnceLock::new();
    M.get_or_init(|| XYZ_TO_LMS.mul(working_to_xyz()))
}

fn lms_to_working() -> &'static Mat3 {
    static M: OnceLock<Mat3> = OnceLock::new();
    M.get_or_init(|| working_to_lms().inverse())
}

fn lab_to_lms() -> &'static Mat3 {
    static M: OnceLock<Mat3> = OnceLock::new();
    M.get_or_init(|| LMS_TO_LAB.inverse())
}

/// Linear working RGB -> OKLab.
pub fn working_to_oklab(rgb: [f32; 3]) -> [f32; 3] {
    let lms = working_to_lms().apply(rgb);
    let c = lms.map(|v| v.cbrt());
    LMS_TO_LAB.apply(c)
}

/// OKLab -> linear working RGB.
pub fn oklab_to_working(lab: [f32; 3]) -> [f32; 3] {
    let c = lab_to_lms().apply(lab);
    lms_to_working().apply(c.map(|v| v * v * v))
}

/// OKLab -> OKLCh (h in radians).
pub fn oklab_to_oklch(lab: [f32; 3]) -> [f32; 3] {
    [lab[0], (lab[1] * lab[1] + lab[2] * lab[2]).sqrt(), lab[2].atan2(lab[1])]
}

pub fn oklch_to_oklab(lch: [f32; 3]) -> [f32; 3] {
    [lch[0], lch[1] * lch[2].cos(), lch[1] * lch[2].sin()]
}

/// The 3x3 matrices needed by GPU shaders to do OKLab conversions from working RGB.
pub fn oklab_gpu_matrices() -> (Mat3, Mat3, Mat3, Mat3) {
    (*working_to_lms(), LMS_TO_LAB, *lab_to_lms(), *lms_to_working())
}

// --------------------------------------------------------------------------------
// HSV helpers (on encoded display RGB, for UI pickers).
// --------------------------------------------------------------------------------

pub fn rgb_to_hsv(rgb: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = rgb;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let h = if d <= 0.0 {
        0.0
    } else if max == r {
        ((g - b) / d).rem_euclid(6.0)
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    } / 6.0;
    let s = if max <= 0.0 { 0.0 } else { d / max };
    [h, s, max]
}

pub fn hsv_to_rgb(hsv: [f32; 3]) -> [f32; 3] {
    let [h, s, v] = hsv;
    let h6 = h.rem_euclid(1.0) * 6.0;
    let i = h6.floor();
    let f = h6 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i as i32 % 6 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// Relative luminance (Y) of linear working RGB.
#[inline]
pub fn working_luminance(rgb: [f32; 3]) -> f32 {
    // Rec.2020 luma coefficients
    0.2627 * rgb[0] + 0.6780 * rgb[1] + 0.0593 * rgb[2]
}

/// Convert an 8-bit sRGB hex like "#ff8800" or "ff8800cc" to linear working RGBA.
pub fn hex_to_working(hex: &str) -> Option<[f32; 4]> {
    let h = hex.trim().trim_start_matches('#');
    let p = |i: usize| u8::from_str_radix(h.get(i..i + 2)?, 16).ok();
    let (r, g, b) = (p(0)?, p(2)?, p(4)?);
    let a = if h.len() >= 8 { p(6)? } else { 255 };
    let w = srgb().to_working([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]);
    Some([w[0], w[1], w[2], a as f32 / 255.0])
}

pub fn working_to_hex(c: [f32; 4]) -> String {
    let s = srgb().from_working([c[0], c[1], c[2]]);
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    if c[3] < 1.0 {
        format!("#{:02x}{:02x}{:02x}{:02x}", q(s[0]), q(s[1]), q(s[2]), q(c[3]))
    } else {
        format!("#{:02x}{:02x}{:02x}", q(s[0]), q(s[1]), q(s[2]))
    }
}

// --------------------------------------------------------------------------------
// CIELAB and ΔE2000 (for accuracy testing and colour-select tolerance).
// --------------------------------------------------------------------------------

/// Linear working RGB -> CIELAB (D65 reference white).
pub fn working_to_lab(rgb: [f32; 3]) -> [f32; 3] {
    let xyz = working_to_xyz().apply(rgb);
    let wn = [0.95047f32, 1.0, 1.08883];
    let f = |t: f32| if t > 216.0 / 24389.0 { t.cbrt() } else { (24389.0 / 27.0 * t + 16.0) / 116.0 };
    let (fx, fy, fz) = (f(xyz[0] / wn[0]), f(xyz[1] / wn[1]), f(xyz[2] / wn[2]));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

pub fn delta_e_2000(l1: [f32; 3], l2: [f32; 3]) -> f32 {
    let (l1, a1, b1) = (l1[0] as f64, l1[1] as f64, l1[2] as f64);
    let (l2, a2, b2) = (l2[0] as f64, l2[1] as f64, l2[2] as f64);
    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let cb = (c1 + c2) / 2.0;
    let g = 0.5 * (1.0 - (cb.powi(7) / (cb.powi(7) + 25f64.powi(7))).sqrt());
    let a1p = (1.0 + g) * a1;
    let a2p = (1.0 + g) * a2;
    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();
    let h = |b: f64, a: f64| {
        if a == 0.0 && b == 0.0 {
            0.0
        } else {
            let h = b.atan2(a).to_degrees();
            if h < 0.0 { h + 360.0 } else { h }
        }
    };
    let h1p = h(b1, a1p);
    let h2p = h(b2, a2p);
    let dl = l2 - l1;
    let dc = c2p - c1p;
    let dhp = if c1p * c2p == 0.0 {
        0.0
    } else if (h2p - h1p).abs() <= 180.0 {
        h2p - h1p
    } else if h2p - h1p > 180.0 {
        h2p - h1p - 360.0
    } else {
        h2p - h1p + 360.0
    };
    let dh = 2.0 * (c1p * c2p).sqrt() * (dhp.to_radians() / 2.0).sin();
    let lbp = (l1 + l2) / 2.0;
    let cbp = (c1p + c2p) / 2.0;
    let hbp = if c1p * c2p == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) / 2.0
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) / 2.0
    } else {
        (h1p + h2p - 360.0) / 2.0
    };
    let t = 1.0 - 0.17 * (hbp - 30.0).to_radians().cos() + 0.24 * (2.0 * hbp).to_radians().cos()
        + 0.32 * (3.0 * hbp + 6.0).to_radians().cos()
        - 0.20 * (4.0 * hbp - 63.0).to_radians().cos();
    let dtheta = 30.0 * (-((hbp - 275.0) / 25.0).powi(2)).exp();
    let rc = 2.0 * (cbp.powi(7) / (cbp.powi(7) + 25f64.powi(7))).sqrt();
    let sl = 1.0 + 0.015 * (lbp - 50.0).powi(2) / (20.0 + (lbp - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * cbp;
    let sh = 1.0 + 0.015 * cbp * t;
    let rt = -(2.0 * dtheta.to_radians()).sin() * rc;
    (((dl / sl).powi(2) + (dc / sc).powi(2) + (dh / sh).powi(2) + rt * (dc / sc) * (dh / sh)).sqrt()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_roundtrip_through_working() {
        let c = srgb();
        for v in [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.2, 0.5, 0.9], [1.0, 0.0, 0.0]] {
            let w = c.to_working(v);
            let b = c.from_working(w);
            for i in 0..3 {
                assert!((v[i] - b[i]).abs() < 1e-5, "{v:?} -> {b:?}");
            }
        }
    }

    #[test]
    fn white_maps_to_white() {
        let w = srgb().to_working([1.0, 1.0, 1.0]);
        for v in w {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn oklab_white_is_l1() {
        let lab = working_to_oklab([1.0, 1.0, 1.0]);
        assert!((lab[0] - 1.0).abs() < 1e-3 && lab[1].abs() < 1e-3 && lab[2].abs() < 1e-3, "{lab:?}");
        let back = oklab_to_working(lab);
        assert!((back[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn de2000_reference_pair() {
        // Sharma et al. test pair #1
        let d = delta_e_2000([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485]);
        assert!((d - 2.0425).abs() < 1e-3, "{d}");
    }

    #[test]
    fn hsv_roundtrip() {
        let c = [0.3, 0.7, 0.2];
        let b = hsv_to_rgb(rgb_to_hsv(c));
        for i in 0..3 {
            assert!((c[i] - b[i]).abs() < 1e-5);
        }
    }
}
