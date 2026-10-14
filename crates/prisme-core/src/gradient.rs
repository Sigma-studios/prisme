use prisme_color::{linear_to_srgb, oklab_to_working, srgb_to_linear, working_to_oklab};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum GradientKind {
    #[default]
    Linear,
    Radial,
    Angular,
    Reflected,
    Diamond,
}

impl GradientKind {
    pub const ALL: [GradientKind; 5] =
        [GradientKind::Linear, GradientKind::Radial, GradientKind::Angular, GradientKind::Reflected, GradientKind::Diamond];

    pub fn name(self) -> &'static str {
        match self {
            GradientKind::Linear => "Linear",
            GradientKind::Radial => "Radial",
            GradientKind::Angular => "Angular",
            GradientKind::Reflected => "Reflected",
            GradientKind::Diamond => "Diamond",
        }
    }
}

/// Colour space used to interpolate between stops.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum GradientInterp {
    #[default]
    OkLab,
    Linear,
    Srgb,
}

impl GradientInterp {
    pub const ALL: [GradientInterp; 3] = [GradientInterp::OkLab, GradientInterp::Linear, GradientInterp::Srgb];

    pub fn name(self) -> &'static str {
        match self {
            GradientInterp::OkLab => "OKLab (perceptual)",
            GradientInterp::Linear => "Linear light",
            GradientInterp::Srgb => "sRGB (classic)",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GradientStop {
    pub pos: f32,
    /// Linear working RGBA, straight alpha.
    pub color: [f32; 4],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gradient {
    pub stops: Vec<GradientStop>,
    pub interp: GradientInterp,
}

impl Default for Gradient {
    fn default() -> Self {
        Self::two([0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0])
    }
}

impl Gradient {
    pub fn two(a: [f32; 4], b: [f32; 4]) -> Self {
        Gradient {
            stops: vec![GradientStop { pos: 0.0, color: a }, GradientStop { pos: 1.0, color: b }],
            interp: GradientInterp::OkLab,
        }
    }

    pub fn sorted_stops(&self) -> Vec<GradientStop> {
        let mut s = self.stops.clone();
        s.sort_by(|a, b| a.pos.total_cmp(&b.pos));
        s
    }

    /// Evaluate at `t` in [0,1]. Returns linear working RGBA (straight alpha).
    pub fn eval(&self, t: f32) -> [f32; 4] {
        let s = self.sorted_stops();
        if s.is_empty() {
            return [0.0; 4];
        }
        let t = t.clamp(0.0, 1.0);
        if t <= s[0].pos {
            return s[0].color;
        }
        for w in s.windows(2) {
            if t <= w[1].pos {
                let span = (w[1].pos - w[0].pos).max(1e-6);
                return self.mix(w[0].color, w[1].color, (t - w[0].pos) / span);
            }
        }
        s[s.len() - 1].color
    }

    fn mix(&self, a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
        let alpha = a[3] + (b[3] - a[3]) * t;
        // Interpolate colour premultiplied so transparent stops don't darken.
        let lerp3 = |x: [f32; 3], y: [f32; 3]| -> [f32; 3] {
            let pa = a[3] * (1.0 - t);
            let pb = b[3] * t;
            let w = pa + pb;
            if w <= 1e-8 {
                return [x[0] + (y[0] - x[0]) * t, x[1] + (y[1] - x[1]) * t, x[2] + (y[2] - x[2]) * t];
            }
            [(x[0] * pa + y[0] * pb) / w, (x[1] * pa + y[1] * pb) / w, (x[2] * pa + y[2] * pb) / w]
        };
        let (ca, cb) = ([a[0], a[1], a[2]], [b[0], b[1], b[2]]);
        let rgb = match self.interp {
            GradientInterp::Linear => lerp3(ca, cb),
            GradientInterp::OkLab => oklab_to_working(lerp3(working_to_oklab(ca), working_to_oklab(cb))),
            GradientInterp::Srgb => {
                let e = |c: [f32; 3]| c.map(linear_to_srgb);
                lerp3(e(ca), e(cb)).map(srgb_to_linear)
            }
        };
        [rgb[0], rgb[1], rgb[2], alpha]
    }

    /// Sample into an `n`-entry lookup table (used by GPU kernels and fast CPU paths).
    pub fn to_lut(&self, n: usize) -> Vec<[f32; 4]> {
        (0..n).map(|i| self.eval(i as f32 / (n - 1) as f32)).collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum GradientRepeat {
    #[default]
    Pad,
    Repeat,
    Mirror,
}

/// Placement of a gradient in document space.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GradientGeometry {
    pub kind: GradientKind,
    pub start: [f32; 2],
    pub end: [f32; 2],
    pub repeat: GradientRepeat,
    pub reverse: bool,
    pub dither: bool,
}

impl Default for GradientGeometry {
    fn default() -> Self {
        Self {
            kind: GradientKind::Linear,
            start: [0.0, 0.0],
            end: [100.0, 0.0],
            repeat: GradientRepeat::Pad,
            reverse: false,
            dither: true,
        }
    }
}

impl GradientGeometry {
    /// Gradient parameter at document position `p`.
    pub fn t_at(&self, p: [f32; 2]) -> f32 {
        let d = [self.end[0] - self.start[0], self.end[1] - self.start[1]];
        let len2 = (d[0] * d[0] + d[1] * d[1]).max(1e-6);
        let len = len2.sqrt();
        let v = [p[0] - self.start[0], p[1] - self.start[1]];
        let mut t = match self.kind {
            GradientKind::Linear => (v[0] * d[0] + v[1] * d[1]) / len2,
            GradientKind::Reflected => ((v[0] * d[0] + v[1] * d[1]) / len2).abs(),
            GradientKind::Radial => (v[0] * v[0] + v[1] * v[1]).sqrt() / len,
            GradientKind::Diamond => {
                // rotate into gradient frame then L1 norm
                let (ux, uy) = (d[0] / len, d[1] / len);
                let a = (v[0] * ux + v[1] * uy).abs();
                let b = (-v[0] * uy + v[1] * ux).abs();
                (a + b) / len
            }
            GradientKind::Angular => {
                let a0 = d[1].atan2(d[0]);
                let a = v[1].atan2(v[0]);
                (a - a0).rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU
            }
        };
        t = match self.repeat {
            GradientRepeat::Pad => t.clamp(0.0, 1.0),
            GradientRepeat::Repeat => t.rem_euclid(1.0),
            GradientRepeat::Mirror => {
                let m = t.rem_euclid(2.0);
                if m > 1.0 { 2.0 - m } else { m }
            }
        };
        if self.reverse { 1.0 - t } else { t }
    }
}

/// Tiny ordered-dither offset (blue-noise-ish interleaved gradient noise) in [-0.5, 0.5].
pub fn dither_noise(x: i32, y: i32) -> f32 {
    let f = 52.982_918 * ((0.067_110_56 * x as f32 + 0.005_837_15 * y as f32).fract());
    f.fract() - 0.5
}
