use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum TipShape {
    #[default]
    Round,
    Square,
    /// Round tip with paper grain (chalk / pastel).
    Chalk,
    /// Grainy, streaky round tip.
    Charcoal,
    /// Cluster of small random droplets.
    Splatter,
}

impl TipShape {
    pub const ALL: [TipShape; 5] = [TipShape::Round, TipShape::Square, TipShape::Chalk, TipShape::Charcoal, TipShape::Splatter];

    pub fn name(self) -> &'static str {
        match self {
            TipShape::Round => "Round",
            TipShape::Square => "Square",
            TipShape::Chalk => "Chalk",
            TipShape::Charcoal => "Charcoal",
            TipShape::Splatter => "Splatter",
        }
    }
}

/// A single stamp of the brush tip.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dab {
    pub center: [f32; 2],
    pub radius: f32,
    pub angle: f32,
    pub roundness: f32,
    pub hardness: f32,
    /// Dab opacity (flow × dynamics).
    pub alpha: f32,
    pub tip: TipShape,
    pub pixel: bool,
    pub grain: f32,
    pub seed: u32,
}

#[inline]
fn hash2(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    (h & 0xffff) as f32 / 65535.0
}

/// Smooth value noise anchored to document pixels (paper texture).
fn value_noise(x: f32, y: f32, scale: f32, seed: u32) -> f32 {
    let fx = x / scale;
    let fy = y / scale;
    let ix = fx.floor() as i32;
    let iy = fy.floor() as i32;
    let tx = fx - ix as f32;
    let ty = fy - iy as f32;
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (s(tx), s(ty));
    let a = hash2(ix, iy, seed);
    let b = hash2(ix + 1, iy, seed);
    let c = hash2(ix, iy + 1, seed);
    let d = hash2(ix + 1, iy + 1, seed);
    (a + (b - a) * sx) * (1.0 - sy) + (c + (d - c) * sx) * sy
}

impl Dab {
    /// Integer pixel bounds [x0, y0, x1, y1) affected by this dab.
    pub fn bounds(&self) -> [i32; 4] {
        let r = self.radius.max(0.5) + 1.5;
        [
            (self.center[0] - r).floor() as i32,
            (self.center[1] - r).floor() as i32,
            (self.center[0] + r).ceil() as i32 + 1,
            (self.center[1] + r).ceil() as i32 + 1,
        ]
    }

    /// Coverage in [0,1] of the pixel whose centre is at (`px`, `py`) (document coords).
    #[inline]
    pub fn coverage(&self, px: f32, py: f32) -> f32 {
        let dx = px - self.center[0];
        let dy = py - self.center[1];
        let (s, c) = self.angle.sin_cos();
        let lx = dx * c + dy * s;
        let ly = (-dx * s + dy * c) / self.roundness.max(0.05);
        let r = self.radius.max(0.5);
        if self.pixel {
            // Aliased: pixel centre inside the shape.
            let inside = match self.tip {
                TipShape::Square => lx.abs() <= r && ly.abs() <= r,
                _ => lx * lx + ly * ly <= r * r + 0.25,
            };
            return if inside { 1.0 } else { 0.0 };
        }
        let d = match self.tip {
            TipShape::Square => lx.abs().max(ly.abs()),
            _ => (lx * lx + ly * ly).sqrt(),
        };
        // Soft falloff from `hardness * r` to `r`, at least ~1px wide for anti-aliasing.
        let feather = ((1.0 - self.hardness) * r).max(0.75);
        let inner = r - feather;
        let mut cov = if d <= inner {
            1.0
        } else if d >= r + 0.5 {
            0.0
        } else {
            let t = ((d - inner) / (r + 0.5 - inner)).clamp(0.0, 1.0);
            // smootherstep-like falloff; gaussian-ish for soft brushes
            if self.hardness < 0.5 {
                (-(t * t) * 4.0).exp() * (1.0 - t)
            } else {
                1.0 - t * t * (3.0 - 2.0 * t)
            }
        };
        match self.tip {
            TipShape::Chalk => {
                let g = value_noise(px, py, 2.2, 7) * 0.6 + value_noise(px, py, 6.0, 3) * 0.4;
                let thr = self.grain * (1.0 - cov * self.alpha * 0.5);
                cov *= ((g - thr * 0.8) / 0.25).clamp(0.0, 1.0);
            }
            TipShape::Charcoal => {
                // Streaks along the tip angle plus fine grain.
                let streak = value_noise(lx * 0.25 + px * 0.02, ly * 3.0 + self.seed as f32 * 0.0, 1.5, 11);
                let g = value_noise(px, py, 1.6, 5);
                let v = streak * 0.5 + g * 0.5;
                cov *= ((v - self.grain * 0.55) / 0.3).clamp(0.0, 1.0);
            }
            TipShape::Splatter => {
                // Droplets: cells of a jittered grid inside the dab.
                let cell = (r * 0.18).max(1.5);
                let cx = (lx / cell).floor();
                let cy = (ly / cell).floor();
                let h = hash2(cx as i32, cy as i32, self.seed);
                if h < 0.72 {
                    return 0.0;
                }
                let ox = (cx + 0.5 + (hash2(cx as i32 + 3, cy as i32, self.seed) - 0.5) * 0.6) * cell;
                let oy = (cy + 0.5 + (hash2(cx as i32, cy as i32 + 5, self.seed) - 0.5) * 0.6) * cell;
                let dr = ((lx - ox).powi(2) + (ly - oy).powi(2)).sqrt();
                let rr = cell * 0.5 * (0.4 + h * 0.6);
                cov = if d > r { 0.0 } else { (rr - dr + 0.5).clamp(0.0, 1.0) };
            }
            _ => {}
        }
        cov
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dab(r: f32, hardness: f32, pixel: bool) -> Dab {
        Dab { center: [10.0, 10.0], radius: r, angle: 0.0, roundness: 1.0, hardness, alpha: 1.0, tip: TipShape::Round, pixel, grain: 0.0, seed: 1 }
    }

    #[test]
    fn hard_round_area() {
        let d = dab(5.0, 1.0, false);
        let mut sum = 0.0;
        for y in 0..20 {
            for x in 0..20 {
                sum += d.coverage(x as f32 + 0.5, y as f32 + 0.5);
            }
        }
        let expect = std::f32::consts::PI * 25.0;
        assert!((sum - expect).abs() / expect < 0.08, "{sum} vs {expect}");
    }

    #[test]
    fn pixel_dab_is_binary() {
        let d = dab(2.0, 1.0, true);
        for y in 0..20 {
            for x in 0..20 {
                let c = d.coverage(x as f32 + 0.5, y as f32 + 0.5);
                assert!(c == 0.0 || c == 1.0);
            }
        }
    }
}
