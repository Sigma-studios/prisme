//! 3D LUT baking for display transforms.
//!
//! The LUT domain is linear working RGB clamped to [0,1], indexed through the sRGB
//! curve for perceptual uniformity. Output is display-encoded RGB.

use crate::icc::{IccProfile, IccTransform};
use crate::{linear_to_srgb, srgb_to_linear, SpaceConverter};

pub const DISPLAY_LUT_SIZE: usize = 33;

/// Bake a `size³` RGBA f32 LUT mapping working RGB to display-encoded RGB for an ICC profile.
pub fn bake_display_lut(display: &IccProfile, size: usize) -> Option<Vec<f32>> {
    let t = IccTransform::from_working(&display.profile)?;
    let mut src = Vec::with_capacity(size * size * size * 3);
    for b in 0..size {
        for g in 0..size {
            for r in 0..size {
                let f = |i: usize| srgb_to_linear(i as f32 / (size - 1) as f32);
                src.extend([f(r), f(g), f(b)]);
            }
        }
    }
    let out = t.apply(&src);
    Some(out.chunks(3).flat_map(|c| [c[0], c[1], c[2], 1.0]).collect())
}

/// Bake an analytic LUT for a standard space (used to test the LUT path and for P3 web canvases).
pub fn bake_standard_lut(space: crate::RgbSpace, size: usize) -> Vec<f32> {
    let conv = SpaceConverter::new(space);
    let mut out = Vec::with_capacity(size * size * size * 4);
    for b in 0..size {
        for g in 0..size {
            for r in 0..size {
                let f = |i: usize| srgb_to_linear(i as f32 / (size - 1) as f32);
                let e = conv.from_working([f(r), f(g), f(b)]);
                out.extend([e[0], e[1], e[2], 1.0]);
            }
        }
    }
    out
}

/// CPU reference of the shader lookup (trilinear).
pub fn sample_lut(lut: &[f32], size: usize, rgb: [f32; 3]) -> [f32; 3] {
    let idx = rgb.map(|v| linear_to_srgb(v.clamp(0.0, 1.0)) * (size - 1) as f32);
    let i0 = idx.map(|v| (v.floor() as usize).min(size - 2));
    let t = [idx[0] - i0[0] as f32, idx[1] - i0[1] as f32, idx[2] - i0[2] as f32];
    let at = |r: usize, g: usize, b: usize| {
        let o = ((b * size + g) * size + r) * 4;
        [lut[o], lut[o + 1], lut[o + 2]]
    };
    let mut out = [0.0f32; 3];
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let w = (if dx == 1 { t[0] } else { 1.0 - t[0] })
                    * (if dy == 1 { t[1] } else { 1.0 - t[1] })
                    * (if dz == 1 { t[2] } else { 1.0 - t[2] });
                let v = at(i0[0] + dx, i0[1] + dy, i0[2] + dz);
                for k in 0..3 {
                    out[k] += v[k] * w;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RgbSpace;

    #[test]
    fn icc_lut_close_to_analytic() {
        let p = IccProfile::parse(&IccProfile::encode_standard(RgbSpace::DisplayP3).unwrap()).unwrap();
        let lut = bake_display_lut(&p, DISPLAY_LUT_SIZE).unwrap();
        let conv = SpaceConverter::new(RgbSpace::DisplayP3);
        for rgb in [[0.18, 0.18, 0.18], [0.5, 0.2, 0.05], [0.9, 0.9, 0.1]] {
            let a = conv.from_working(rgb);
            let b = sample_lut(&lut, DISPLAY_LUT_SIZE, rgb);
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() < 4e-3, "{rgb:?}: {a:?} vs {b:?}");
            }
        }
    }
}
