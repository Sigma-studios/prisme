//! HDR / extended-range display helpers.
//!
//! On an extended-range surface, linear `1.0` is SDR reference white (the brightness of
//! UI white) and values above it are brighter-than-white highlights, up to the display's
//! *headroom* (a multiplier of SDR white that the OS reports and that changes with
//! brightness, ambient light and which display the window is on).
//!
//! The display pass maps working-space values to such a surface without touching
//! anything at or below SDR white, and handles highlights above the headroom either by
//! clipping or by a soft roll-off. These CPU versions mirror `display.wgsl` exactly and
//! are what the unit tests check.

/// Start of the soft roll-off for a display with `peak` headroom (× SDR white).
///
/// Values up to `max(1, 0.75·peak)` are passed through unchanged, so SDR content (≤ 1.0)
/// never changes with the headroom. When the knee reaches the peak (peak ≤ 1) the roll-off
/// degenerates to a clip.
pub fn rolloff_knee(peak: f32) -> f32 {
    let peak = peak.max(1.0);
    (0.75 * peak).clamp(1.0, peak)
}

/// Soft highlight roll-off of a non-negative magnitude `m` (e.g. max RGB) into `[0, peak)`.
///
/// Identity below the knee, then an exponential shoulder with slope 1 at the knee that
/// approaches `peak` asymptotically: C¹-continuous and strictly monotonic.
pub fn soft_rolloff(m: f32, peak: f32) -> f32 {
    let peak = peak.max(1.0);
    let k = rolloff_knee(peak);
    let span = peak - k;
    if span < 1e-4 {
        return m.min(peak);
    }
    if m <= k {
        return m;
    }
    k + span * (1.0 - (-(m - k) / span).exp())
}

/// Applies the highlight handling to a linear RGB triple (display primaries).
///
/// `soft`: scale the whole colour by `soft_rolloff(max)/max`, which keeps hue and channel
/// ratios (negative, out-of-gamut channels are scaled along). Otherwise each channel is
/// clipped at `peak` (classic clip; may shift hue of very bright saturated colours).
pub fn map_highlights(rgb: [f32; 3], peak: f32, soft: bool) -> [f32; 3] {
    let peak = peak.max(1.0);
    if !soft {
        return rgb.map(|c| c.min(peak));
    }
    let m = rgb[0].max(rgb[1]).max(rgb[2]);
    if m <= rolloff_knee(peak) {
        return rgb;
    }
    let s = soft_rolloff(m, peak) / m;
    rgb.map(|c| c * s)
}

/// Extended sRGB encoding (IEC 61966-2-2 style): the sRGB OETF mirrored for negative
/// values and continued above 1.0. This is what `ExtendedSrgb` surfaces (Metal
/// `kCGColorSpaceExtendedSRGB`, WebGPU canvas `toneMapping: "extended"`) expect.
pub fn encode_extended_srgb(x: f32) -> f32 {
    let a = x.abs();
    let e = if a <= 0.003_130_8 { a * 12.92 } else { 1.055 * a.powf(1.0 / 2.4) - 0.055 };
    e.copysign(x)
}

/// Inverse of [`encode_extended_srgb`].
pub fn decode_extended_srgb(e: f32) -> f32 {
    let a = e.abs();
    let x = if a <= 0.040_45 { a / 12.92 } else { ((a + 0.055) / 1.055).powf(2.4) };
    x.copysign(e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdr_range_untouched() {
        for peak in [1.0, 1.5, 2.0, 4.0, 16.0] {
            for i in 0..=100 {
                let x = i as f32 / 100.0;
                assert_eq!(soft_rolloff(x, peak), x, "peak {peak} x {x}");
                assert_eq!(map_highlights([x, x * 0.5, 0.0], peak, true), [x, x * 0.5, 0.0]);
            }
        }
    }

    #[test]
    fn rolloff_monotonic_bounded_continuous() {
        for peak in [1.2, 2.0, 4.0, 10.0] {
            let k = rolloff_knee(peak);
            let mut prev = -1.0;
            for i in 0..4000 {
                let x = i as f32 * 0.01;
                let y = soft_rolloff(x, peak);
                assert!(y >= prev, "monotonic at {x}");
                assert!(y <= peak + 1e-5, "bounded at {x}: {y} > {peak}");
                prev = y;
            }
            // Continuity and unit slope at the knee.
            let e = 1e-3;
            assert!((soft_rolloff(k + e, peak) - (k + e)).abs() < 1e-4);
            assert!(soft_rolloff(1e6, peak) > peak - 1e-3);
        }
    }

    #[test]
    fn no_headroom_clips() {
        assert_eq!(soft_rolloff(3.0, 1.0), 1.0);
        let c = map_highlights([3.0, 0.5, -0.1], 1.0, true);
        for (a, b) in c.iter().zip([1.0, 1.0 / 6.0, -0.1 / 3.0]) {
            assert!((a - b).abs() < 1e-6, "{c:?}");
        }
        assert_eq!(map_highlights([3.0, 0.5, -0.1], 2.0, false), [2.0, 0.5, -0.1]);
    }

    #[test]
    fn soft_keeps_ratios() {
        let c = map_highlights([8.0, 4.0, -0.4], 4.0, true);
        assert!((c[1] / c[0] - 0.5).abs() < 1e-6);
        assert!((c[2] / c[0] + 0.05).abs() < 1e-6);
        assert!(c[0] < 4.0);
    }

    #[test]
    fn extended_srgb_roundtrip() {
        for x in [-2.0, -0.5, -0.001, 0.0, 0.002, 0.18, 0.5, 1.0, 2.5, 16.0] {
            let e = encode_extended_srgb(x);
            assert!((decode_extended_srgb(e) - x).abs() < 1e-4 * x.abs().max(1.0), "{x}");
        }
        assert!((encode_extended_srgb(1.0) - 1.0).abs() < 1e-6);
        assert!(encode_extended_srgb(4.0) > 1.0);
        assert_eq!(encode_extended_srgb(-0.5), -encode_extended_srgb(0.5));
    }
}
