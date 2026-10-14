//! ICC profile handling through `moxcms` (pure Rust, works on wasm).

use moxcms::{ColorProfile, DataColorSpace, Layout, ToneReprCurve, TransformOptions};
use std::sync::Arc;

/// A parsed ICC profile plus its original bytes (kept for re-embedding).
#[derive(Clone)]
pub struct IccProfile {
    pub bytes: Arc<Vec<u8>>,
    pub profile: Arc<ColorProfile>,
}

impl std::fmt::Debug for IccProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IccProfile({}, {} bytes)", self.description(), self.bytes.len())
    }
}

impl IccProfile {
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        match ColorProfile::new_from_slice(bytes) {
            Ok(p) => Some(Self { bytes: Arc::new(bytes.to_vec()), profile: Arc::new(p) }),
            Err(e) => {
                log::warn!("invalid ICC profile: {e:?}");
                None
            }
        }
    }

    pub fn description(&self) -> String {
        use moxcms::ProfileText;
        match &self.profile.description {
            Some(ProfileText::PlainString(s)) => s.clone(),
            Some(ProfileText::Localizable(v)) => v.first().map(|l| l.value.clone()).unwrap_or_default(),
            Some(ProfileText::Description(d)) => d.ascii_string.clone(),
            None => "Unnamed profile".into(),
        }
    }

    pub fn is_gray(&self) -> bool {
        self.profile.color_space == DataColorSpace::Gray
    }

    pub fn is_rgb(&self) -> bool {
        self.profile.color_space == DataColorSpace::Rgb
    }

    pub fn is_cmyk(&self) -> bool {
        self.profile.color_space == DataColorSpace::Cmyk
    }

    /// Encoded ICC bytes for a standard space (for embedding on export).
    pub fn encode_standard(space: crate::RgbSpace) -> Option<Vec<u8>> {
        standard_profile(space).encode().ok()
    }
}

/// Linear Rec.2020 profile == the working space.
pub fn working_profile() -> ColorProfile {
    let mut p = ColorProfile::new_bt2020();
    let lin = ToneReprCurve::Parametric(vec![1.0]);
    p.red_trc = Some(lin.clone());
    p.green_trc = Some(lin.clone());
    p.blue_trc = Some(lin);
    p
}

pub fn standard_profile(space: crate::RgbSpace) -> ColorProfile {
    use crate::RgbSpace::*;
    let linearize = |mut p: ColorProfile| {
        let lin = ToneReprCurve::Parametric(vec![1.0]);
        p.red_trc = Some(lin.clone());
        p.green_trc = Some(lin.clone());
        p.blue_trc = Some(lin);
        p
    };
    match space {
        Srgb => ColorProfile::new_srgb(),
        LinearSrgb => linearize(ColorProfile::new_srgb()),
        DisplayP3 => ColorProfile::new_display_p3(),
        Rec2020 => ColorProfile::new_bt2020(),
        LinearRec2020 => working_profile(),
        AdobeRgb => ColorProfile::new_adobe_rgb(),
        ProPhoto => ColorProfile::new_pro_photo_rgb(),
        Rec2100Pq => ColorProfile::new_bt2020_pq(),
        Rec2100Hlg => ColorProfile::new_bt2020_hlg(),
    }
}

/// Transform from a source ICC profile into the linear working space.
pub struct IccTransform {
    exec: Arc<moxcms::TransformF32Executor>,
    src_channels: usize,
    dst_channels: usize,
}

impl IccTransform {
    /// `channels` is the source channel count: 1 (gray), 2 (gray+alpha), 3 (RGB), 4 (RGBA or CMYK).
    pub fn to_working(src: &IccProfile, channels: usize) -> Option<Self> {
        let dst = working_profile();
        let (sl, dl, dc) = match (src.profile.color_space, channels) {
            (DataColorSpace::Gray, 1) => (Layout::Gray, Layout::Rgb, 3),
            (DataColorSpace::Gray, 2) => (Layout::GrayAlpha, Layout::Rgba, 4),
            (DataColorSpace::Rgb, 3) => (Layout::Rgb, Layout::Rgb, 3),
            (DataColorSpace::Rgb, 4) => (Layout::Rgba, Layout::Rgba, 4),
            (DataColorSpace::Cmyk, 4) => (Layout::Rgba, Layout::Rgb, 3),
            _ => return None,
        };
        let opts = TransformOptions { prefer_fixed_point: false, ..Default::default() };
        match src.profile.create_transform_f32(sl, &dst, dl, opts) {
            Ok(exec) => Some(Self { exec, src_channels: channels, dst_channels: dc }),
            Err(e) => {
                log::warn!("ICC transform failed: {e:?}");
                None
            }
        }
    }

    /// Transform from the working space into an arbitrary destination profile (RGB in, RGB out).
    pub fn from_working(dst: &ColorProfile) -> Option<Self> {
        let src = working_profile();
        let opts = TransformOptions { prefer_fixed_point: false, ..Default::default() };
        src.create_transform_f32(Layout::Rgb, dst, Layout::Rgb, opts)
            .ok()
            .map(|exec| Self { exec, src_channels: 3, dst_channels: 3 })
    }

    pub fn src_channels(&self) -> usize {
        self.src_channels
    }

    pub fn dst_channels(&self) -> usize {
        self.dst_channels
    }

    /// Transform a buffer of normalized samples. Output has `dst_channels` per pixel.
    pub fn apply(&self, src: &[f32]) -> Vec<f32> {
        let n = src.len() / self.src_channels;
        let mut dst = vec![0.0f32; n * self.dst_channels];
        // moxcms processes whole rows; chunk to keep memory bounded.
        const CHUNK: usize = 1 << 16;
        for (s, d) in src.chunks(CHUNK * self.src_channels).zip(dst.chunks_mut(CHUNK * self.dst_channels)) {
            if let Err(e) = self.exec.transform(s, d) {
                log::warn!("ICC transform error: {e:?}");
            }
        }
        dst
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{delta_e_2000, srgb, working_to_lab, RgbSpace};

    /// ICC (moxcms) and analytic paths must agree within ΔE2000 < 0.5.
    #[test]
    fn icc_matches_analytic_srgb() {
        let bytes = IccProfile::encode_standard(RgbSpace::Srgb).unwrap();
        let p = IccProfile::parse(&bytes).unwrap();
        let t = IccTransform::to_working(&p, 3).unwrap();
        let mut src = vec![];
        for r in 0..8 {
            for g in 0..8 {
                for b in 0..8 {
                    src.extend([r as f32 / 7.0, g as f32 / 7.0, b as f32 / 7.0]);
                }
            }
        }
        let out = t.apply(&src);
        let mut worst = 0.0f32;
        for (s, o) in src.chunks(3).zip(out.chunks(3)) {
            let a = srgb().to_working([s[0], s[1], s[2]]);
            let d = delta_e_2000(working_to_lab(a), working_to_lab([o[0], o[1], o[2]]));
            worst = worst.max(d);
        }
        assert!(worst < 0.5, "worst ΔE2000 = {worst}");
    }

    #[test]
    fn icc_matches_analytic_p3_and_adobe() {
        for space in [RgbSpace::DisplayP3, RgbSpace::AdobeRgb] {
            let p = IccProfile::parse(&IccProfile::encode_standard(space).unwrap()).unwrap();
            let t = IccTransform::to_working(&p, 3).unwrap();
            let conv = crate::SpaceConverter::new(space);
            let src: Vec<f32> = (0..300).map(|i| ((i * 37) % 101) as f32 / 100.0).collect();
            let out = t.apply(&src);
            for (s, o) in src.chunks(3).zip(out.chunks(3)) {
                let a = conv.to_working([s[0], s[1], s[2]]);
                let d = delta_e_2000(working_to_lab(a), working_to_lab([o[0], o[1], o[2]]));
                assert!(d < 0.5, "{space:?}: {s:?} ΔE {d}");
            }
        }
    }
}
