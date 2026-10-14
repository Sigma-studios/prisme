//! Conversion of decoded samples into straight-alpha linear working-space RGBA.
//!
//! Every decoder funnels its pixels through [`to_working`], which handles the sample
//! type (8/16-bit integer, float), the channel layout (gray, RGB, CMYK, with or without
//! alpha) and the source colour description (analytic transfer + primaries, or an
//! embedded ICC profile). Work is split over row blocks and parallelised on native.

use prisme_color::{IccProfile, IccTransform, Mat3, RgbSpace, Transfer, D65};

/// Borrowed interleaved samples. Integer samples are normalised by their type's max.
#[derive(Clone, Copy)]
pub(crate) enum Samples<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    F32(&'a [f32]),
}

impl Samples<'_> {
    fn len(&self) -> usize {
        match self {
            Samples::U8(s) => s.len(),
            Samples::U16(s) => s.len(),
            Samples::F32(s) => s.len(),
        }
    }

    #[inline]
    fn get(&self, i: usize) -> f32 {
        match self {
            Samples::U8(s) => s[i] as f32 * (1.0 / 255.0),
            Samples::U16(s) => s[i] as f32 * (1.0 / 65535.0),
            Samples::F32(s) => s[i],
        }
    }
}

/// Channel layout of interleaved samples. CMYK samples are ink amounts (0 = no ink).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Layout {
    Gray,
    GrayAlpha,
    Rgb,
    Rgba,
    Cmyk,
    Cmyka,
}

impl Layout {
    pub fn channels(self) -> usize {
        match self {
            Layout::Gray => 1,
            Layout::GrayAlpha => 2,
            Layout::Rgb => 3,
            Layout::Rgba | Layout::Cmyk => 4,
            Layout::Cmyka => 5,
        }
    }

    pub fn has_alpha(self) -> bool {
        self.alpha_index().is_some()
    }

    fn alpha_index(self) -> Option<usize> {
        match self {
            Layout::GrayAlpha => Some(1),
            Layout::Rgba => Some(3),
            Layout::Cmyka => Some(4),
            _ => None,
        }
    }

    fn colour_channels(self) -> usize {
        match self {
            Layout::Gray | Layout::GrayAlpha => 1,
            Layout::Rgb | Layout::Rgba => 3,
            Layout::Cmyk | Layout::Cmyka => 4,
        }
    }

    fn is_cmyk(self) -> bool {
        matches!(self, Layout::Cmyk | Layout::Cmyka)
    }
}

/// Analytic colour encoding: per-channel transfer function then a 3x3 matrix to the
/// working space.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Analytic {
    pub transfer: Transfer,
    pub to_working: Mat3,
}

impl Analytic {
    pub fn space(space: RgbSpace) -> Self {
        Self { transfer: space.transfer(), to_working: space.to_working_matrix() }
    }

    /// Custom primaries (xy) with D65-relative adaptation handled by Bradford.
    pub fn primaries(transfer: Transfer, r: [f64; 2], g: [f64; 2], b: [f64; 2], w: [f64; 2]) -> Self {
        let mut to_xyz = Mat3::rgb_to_xyz(r, g, b, w);
        if (w[0] - D65[0]).abs() > 1e-4 || (w[1] - D65[1]).abs() > 1e-4 {
            to_xyz = Mat3::bradford(w, D65).mul(&to_xyz);
        }
        let from_xyz = RgbSpace::LinearRec2020.to_xyz_d65().inverse();
        Self { transfer, to_working: from_xyz.mul(&to_xyz) }
    }
}

/// Where the colour meaning of the samples comes from.
#[derive(Clone)]
pub(crate) enum ColorSource {
    Analytic(Analytic),
    Icc(IccProfile),
}

impl ColorSource {
    pub fn srgb() -> Self {
        ColorSource::Analytic(Analytic::space(RgbSpace::Srgb))
    }

    pub fn space(space: RgbSpace) -> Self {
        ColorSource::Analytic(Analytic::space(space))
    }

    /// Parse an embedded ICC profile. Returns the source and the profile description.
    pub fn from_icc(bytes: &[u8]) -> Option<(ColorSource, String)> {
        if bytes.is_empty() {
            return None;
        }
        let p = IccProfile::parse(bytes)?;
        let d = p.description();
        Some((ColorSource::Icc(p), d))
    }
}

/// "sRGB (untagged)" fallback used by most decoders.
pub(crate) fn untagged_srgb() -> (ColorSource, String) {
    (ColorSource::srgb(), "sRGB (untagged)".to_string())
}

/// Map ITU-T H.273 colour primaries / transfer characteristics to an analytic encoding.
/// Returns `None` for unsupported combinations.
pub(crate) fn cicp(primaries: u8, transfer: u8) -> Option<(Analytic, String)> {
    let (prim, pname) = match primaries {
        1 | 2 => (RgbSpace::Srgb.primaries(), "BT.709"),
        9 => (RgbSpace::Rec2020.primaries(), "BT.2020"),
        12 => (RgbSpace::DisplayP3.primaries(), "Display P3"),
        11 => (([0.680, 0.320], [0.265, 0.690], [0.150, 0.060], [0.314, 0.351]), "DCI-P3"),
        5 | 6 => (([0.64, 0.33], [0.29, 0.60], [0.15, 0.06], D65), "BT.601"),
        _ => return None,
    };
    let (tf, tname) = match transfer {
        1 | 6 | 14 | 15 => (Transfer::Rec709, "BT.709"),
        2 | 13 => (Transfer::Srgb, "sRGB"),
        4 => (Transfer::Gamma(2.2), "Gamma 2.2"),
        5 => (Transfer::Gamma(2.8), "Gamma 2.8"),
        8 => (Transfer::Linear, "Linear"),
        16 => (Transfer::Pq, "PQ"),
        18 => (Transfer::Hlg, "HLG"),
        _ => return None,
    };
    let (r, g, b, w) = prim;
    Some((Analytic::primaries(tf, r, g, b, w), format!("{pname} primaries, {tname} transfer (CICP)")))
}

// --------------------------------------------------------------------------------
// Parallel helpers
// --------------------------------------------------------------------------------

/// Run `f(first_row, chunk)` over blocks of rows of `out` (`row_len` floats per row),
/// in parallel on native.
pub(crate) fn for_row_blocks(out: &mut [f32], row_len: usize, f: impl Fn(usize, &mut [f32]) + Sync + Send) {
    if row_len == 0 || out.is_empty() {
        return;
    }
    // ~64k pixels per block keeps the per-task overhead negligible.
    let rows_per_block = (65536 * 4 / row_len).max(1);
    let block = rows_per_block * row_len;
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(block).enumerate().for_each(|(i, c)| f(i * rows_per_block, c));
    }
    #[cfg(target_arch = "wasm32")]
    {
        out.chunks_mut(block).enumerate().for_each(|(i, c)| f(i * rows_per_block, c));
    }
}

/// Lookup table mapping integer codes to linear values.
fn build_lut(n: usize, transfer: Transfer) -> Vec<f32> {
    let scale = 1.0 / (n - 1) as f32;
    (0..n).map(|i| transfer.to_linear(i as f32 * scale)).collect()
}

/// Convert interleaved samples into straight-alpha linear working-space RGBA.
pub(crate) fn to_working(samples: Samples<'_>, width: u32, height: u32, layout: Layout, source: &ColorSource) -> Vec<f32> {
    let n = width as usize * height as usize;
    let ch = layout.channels();
    debug_assert!(samples.len() >= n * ch);
    let mut out = vec![0.0f32; n * 4];
    match source {
        ColorSource::Icc(profile) => {
            if let Some(()) = icc_to_working(samples, width, layout, profile, &mut out) {
                return out;
            }
            log::warn!("ICC profile '{}' does not match pixel layout {layout:?}; assuming sRGB", profile.description());
            analytic_to_working(samples, width, layout, &Analytic::space(RgbSpace::Srgb), &mut out);
        }
        ColorSource::Analytic(a) => analytic_to_working(samples, width, layout, a, &mut out),
    }
    out
}

fn analytic_to_working(samples: Samples<'_>, width: u32, layout: Layout, a: &Analytic, out: &mut [f32]) {
    let ch = layout.channels();
    let w = width as usize;
    let lut = match (samples, a.transfer) {
        (_, Transfer::Linear) | (Samples::F32(_), _) => None,
        (Samples::U8(_), t) => Some(build_lut(256, t)),
        (Samples::U16(_), t) => Some(build_lut(65536, t)),
    };
    let m = a.to_working.0.map(|r| r.map(|v| v as f32));
    let identity = a.to_working == Mat3::IDENTITY;
    let transfer = a.transfer;
    let lin = |i: usize| -> f32 {
        match (&lut, samples) {
            (Some(l), Samples::U8(s)) => l[s[i] as usize],
            (Some(l), Samples::U16(s)) => l[s[i] as usize],
            _ => transfer.to_linear(samples.get(i)),
        }
    };
    let alpha_index = layout.alpha_index();
    for_row_blocks(out, w * 4, |row0, block| {
        let base = row0 * w;
        for (j, px) in block.chunks_exact_mut(4).enumerate() {
            let s = (base + j) * ch;
            let rgb = if layout.is_cmyk() {
                let k = 1.0 - samples.get(s + 3);
                let c = |i: usize| transfer.to_linear((1.0 - samples.get(s + i)) * k);
                [c(0), c(1), c(2)]
            } else if layout.colour_channels() == 1 {
                let v = lin(s);
                [v, v, v]
            } else {
                [lin(s), lin(s + 1), lin(s + 2)]
            };
            let rgb = if identity {
                rgb
            } else {
                [
                    m[0][0] * rgb[0] + m[0][1] * rgb[1] + m[0][2] * rgb[2],
                    m[1][0] * rgb[0] + m[1][1] * rgb[1] + m[1][2] * rgb[2],
                    m[2][0] * rgb[0] + m[2][1] * rgb[1] + m[2][2] * rgb[2],
                ]
            };
            px[..3].copy_from_slice(&rgb);
            px[3] = alpha_index.map_or(1.0, |ai| samples.get(s + ai));
        }
    });
}

fn icc_to_working(samples: Samples<'_>, width: u32, layout: Layout, profile: &IccProfile, out: &mut [f32]) -> Option<()> {
    // CMYK+alpha goes through the 4-channel CMYK transform; alpha is copied separately.
    let icc_channels = match layout {
        Layout::Cmyka => 4,
        l => l.channels(),
    };
    let transform = IccTransform::to_working(profile, icc_channels)?;
    let ch = layout.channels();
    let dst_ch = transform.dst_channels();
    let w = width as usize;
    let alpha_index = layout.alpha_index();
    for_row_blocks(out, w * 4, |row0, block| {
        let base = row0 * w;
        let npx = block.len() / 4;
        let mut src = Vec::with_capacity(npx * icc_channels);
        for j in 0..npx {
            let s = (base + j) * ch;
            for c in 0..icc_channels {
                src.push(samples.get(s + c));
            }
        }
        let dst = transform.apply(&src);
        for (j, px) in block.chunks_exact_mut(4).enumerate() {
            let d = &dst[j * dst_ch..];
            px[..3].copy_from_slice(&d[..3]);
            px[3] = if dst_ch == 4 {
                d[3]
            } else {
                alpha_index.map_or(1.0, |ai| samples.get((base + j) * ch + ai))
            };
        }
    });
    Some(())
}

/// Convert premultiplied RGBA in place to straight alpha.
pub(crate) fn unpremultiply(rgba: &mut [f32], width: u32) {
    for_row_blocks(rgba, width as usize * 4, |_, block| {
        for px in block.chunks_exact_mut(4) {
            let a = px[3];
            if a > 0.0 && a != 1.0 {
                let inv = 1.0 / a;
                px[0] *= inv;
                px[1] *= inv;
                px[2] *= inv;
            }
        }
    });
}

/// Normalised copy of `n` interleaved samples with colour channels divided by alpha
/// (for sources with associated alpha).
pub(crate) fn unpremultiplied_samples(samples: Samples<'_>, layout: Layout, n: usize) -> Vec<f32> {
    let ch = layout.channels();
    let mut out: Vec<f32> = (0..n).map(|i| samples.get(i)).collect();
    let Some(ai) = layout.alpha_index() else { return out };
    for px in out.chunks_exact_mut(ch) {
        let a = px[ai];
        if a > 0.0 {
            for (c, v) in px.iter_mut().enumerate() {
                if c != ai {
                    *v /= a;
                }
            }
        }
    }
    out
}

// --------------------------------------------------------------------------------
// Geometry: EXIF orientation and downscaling
// --------------------------------------------------------------------------------

/// Read the EXIF orientation tag (1..=8) from raw EXIF data (TIFF header, optionally
/// prefixed by `Exif\0\0`).
pub(crate) fn exif_orientation(exif: &[u8]) -> Option<u16> {
    let d = exif.strip_prefix(b"Exif\0\0").unwrap_or(exif);
    let le = match d.get(..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |o: usize| -> Option<u16> {
        let b: [u8; 2] = d.get(o..o + 2)?.try_into().ok()?;
        Some(if le { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) })
    };
    let u32_at = |o: usize| -> Option<u32> {
        let b: [u8; 4] = d.get(o..o + 4)?.try_into().ok()?;
        Some(if le { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) })
    };
    if u16_at(2)? != 42 {
        return None;
    }
    let ifd = u32_at(4)? as usize;
    let count = u16_at(ifd)? as usize;
    for i in 0..count.min(512) {
        let e = ifd + 2 + i * 12;
        if u16_at(e)? == 0x0112 {
            let v = u16_at(e + 8)?;
            return (1..=8).contains(&v).then_some(v);
        }
    }
    None
}

/// Apply an EXIF orientation (1..=8) to an RGBA f32 image. Returns the new buffer and size.
pub(crate) fn apply_orientation(pixels: Vec<f32>, w: u32, h: u32, orientation: u16) -> (Vec<f32>, u32, u32) {
    if !(2..=8).contains(&orientation) {
        return (pixels, w, h);
    }
    let (wu, hu) = (w as usize, h as usize);
    let (ow, oh) = if orientation >= 5 { (h, w) } else { (w, h) };
    let owu = ow as usize;
    let mut out = vec![0.0f32; pixels.len()];
    let src = &pixels;
    for_row_blocks(&mut out, owu * 4, |row0, block| {
        for (j, px) in block.chunks_exact_mut(4).enumerate() {
            let i = row0 * owu + j;
            let (x, y) = (i % owu, i / owu);
            let (sx, sy) = match orientation {
                2 => (wu - 1 - x, y),
                3 => (wu - 1 - x, hu - 1 - y),
                4 => (x, hu - 1 - y),
                5 => (y, x),
                6 => (y, hu - 1 - x),
                7 => (wu - 1 - y, hu - 1 - x),
                _ => (wu - 1 - y, x),
            };
            let s = (sy * wu + sx) * 4;
            px.copy_from_slice(&src[s..s + 4]);
        }
    });
    (out, ow, oh)
}

/// Area-average downscale of straight-alpha RGBA so the longest side is <= `max_side`.
pub(crate) fn downscale(pixels: &[f32], w: u32, h: u32, max_side: u32) -> (Vec<f32>, u32, u32) {
    let max_side = max_side.max(1);
    let longest = w.max(h);
    if longest <= max_side {
        return (pixels.to_vec(), w, h);
    }
    let s = max_side as f64 / longest as f64;
    let ow = ((w as f64 * s).round() as u32).clamp(1, max_side);
    let oh = ((h as f64 * s).round() as u32).clamp(1, max_side);
    let (sx, sy) = (w as f64 / ow as f64, h as f64 / oh as f64);
    let mut out = vec![0.0f32; ow as usize * oh as usize * 4];
    for_row_blocks(&mut out, ow as usize * 4, |row0, block| {
        for (j, px) in block.chunks_exact_mut(4).enumerate() {
            let i = row0 * ow as usize + j;
            let (ox, oy) = (i % ow as usize, i / ow as usize);
            let x0 = (ox as f64 * sx) as usize;
            let x1 = (((ox + 1) as f64 * sx).ceil() as usize).min(w as usize).max(x0 + 1);
            let y0 = (oy as f64 * sy) as usize;
            let y1 = (((oy + 1) as f64 * sy).ceil() as usize).min(h as usize).max(y0 + 1);
            let mut acc = [0.0f32; 4];
            let mut n = 0.0f32;
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = &pixels[(y * w as usize + x) * 4..][..4];
                    acc[0] += p[0] * p[3];
                    acc[1] += p[1] * p[3];
                    acc[2] += p[2] * p[3];
                    acc[3] += p[3];
                    n += 1.0;
                }
            }
            if acc[3] > 0.0 {
                px[0] = acc[0] / acc[3];
                px[1] = acc[1] / acc[3];
                px[2] = acc[2] / acc[3];
            }
            px[3] = acc[3] / n;
        }
    });
    (out, ow, oh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_maps_corners() {
        // 2x1 image: red, green.
        let px = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let (o, w, h) = apply_orientation(px.clone(), 2, 1, 6);
        assert_eq!((w, h), (1, 2));
        assert_eq!(&o[..4], &[1.0, 0.0, 0.0, 1.0]); // rotate CW: left pixel goes on top
        let (o, w, h) = apply_orientation(px.clone(), 2, 1, 8);
        assert_eq!((w, h), (1, 2));
        assert_eq!(&o[..4], &[0.0, 1.0, 0.0, 1.0]);
        let (o, _, _) = apply_orientation(px, 2, 1, 2);
        assert_eq!(&o[..4], &[0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn exif_parse() {
        // Minimal little-endian TIFF with one IFD entry: Orientation = 6.
        let mut d = b"Exif\0\0II*\0\x08\0\0\0".to_vec();
        d.extend_from_slice(&1u16.to_le_bytes());
        d.extend_from_slice(&0x0112u16.to_le_bytes());
        d.extend_from_slice(&3u16.to_le_bytes());
        d.extend_from_slice(&1u32.to_le_bytes());
        d.extend_from_slice(&6u16.to_le_bytes());
        d.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
        assert_eq!(exif_orientation(&d), Some(6));
        assert_eq!(exif_orientation(&d[..12]), None);
    }

    #[test]
    fn cicp_srgb_matches_analytic() {
        let (a, _) = cicp(1, 13).unwrap();
        let s = Analytic::space(RgbSpace::Srgb);
        for (r1, r2) in a.to_working.0.iter().zip(s.to_working.0.iter()) {
            for (x, y) in r1.iter().zip(r2) {
                assert!((x - y).abs() < 1e-6);
            }
        }
    }
}
