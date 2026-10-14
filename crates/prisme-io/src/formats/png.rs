//! PNG through the `png` crate (8/16-bit, cICP > iCCP > sRGB > gAMA/cHRM).

use super::ImageDecoder;
use crate::color::{self, cicp, Analytic, ColorSource, Layout, Samples};
use crate::{Decoded, DecodedImage, IoError};
use prisme_color::{RgbSpace, Transfer};
use std::io::Cursor;

pub struct Png;

impl ImageDecoder for Png {
    fn name(&self) -> &'static str {
        "PNG"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["png", "apng"]
    }

    fn sniff(&self, h: &[u8]) -> bool {
        h.starts_with(b"\x89PNG\r\n\x1a\n")
    }

    fn decode(&self, bytes: &[u8]) -> Result<Decoded, IoError> {
        decode_png(bytes).map(Decoded::Raster)
    }
}

fn describe_source(info: &png::Info<'_>) -> (ColorSource, String) {
    if let Some(c) = &info.coding_independent_code_points
        && let Some((a, name)) = cicp(c.color_primaries, c.transfer_function)
    {
        if !c.is_video_full_range_image {
            log::warn!("PNG cICP narrow range is not supported; treating as full range");
        }
        return (ColorSource::Analytic(a), name);
    }
    if let Some(r) = info.icc_profile.as_deref().and_then(ColorSource::from_icc) {
        return r;
    }
    if info.srgb.is_some() {
        return (ColorSource::srgb(), "sRGB".into());
    }
    let gamma = info.source_gamma.map(|g| g.into_value()).filter(|g| *g > 0.0);
    let chrm = info.source_chromaticities.map(|c| {
        let p = |v: (png::ScaledFloat, png::ScaledFloat)| [v.0.into_value() as f64, v.1.into_value() as f64];
        (p(c.red), p(c.green), p(c.blue), p(c.white))
    });
    if gamma.is_none() && chrm.is_none() {
        return color::untagged_srgb();
    }
    // gAMA stores the encoding exponent (0.45455 for a 2.2 display gamma).
    let near_srgb_gamma = gamma.is_none_or(|g| (g - 1.0 / 2.2).abs() < 0.001);
    let srgb_prims = RgbSpace::Srgb.primaries();
    let near_srgb_prims = chrm.is_none_or(|c| {
        let close = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() < 0.002 && (a[1] - b[1]).abs() < 0.002;
        close(c.0, srgb_prims.0) && close(c.1, srgb_prims.1) && close(c.2, srgb_prims.2) && close(c.3, srgb_prims.3)
    });
    if near_srgb_gamma && near_srgb_prims {
        return (ColorSource::srgb(), "sRGB (gAMA/cHRM)".into());
    }
    let transfer = match gamma {
        Some(g) if (g - 1.0).abs() < 1e-3 => Transfer::Linear,
        Some(g) => Transfer::Gamma(1.0 / g),
        None => Transfer::Srgb,
    };
    let (r, g, b, w) = chrm.unwrap_or(srgb_prims);
    let name = match gamma {
        Some(g) => format!("gAMA {:.4}{}", g, if chrm.is_some() { " + cHRM" } else { "" }),
        None => "cHRM".to_string(),
    };
    (ColorSource::Analytic(Analytic::primaries(transfer, r, g, b, w)), name)
}

pub(crate) fn decode_png(bytes: &[u8]) -> Result<DecodedImage, IoError> {
    let mut dec = png::Decoder::new_with_limits(Cursor::new(bytes), png::Limits { bytes: usize::MAX });
    dec.set_transformations(png::Transformations::EXPAND);
    dec.set_ignore_text_chunk(true);
    let mut reader = dec.read_info().map_err(IoError::decode)?;
    let info = reader.info();
    let (w, h) = (info.width, info.height);
    let src_bits = info.bit_depth as u8;
    let (source, desc) = describe_source(info);
    let orientation = info.exif_metadata.as_deref().and_then(color::exif_orientation);
    let size = reader.output_buffer_size().ok_or_else(|| IoError::Decode("PNG too large".into()))?;
    let mut buf = vec![0u8; size];
    let out = reader.next_frame(&mut buf).map_err(IoError::decode)?;
    let layout = match out.color_type {
        png::ColorType::Grayscale => Layout::Gray,
        png::ColorType::GrayscaleAlpha => Layout::GrayAlpha,
        png::ColorType::Rgb => Layout::Rgb,
        png::ColorType::Rgba => Layout::Rgba,
        png::ColorType::Indexed => return Err(IoError::Decode("unexpanded palette".into())),
    };
    let n = w as usize * h as usize * layout.channels();
    let pixels = if out.bit_depth == png::BitDepth::Sixteen {
        let v: Vec<u16> = buf[..n * 2].chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
        color::to_working(Samples::U16(&v), w, h, layout, &source)
    } else {
        color::to_working(Samples::U8(&buf[..n]), w, h, layout, &source)
    };
    Ok(DecodedImage::new(w, h, pixels, desc, src_bits, false).oriented(orientation))
}
