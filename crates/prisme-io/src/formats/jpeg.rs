//! JPEG through `zune-jpeg` (full decode) and `jpeg-decoder` (DCT-domain scaled preview).
//! Honours embedded ICC profiles (including CMYK) and EXIF orientation.

use super::ImageDecoder;
use crate::color::{self, ColorSource, Layout, Samples};
use crate::{Decoded, DecodedImage, IoError};
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_jpeg::JpegDecoder;

pub struct Jpeg;

impl ImageDecoder for Jpeg {
    fn name(&self) -> &'static str {
        "JPEG"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["jpg", "jpeg", "jpe", "jfif"]
    }

    fn sniff(&self, h: &[u8]) -> bool {
        h.starts_with(&[0xFF, 0xD8, 0xFF])
    }

    fn decode(&self, bytes: &[u8]) -> Result<Decoded, IoError> {
        decode_jpeg(bytes).map(Decoded::Raster)
    }

    fn decode_preview(&self, bytes: &[u8], max_side: u32) -> Option<DecodedImage> {
        preview_jpeg(bytes, max_side)
    }
}

fn icc_source(icc: Option<&[u8]>) -> (ColorSource, String) {
    icc.and_then(ColorSource::from_icc).unwrap_or_else(color::untagged_srgb)
}

pub(crate) fn decode_jpeg(bytes: &[u8]) -> Result<DecodedImage, IoError> {
    let mut dec = JpegDecoder::new(ZCursor::new(bytes));
    dec.set_options(dec.options().set_max_width(1 << 20).set_max_height(1 << 20));
    dec.decode_headers().map_err(IoError::decode)?;
    let input = dec.input_colorspace().ok_or_else(|| IoError::Decode("missing JPEG header".into()))?;
    let out_cs = match input {
        ColorSpace::Luma => ColorSpace::Luma,
        ColorSpace::CMYK => ColorSpace::CMYK,
        ColorSpace::YCCK => ColorSpace::YCCK,
        _ => ColorSpace::RGB,
    };
    dec.set_options(dec.options().jpeg_set_out_colorspace(out_cs));
    let icc = dec.icc_profile();
    let orientation = dec.exif().and_then(|e| color::exif_orientation(e));
    let data = dec.decode().map_err(IoError::decode)?;
    let info = dec.info().ok_or_else(|| IoError::Decode("missing JPEG info".into()))?;
    let (w, h) = (info.width as u32, info.height as u32);
    let (mut source, mut desc) = icc_source(icc.as_deref());
    let (layout, samples): (Layout, std::borrow::Cow<[u8]>) = match out_cs {
        ColorSpace::Luma => (Layout::Gray, data.into()),
        ColorSpace::CMYK => (Layout::Cmyk, adobe_cmyk_to_ink(&data).into()),
        ColorSpace::YCCK => (Layout::Cmyk, ycck_to_ink(&data).into()),
        _ => (Layout::Rgb, data.into()),
    };
    if layout == Layout::Cmyk && !matches!(&source, ColorSource::Icc(p) if p.is_cmyk()) {
        source = ColorSource::srgb();
        desc = "CMYK (untagged, naive conversion)".into();
    }
    let pixels = color::to_working(Samples::U8(&samples), w, h, layout, &source);
    Ok(DecodedImage::new(w, h, pixels, desc, 8, false).oriented(orientation))
}

/// Adobe CMYK JPEGs store inverted values (255 = no ink).
fn adobe_cmyk_to_ink(data: &[u8]) -> Vec<u8> {
    data.iter().map(|v| 255 - v).collect()
}

/// YCCK (Adobe transform 2): YCbCr encodes inverted CMY; K is stored inverted.
fn ycck_to_ink(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    for p in data.chunks_exact(4) {
        let (y, cb, cr) = (p[0] as f32, p[1] as f32 - 128.0, p[2] as f32 - 128.0);
        let r = y + 1.402 * cr;
        let g = y - 0.344_136 * cb - 0.714_136 * cr;
        let b = y + 1.772 * cb;
        // inverted C = 255 - R, so ink C = R.
        let q = |v: f32| v.round().clamp(0.0, 255.0) as u8;
        out.extend_from_slice(&[q(r), q(g), q(b), 255 - p[3]]);
    }
    out
}

/// Preview via `jpeg-decoder`'s DCT-domain scaling (1/2, 1/4, 1/8), then box filtering.
/// None when no scaling applies (longest side under 2 × `max_side`): the preview would then
/// be a full decode, costing as much as the real one (measured 1.0–1.5×).
fn preview_jpeg(bytes: &[u8], max_side: u32) -> Option<DecodedImage> {
    let mut dec = jpeg_decoder::Decoder::new(bytes);
    dec.read_info().ok()?;
    let info = dec.info()?;
    let (w, h) = (info.width as u32, info.height as u32);
    let longest = w.max(h).max(1);
    let s = (max_side as f64 / longest as f64).min(1.0);
    let rw = ((w as f64 * s).ceil() as u32).clamp(1, u16::MAX as u32) as u16;
    let rh = ((h as f64 * s).ceil() as u32).clamp(1, u16::MAX as u32) as u16;
    let (sw, sh) = dec.scale(rw, rh).ok()?;
    if (sw as u32, sh as u32) == (w, h) {
        return None;
    }
    let data = dec.decode().ok()?;
    let layout = match dec.info()?.pixel_format {
        jpeg_decoder::PixelFormat::L8 => Layout::Gray,
        jpeg_decoder::PixelFormat::RGB24 => Layout::Rgb,
        _ => return None,
    };
    let (source, desc) = icc_source(dec.icc_profile().as_deref());
    let orientation = dec.exif_data().and_then(color::exif_orientation);
    let (sw, sh) = (sw as u32, sh as u32);
    if data.len() < sw as usize * sh as usize * layout.channels() {
        return None;
    }
    let pixels = color::to_working(Samples::U8(&data), sw, sh, layout, &source);
    Some(DecodedImage::new(sw, sh, pixels, desc, 8, false).downscaled(max_side).oriented(orientation))
}
