//! TIFF through the `tiff` crate: 1–16-bit integer, 16/32/64-bit float, Gray/RGB/CMYK
//! with alpha, embedded ICC, orientation; multi-page files become layers.

use super::ImageDecoder;
use crate::color::{self, ColorSource, Layout, Samples};
use crate::{Decoded, DecodedImage, IoError};
use prisme_color::RgbSpace;
use std::io::Cursor;
use tiff::decoder::{Decoder, DecodingResult, Limits};
use tiff::tags::Tag;
use tiff::ColorType;

pub struct Tiff;

impl ImageDecoder for Tiff {
    fn name(&self) -> &'static str {
        "TIFF"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["tif", "tiff"]
    }

    fn sniff(&self, h: &[u8]) -> bool {
        is_tiff_header(h)
    }

    fn decode(&self, bytes: &[u8]) -> Result<Decoded, IoError> {
        decode_tiff(bytes)
    }
}

pub(crate) fn is_tiff_header(h: &[u8]) -> bool {
    h.starts_with(b"II*\0") || h.starts_with(b"MM\0*") || h.starts_with(b"II+\0") || h.starts_with(b"MM\0+")
}

fn decode_tiff(bytes: &[u8]) -> Result<Decoded, IoError> {
    let mut dec = Decoder::new(Cursor::new(bytes)).map_err(IoError::decode)?.with_limits(Limits::unlimited());
    let mut pages = vec![];
    let mut first_err = None;
    let mut index = 0;
    loop {
        index += 1;
        // Skip reduced-resolution pages (thumbnails / pyramids).
        let reduced = dec.find_tag_unsigned::<u32>(Tag::NewSubfileType).ok().flatten().is_some_and(|v| v & 1 != 0);
        if !reduced || pages.is_empty() && !dec.more_images() {
            match decode_page(&mut dec) {
                Ok(img) => {
                    let name = dec
                        .get_tag_ascii_string(Tag::Unknown(285))
                        .ok()
                        .map(|s| s.trim_end_matches('\0').trim().to_string())
                        .filter(|s| !s.is_empty())
                        .unwrap_or_else(|| format!("Page {index}"));
                    pages.push((name, img));
                }
                Err(e) => {
                    log::warn!("TIFF page {index}: {e}");
                    first_err.get_or_insert(e);
                }
            }
        }
        if !dec.more_images() || dec.next_image().is_err() {
            break;
        }
    }
    match pages.len() {
        0 => Err(first_err.unwrap_or_else(|| IoError::Decode("TIFF contains no image".into()))),
        1 => Ok(Decoded::Raster(pages.pop().expect("one page").1)),
        _ => Ok(Decoded::Layers(pages)),
    }
}

/// Normalised sample storage after reading a page.
enum Buf {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

impl Buf {
    fn samples(&self) -> Samples<'_> {
        match self {
            Buf::U8(v) => Samples::U8(v),
            Buf::U16(v) => Samples::U16(v),
            Buf::F32(v) => Samples::F32(v),
        }
    }
}

fn decode_page(dec: &mut Decoder<Cursor<&[u8]>>) -> Result<DecodedImage, IoError> {
    let (w, h) = dec.dimensions().map_err(IoError::decode)?;
    let ct = dec.colortype().map_err(IoError::decode)?;
    let extra = dec.find_tag_unsigned_vec::<u16>(Tag::ExtraSamples).ok().flatten().unwrap_or_default();
    let premultiplied = extra.first() == Some(&1);
    let (layout, bits) = match ct {
        ColorType::Gray(b) => (Layout::Gray, b),
        ColorType::GrayA(b) => (Layout::GrayAlpha, b),
        ColorType::Multiband { bit_depth, num_samples: 2 } => (Layout::GrayAlpha, bit_depth),
        ColorType::RGB(b) => (Layout::Rgb, b),
        ColorType::RGBA(b) => (Layout::Rgba, b),
        ColorType::CMYK(b) => (Layout::Cmyk, b),
        ColorType::CMYKA(b) => (Layout::Cmyka, b),
        other => return Err(IoError::Unsupported(format!("TIFF colour type {other:?}"))),
    };
    let icc = dec.get_tag_u8_vec(Tag::IccProfile).ok();
    let orientation = dec.find_tag_unsigned::<u16>(Tag::Orientation).ok().flatten();
    let data = dec.read_image().map_err(IoError::decode)?;
    let ch = layout.channels();
    let n = w as usize * h as usize * ch;
    let float = matches!(data, DecodingResult::F16(_) | DecodingResult::F32(_) | DecodingResult::F64(_));
    let to_f32 = |it: &mut dyn Iterator<Item = f32>| -> Vec<f32> { it.collect() };
    let buf = match data {
        DecodingResult::U8(v) if bits < 8 => Buf::U8(unpack_bits(&v, w as usize, h as usize, ch, bits)),
        DecodingResult::U8(v) => Buf::U8(v),
        DecodingResult::U16(v) => Buf::U16(v),
        DecodingResult::U32(v) => Buf::F32(to_f32(&mut v.iter().map(|&x| x as f32 / u32::MAX as f32))),
        DecodingResult::U64(v) => Buf::F32(to_f32(&mut v.iter().map(|&x| (x as f64 / u64::MAX as f64) as f32))),
        DecodingResult::I8(v) => Buf::F32(to_f32(&mut v.iter().map(|&x| (x as f32 / i8::MAX as f32).max(0.0)))),
        DecodingResult::I16(v) => Buf::F32(to_f32(&mut v.iter().map(|&x| (x as f32 / i16::MAX as f32).max(0.0)))),
        DecodingResult::I32(v) => Buf::F32(to_f32(&mut v.iter().map(|&x| (x as f32 / i32::MAX as f32).max(0.0)))),
        DecodingResult::I64(v) => Buf::F32(to_f32(&mut v.iter().map(|&x| (x as f64 / i64::MAX as f64).max(0.0) as f32))),
        DecodingResult::F16(v) => Buf::F32(to_f32(&mut v.iter().map(|x| x.to_f32()))),
        DecodingResult::F32(v) => Buf::F32(v),
        DecodingResult::F64(v) => Buf::F32(to_f32(&mut v.iter().map(|&x| x as f32))),
    };
    let len = match &buf {
        Buf::U8(v) => v.len(),
        Buf::U16(v) => v.len(),
        Buf::F32(v) => v.len(),
    };
    if len < n {
        return Err(IoError::Decode(format!("TIFF sample buffer too small ({len} < {n})")));
    }
    let fallback = if float {
        (ColorSource::space(RgbSpace::LinearSrgb), "Linear Rec.709 (untagged float)".to_string())
    } else {
        color::untagged_srgb()
    };
    let (mut source, mut desc) = icc.as_deref().and_then(ColorSource::from_icc).unwrap_or(fallback);
    if matches!(layout, Layout::Cmyk | Layout::Cmyka) && !matches!(&source, ColorSource::Icc(p) if p.is_cmyk()) {
        source = ColorSource::srgb();
        desc = "CMYK (untagged, naive conversion)".into();
    }
    let pixels = if premultiplied && layout.has_alpha() {
        // Associated alpha: un-premultiply the encoded values before colour conversion.
        let straight = color::unpremultiplied_samples(buf.samples(), layout, n);
        color::to_working(Samples::F32(&straight), w, h, layout, &source)
    } else {
        color::to_working(buf.samples(), w, h, layout, &source)
    };
    let src_bits = if float { bits.max(16) } else { bits };
    Ok(DecodedImage::new(w, h, pixels, desc, src_bits, float).oriented(orientation))
}

/// Expand packed sub-byte samples (rows padded to whole bytes) to full 8-bit range.
fn unpack_bits(data: &[u8], w: usize, h: usize, ch: usize, bits: u8) -> Vec<u8> {
    let bits = bits as usize;
    let per_row = w * ch;
    let stride = (per_row * bits).div_ceil(8);
    let max = (1u32 << bits) - 1;
    let mut out = Vec::with_capacity(per_row * h);
    for y in 0..h {
        let row = data.get(y * stride..(y + 1) * stride).unwrap_or(&[]);
        for i in 0..per_row {
            let bit = i * bits;
            let byte = row.get(bit / 8).copied().unwrap_or(0) as u32;
            let v = (byte >> (8 - bits - bit % 8)) & max;
            out.push((v * 255 / max) as u8);
        }
    }
    out
}
