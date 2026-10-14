//! Formats decoded through the `image` crate: WebP, GIF (first frame), BMP, ICO, TGA,
//! QOI, PNM and Radiance HDR.

use super::ImageDecoder;
use crate::color::{self, ColorSource, Layout, Samples};
use crate::{Decoded, DecodedImage, IoError};
use image::{DynamicImage, ImageDecoder as _, ImageFormat, ImageReader};
use prisme_color::RgbSpace;
use std::io::Cursor;

pub struct Generic {
    name: &'static str,
    format: ImageFormat,
    exts: &'static [&'static str],
    sniff: fn(&[u8]) -> bool,
}

pub const WEBP: Generic = Generic {
    name: "WebP",
    format: ImageFormat::WebP,
    exts: &["webp"],
    sniff: |h| h.len() >= 12 && &h[..4] == b"RIFF" && &h[8..12] == b"WEBP",
};
pub const GIF: Generic =
    Generic { name: "GIF", format: ImageFormat::Gif, exts: &["gif"], sniff: |h| h.starts_with(b"GIF8") };
pub const BMP: Generic =
    Generic { name: "BMP", format: ImageFormat::Bmp, exts: &["bmp", "dib"], sniff: |h| h.starts_with(b"BM") };
pub const ICO: Generic = Generic {
    name: "ICO",
    format: ImageFormat::Ico,
    exts: &["ico", "cur"],
    sniff: |h| h.starts_with(&[0, 0, 1, 0]) || h.starts_with(&[0, 0, 2, 0]),
};
pub const QOI: Generic =
    Generic { name: "QOI", format: ImageFormat::Qoi, exts: &["qoi"], sniff: |h| h.starts_with(b"qoif") };
pub const HDR: Generic = Generic {
    name: "Radiance HDR",
    format: ImageFormat::Hdr,
    exts: &["hdr", "rgbe", "pic"],
    sniff: |h| h.starts_with(b"#?RADIANCE") || h.starts_with(b"#?RGBE"),
};
pub const PNM: Generic = Generic {
    name: "PNM",
    format: ImageFormat::Pnm,
    exts: &["pnm", "pbm", "pgm", "ppm", "pam"],
    sniff: |h| h.len() >= 3 && h[0] == b'P' && (b'1'..=b'7').contains(&h[1]) && h[2].is_ascii_whitespace(),
};
pub const TGA: Generic = Generic { name: "TGA", format: ImageFormat::Tga, exts: &["tga", "icb", "vda", "vst"], sniff: |_| false };

impl ImageDecoder for Generic {
    fn name(&self) -> &'static str {
        self.name
    }

    fn extensions(&self) -> &'static [&'static str] {
        self.exts
    }

    fn sniff(&self, header: &[u8]) -> bool {
        (self.sniff)(header)
    }

    fn decode(&self, bytes: &[u8]) -> Result<Decoded, IoError> {
        let mut reader = ImageReader::with_format(Cursor::new(bytes), self.format);
        reader.no_limits();
        let mut dec = reader.into_decoder().map_err(IoError::decode)?;
        let icc = dec.icc_profile().ok().flatten();
        let orientation = dec.exif_metadata().ok().flatten().and_then(|e| color::exif_orientation(&e));
        let img = DynamicImage::from_decoder(dec).map_err(IoError::decode)?;
        let hdr = self.format == ImageFormat::Hdr;
        let (source, desc) = if hdr {
            (ColorSource::space(RgbSpace::LinearSrgb), "Linear Rec.709 (Radiance HDR)".to_string())
        } else {
            icc.as_deref().and_then(ColorSource::from_icc).unwrap_or_else(color::untagged_srgb)
        };
        Ok(Decoded::Raster(dynamic_to_decoded(img, &source, desc).oriented(orientation)))
    }
}

/// Convert an `image::DynamicImage` into a working-space image.
pub(crate) fn dynamic_to_decoded(img: DynamicImage, source: &ColorSource, desc: String) -> DecodedImage {
    let (w, h) = (img.width(), img.height());
    let run = |s: Samples<'_>, l: Layout| color::to_working(s, w, h, l, source);
    let (pixels, bits, float) = match &img {
        DynamicImage::ImageLuma8(b) => (run(Samples::U8(b.as_raw()), Layout::Gray), 8, false),
        DynamicImage::ImageLumaA8(b) => (run(Samples::U8(b.as_raw()), Layout::GrayAlpha), 8, false),
        DynamicImage::ImageRgb8(b) => (run(Samples::U8(b.as_raw()), Layout::Rgb), 8, false),
        DynamicImage::ImageRgba8(b) => (run(Samples::U8(b.as_raw()), Layout::Rgba), 8, false),
        DynamicImage::ImageLuma16(b) => (run(Samples::U16(b.as_raw()), Layout::Gray), 16, false),
        DynamicImage::ImageLumaA16(b) => (run(Samples::U16(b.as_raw()), Layout::GrayAlpha), 16, false),
        DynamicImage::ImageRgb16(b) => (run(Samples::U16(b.as_raw()), Layout::Rgb), 16, false),
        DynamicImage::ImageRgba16(b) => (run(Samples::U16(b.as_raw()), Layout::Rgba), 16, false),
        DynamicImage::ImageRgb32F(b) => (run(Samples::F32(b.as_raw()), Layout::Rgb), 32, true),
        DynamicImage::ImageRgba32F(b) => (run(Samples::F32(b.as_raw()), Layout::Rgba), 32, true),
        other => {
            let b = other.to_rgba32f();
            (run(Samples::F32(b.as_raw()), Layout::Rgba), 32, true)
        }
    };
    DecodedImage::new(w, h, pixels, desc, bits, float)
}
