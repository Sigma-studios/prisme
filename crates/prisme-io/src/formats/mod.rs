//! Decoder registry.

mod avif;
mod exr;
mod generic;
mod heic;
mod jpeg;
mod jxl;
mod png;
mod raw;
mod svg;
mod tiff;

use crate::{Decoded, DecodedImage, IoError};

/// A file format decoder.
pub trait ImageDecoder: Sync {
    fn name(&self) -> &'static str;
    /// Lowercase extensions (no dot) handled by this decoder.
    fn extensions(&self) -> &'static [&'static str];
    /// Magic-byte detection on the first bytes of the file (up to 64).
    fn sniff(&self, header: &[u8]) -> bool;
    fn matches(&self, header: &[u8], ext: &str) -> bool {
        self.sniff(header) || self.extensions().contains(&ext)
    }
    fn decode(&self, bytes: &[u8]) -> Result<Decoded, IoError>;
    /// Cheap low-resolution preview (longest side about `max_side`), if the format allows one.
    fn decode_preview(&self, _bytes: &[u8], _max_side: u32) -> Option<DecodedImage> {
        None
    }
}

/// All decoders in detection priority order (TIFF before camera RAW so that plain
/// TIFF-based files without a RAW extension go to the TIFF decoder).
pub fn decoders() -> &'static [&'static dyn ImageDecoder] {
    &[
        &png::Png,
        &jpeg::Jpeg,
        &generic::WEBP,
        &generic::GIF,
        &generic::BMP,
        &generic::ICO,
        &generic::QOI,
        &generic::HDR,
        &generic::PNM,
        &generic::TGA,
        &jxl::Jxl,
        &exr::Exr,
        &avif::Avif,
        &heic::Heic,
        &tiff::Tiff,
        &raw::Raw,
        &svg::Svg,
    ]
}

/// ISO-BMFF `ftyp` brands (major + compatible) of a file header.
pub(crate) fn ftyp_brands(header: &[u8]) -> Vec<[u8; 4]> {
    if header.len() < 16 || &header[4..8] != b"ftyp" {
        return vec![];
    }
    let size = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let end = size.min(header.len());
    let mut v = vec![[header[8], header[9], header[10], header[11]]];
    let mut o = 16;
    while o + 4 <= end {
        v.push([header[o], header[o + 1], header[o + 2], header[o + 3]]);
        o += 4;
    }
    v
}

#[cfg(test)]
pub(crate) fn avif_props_for_test(file: &[u8]) -> (Option<(u8, u8, u8, bool)>, u8) {
    let p = avif::item_props(file);
    (p.nclx, p.irot)
}
