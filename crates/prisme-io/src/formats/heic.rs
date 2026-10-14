//! HEIC/HEIF through the system libheif (`heic` cargo feature, native only).

use super::{ftyp_brands, ImageDecoder};
use crate::{Decoded, IoError};

pub struct Heic;

impl ImageDecoder for Heic {
    fn name(&self) -> &'static str {
        "HEIF"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["heic", "heif", "hif"]
    }

    fn sniff(&self, h: &[u8]) -> bool {
        let brands = ftyp_brands(h);
        let heif = [b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"hevm", b"hevs", b"mif1", b"msf1"];
        !brands.iter().any(|b| b == b"avif" || b == b"avis") && brands.iter().any(|b| heif.contains(&b))
    }

    #[cfg(all(feature = "heic", not(target_arch = "wasm32")))]
    fn decode(&self, bytes: &[u8]) -> Result<Decoded, IoError> {
        imp::decode(bytes).map(Decoded::Raster)
    }

    #[cfg(not(all(feature = "heic", not(target_arch = "wasm32"))))]
    fn decode(&self, _bytes: &[u8]) -> Result<Decoded, IoError> {
        Err(IoError::Unsupported(
            "HEIC/HEIF needs the system libheif: build prisme-io with the `heic` feature (native only)".into(),
        ))
    }
}

#[cfg(all(feature = "heic", not(target_arch = "wasm32")))]
mod imp {
    use crate::color::{self, cicp, ColorSource, Layout, Samples};
    use crate::{DecodedImage, IoError};
    use libheif_rs::{ColorSpace, HeifContext, LibHeif, RgbChroma};

    pub fn decode(bytes: &[u8]) -> Result<DecodedImage, IoError> {
        let lib = LibHeif::new();
        let ctx = HeifContext::read_from_bytes(bytes).map_err(IoError::decode)?;
        let handle = ctx.primary_image_handle().map_err(IoError::decode)?;
        let alpha = handle.has_alpha_channel();
        let bits = handle.luma_bits_per_pixel();
        let hdr = bits > 8;
        let chroma = match (hdr, alpha) {
            (false, false) => RgbChroma::Rgb,
            (false, true) => RgbChroma::Rgba,
            (true, false) => RgbChroma::HdrRgbLe,
            (true, true) => RgbChroma::HdrRgbaLe,
        };
        let (source, desc) = match handle.color_profile_raw().and_then(|p| ColorSource::from_icc(&p.data)) {
            Some(s) => s,
            None => handle
                .color_profile_nclx()
                .and_then(|n| cicp(n.color_primaries() as u8, n.transfer_characteristics() as u8))
                .map(|(a, d)| (ColorSource::Analytic(a), d))
                .unwrap_or_else(color::untagged_srgb),
        };
        // libheif applies irot/imir itself by default.
        let image = lib.decode(&handle, ColorSpace::Rgb(chroma), None).map_err(IoError::decode)?;
        let planes = image.planes();
        let plane = planes.interleaved.ok_or_else(|| IoError::Decode("HEIF: no interleaved plane".into()))?;
        let (w, h) = (plane.width, plane.height);
        let ch = if alpha { 4 } else { 3 };
        let layout = if alpha { Layout::Rgba } else { Layout::Rgb };
        let row_bytes = w as usize * ch * if hdr { 2 } else { 1 };
        let pixels = if hdr {
            let max = ((1u32 << bits.clamp(1, 16)) - 1) as f32;
            let mut v = Vec::with_capacity(w as usize * h as usize * ch);
            for y in 0..h as usize {
                let row = &plane.data[y * plane.stride..][..row_bytes];
                v.extend(row.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as f32 / max));
            }
            color::to_working(Samples::F32(&v), w, h, layout, &source)
        } else {
            let mut v = Vec::with_capacity(w as usize * h as usize * ch);
            for y in 0..h as usize {
                v.extend_from_slice(&plane.data[y * plane.stride..][..row_bytes]);
            }
            color::to_working(Samples::U8(&v), w, h, layout, &source)
        };
        Ok(DecodedImage::new(w, h, pixels, desc, bits, false))
    }
}
