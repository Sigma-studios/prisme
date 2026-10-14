//! Camera RAW through `rawler` (pure Rust).
//!
//! rawler provides sensor decoding, black/white level scaling, PPG (Bayer) / bilinear
//! (X-Trans, 4-colour) demosaicing and crops. White balance and the camera matrix are
//! applied here, going camera RGB -> XYZ (D65) -> linear Rec.2020 without gamut clipping.

use super::generic::dynamic_to_decoded;
use super::ImageDecoder;
use crate::color::{self, ColorSource};
use crate::{Decoded, DecodedImage, IoError};
use rawler::decoders::RawDecodeParams;
use rawler::imgop::chromatic_adaption::adapt_bradford;
use rawler::imgop::develop::{Intermediate, ProcessingStep, RawDevelop};
use rawler::imgop::matrix::{multiply, normalize, pseudo_inverse};
use rawler::imgop::xyz::Illuminant;
use rawler::rawsource::RawSource;
use rawler::RawImage;

pub struct Raw;

const EXTS: &[&str] = &[
    "cr2", "cr3", "crw", "nef", "nrw", "arw", "srf", "sr2", "raf", "orf", "rw2", "rwl", "raw", "dng", "pef", "srw",
    "3fr", "fff", "iiq", "mos", "mef", "mrw", "erf", "kdc", "dcr", "dcs", "ari", "x3f",
];

impl ImageDecoder for Raw {
    fn name(&self) -> &'static str {
        "Camera RAW"
    }

    fn extensions(&self) -> &'static [&'static str] {
        EXTS
    }

    fn sniff(&self, h: &[u8]) -> bool {
        // TIFF-based RAWs (CR2, NEF, ARW, DNG, PEF ...) are only taken over the TIFF
        // decoder when the extension says RAW (see `crate::candidates`).
        super::tiff::is_tiff_header(h)
            || h.starts_with(b"FUJIFILMCCD-RAW")
            || h.starts_with(b"IIRO")
            || h.starts_with(b"IIRS")
            || h.starts_with(b"MMOR")
            || h.starts_with(b"IIU\0")
            || (h.len() >= 14 && &h[6..14] == b"HEAPCCDR")
            || super::ftyp_brands(h).iter().any(|b| b == b"crx ")
            || h.starts_with(b"\0MRM")
            || h.starts_with(b"FOVb")
    }

    fn decode(&self, bytes: &[u8]) -> Result<Decoded, IoError> {
        guarded(|| decode_raw(bytes)).map(Decoded::Raster)
    }

    fn decode_preview(&self, bytes: &[u8], max_side: u32) -> Option<DecodedImage> {
        guarded(|| preview_raw(bytes, max_side).ok_or_else(|| IoError::Decode("no preview".into()))).ok()
    }
}

/// rawler may panic on malformed files; turn panics into errors where unwinding exists.
fn guarded<T>(f: impl FnOnce() -> Result<T, IoError>) -> Result<T, IoError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|_| Err(IoError::Decode("camera RAW decoder panicked".into())))
}

fn raw_err(e: rawler::RawlerError) -> IoError {
    IoError::Decode(format!("RAW: {e}"))
}

/// Camera -> working space matrix (rows = working RGB, columns = camera channels).
fn cam_to_working<const N: usize>(raw: &RawImage) -> [[f32; N]; 3] {
    let working_to_xyz = prisme_color::working_to_xyz().0.map(|r| r.map(|v| v as f32));
    let (illu, m) = raw
        .color_matrix_find_first([
            Illuminant::D65,
            Illuminant::D55,
            Illuminant::D50,
            Illuminant::D75,
            Illuminant::Daylight,
            Illuminant::A,
            Illuminant::B,
            Illuminant::C,
            Illuminant::Flash,
        ])
        .unwrap_or((Illuminant::D65, vec![]));
    let mut xyz2cam = [[0.0f32; 3]; N];
    let rows = m.len() / 3;
    if rows == 3 && illu != Illuminant::D65 {
        let a = adapt_bradford(&illu, &Illuminant::D65, &[[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]]);
        xyz2cam[..3].copy_from_slice(&a);
    } else if rows >= 3 {
        for (row, src) in xyz2cam.iter_mut().zip(m.chunks_exact(3)) {
            row.copy_from_slice(src);
        }
    } else {
        log::warn!("RAW without colour matrix; assuming camera RGB = XYZ");
        for (i, row) in xyz2cam.iter_mut().enumerate().take(3) {
            row[i] = 1.0;
        }
    }
    let rgb2cam = normalize(multiply(&xyz2cam, &working_to_xyz));
    pseudo_inverse(rgb2cam)
}

/// White balance multipliers normalised so the smallest is 1 (dcraw convention).
fn wb_coeffs(raw: &RawImage, n: usize) -> [f32; 4] {
    let mut wb = raw.wb_coeffs;
    if wb.iter().take(n).any(|v| !v.is_finite() || *v <= 0.0) {
        wb = [1.0; 4];
    }
    let min = wb.iter().take(n).copied().fold(f32::INFINITY, f32::min);
    wb.map(|v| v / min)
}

fn map_pixels<const N: usize>(data: &[[f32; N]], w: u32, wb: [f32; 4], m: [[f32; N]; 3]) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len() * 4];
    let wu = w as usize;
    color::for_row_blocks(&mut out, wu * 4, |row0, block| {
        let src = &data[row0 * wu..];
        for (px, s) in block.chunks_exact_mut(4).zip(src) {
            let mut cam = [0.0f32; N];
            for c in 0..N {
                // Clip at the WB-scaled white so blown highlights stay neutral.
                cam[c] = (s[c] * wb[c]).clamp(0.0, 1.0);
            }
            for (r, row) in m.iter().enumerate() {
                px[r] = row.iter().zip(&cam).map(|(a, b)| a * b).sum();
            }
            px[3] = 1.0;
        }
    });
    out
}

fn decode_raw(bytes: &[u8]) -> Result<DecodedImage, IoError> {
    let src = RawSource::new_from_slice(bytes);
    let dec = rawler::get_decoder(&src).map_err(raw_err)?;
    let params = RawDecodeParams::default();
    let raw = dec.raw_image(&src, &params, false).map_err(raw_err)?;
    let develop = RawDevelop::new_with(&[
        ProcessingStep::Rescale,
        ProcessingStep::Demosaic,
        ProcessingStep::FujiRotate,
        ProcessingStep::CropActiveArea,
        ProcessingStep::CropDefault,
    ]);
    let inter = develop.develop_intermediate(&raw).map_err(raw_err)?;
    let dim = inter.dim();
    let (w, h) = (dim.w as u32, dim.h as u32);
    let pixels = match &inter {
        Intermediate::Monochrome(p) => p.data.iter().flat_map(|&v| [v, v, v, 1.0]).collect(),
        Intermediate::ThreeColor(p) => map_pixels(&p.data, w, wb_coeffs(&raw, 3), cam_to_working::<3>(&raw)),
        Intermediate::FourColor(p) => map_pixels(&p.data, w, wb_coeffs(&raw, 4), cam_to_working::<4>(&raw)),
    };
    let desc = format!("Camera RAW ({} {})", raw.clean_make, raw.clean_model).replace("( ", "(").replace(" )", ")");
    let orientation = Some(raw.orientation.to_u16());
    Ok(DecodedImage::new(w, h, pixels, desc, raw.bps.clamp(8, 32) as u8, true).oriented(orientation))
}

/// Embedded JPEG preview (or thumbnail) of the RAW file.
fn preview_raw(bytes: &[u8], max_side: u32) -> Option<DecodedImage> {
    let src = RawSource::new_from_slice(bytes);
    let dec = rawler::get_decoder(&src).ok()?;
    let params = RawDecodeParams::default();
    let img = dec
        .preview_image(&src, &params)
        .ok()
        .flatten()
        .or_else(|| dec.thumbnail_image(&src, &params).ok().flatten())
        .or_else(|| dec.full_image(&src, &params).ok().flatten())?;
    let orientation = dec.raw_image(&src, &params, true).ok().map(|r| r.orientation.to_u16());
    let img = img.thumbnail(max_side, max_side);
    let (source, desc) = (ColorSource::srgb(), "sRGB (embedded RAW preview)".to_string());
    Some(dynamic_to_decoded(img, &source, desc).oriented(orientation))
}
