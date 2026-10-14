//! PNG export for Prisme.
//!
//! Input is straight-alpha, scene-linear RGBA `f32` in the working space (linear Rec.2020, D65).
//! Pixels are converted to the chosen target space, reduced to the requested (or automatically
//! detected) colour type / bit depth, optionally palette-quantized with `quantette` in Oklab, and
//! compressed. On native targets compression is done by `oxipng` (libdeflate / zopfli); on
//! wasm32 a pure-Rust writer (`miniz_oxide` / `zopfli`) is used because libdeflate is C code.
//! Colour tagging chunks (iCCP / cICP / sRGB) are inserted after optimization so they always
//! survive.
//!
//! The crate also implements "export to a target file size" (palette reduction and/or
//! downscaling, searched with a fast effort level and re-encoded at the requested effort).

mod compress;
mod convert;
mod quantize;
mod raw;
mod resize;
mod target;
mod writer;

use serde::{Deserialize, Serialize};

pub use resize::resize_rgba;

/// Output bit depth (per sample). `Auto` = 8-bit, reduced losslessly where possible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BitDepth {
    Auto,
    One,
    Two,
    Four,
    Eight,
    Sixteen,
}

/// Output colour type. `Auto` detects opacity, grayness and small palettes losslessly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorMode {
    Auto,
    Rgba,
    Rgb,
    GrayAlpha,
    Gray,
    Palette,
}

/// PNG row filter strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterStrategy {
    /// Let the optimizer try the strategies of the effort preset.
    Auto,
    None,
    Sub,
    Up,
    Average,
    Paeth,
    MinSum,
    Entropy,
    /// Per-row trial compression (slow).
    Brute,
}

/// How the colour space is tagged in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorTag {
    /// iCCP chunk with the target space profile.
    Icc,
    /// cICP chunk (falls back to ICC for spaces without CICP codes).
    Cicp,
    /// sRGB chunk (only valid for an sRGB target; otherwise falls back to ICC).
    SrgbChunk,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResizeFilter {
    Lanczos3,
    Mitchell,
    Bilinear,
    Nearest,
}

/// What to sacrifice first when a target file size requires both colour and resolution loss.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetPriority {
    /// Reduce palette colours down to ~32 before shrinking.
    ColorsFirst,
    /// Shrink first (keeping up to 256 colours), then reduce colours.
    ResolutionFirst,
    /// Reduce both together.
    Balanced,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TargetSize {
    pub max_bytes: u64,
    pub allow_resize: bool,
    /// Smallest allowed scale factor (0.05..=1).
    pub min_scale: f32,
    pub priority: TargetPriority,
    pub allow_lossy_palette: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PngOptions {
    pub bit_depth: BitDepth,
    pub color_mode: ColorMode,
    /// 2..=256 when `color_mode == Palette`.
    pub palette_colors: u16,
    /// 0..=1 Floyd–Steinberg strength (palette and low bit-depth gray).
    pub dither: f32,
    /// Adam7 interlacing.
    pub interlace: bool,
    pub filter: FilterStrategy,
    /// 0 (fastest) ..= 6 (oxipng max preset).
    pub effort: u8,
    pub zopfli: bool,
    pub zopfli_iterations: u8,
    pub strip_metadata: bool,
    pub color_tag: ColorTag,
    pub target_space: prisme_color::RgbSpace,
    /// Zero RGB of fully transparent pixels for better compression.
    pub clean_transparent: bool,
    /// Explicit resize factor (1.0 = none).
    pub scale: f32,
    pub resize_filter: ResizeFilter,
    pub target: Option<TargetSize>,
}

impl Default for PngOptions {
    fn default() -> Self {
        Self {
            bit_depth: BitDepth::Auto,
            color_mode: ColorMode::Auto,
            palette_colors: 256,
            dither: 1.0,
            interlace: false,
            filter: FilterStrategy::Auto,
            effort: 2,
            zopfli: false,
            zopfli_iterations: 15,
            strip_metadata: true,
            color_tag: ColorTag::Icc,
            target_space: prisme_color::RgbSpace::Srgb,
            clean_transparent: true,
            scale: 1.0,
            resize_filter: ResizeFilter::Lanczos3,
            target: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExportResult {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Bit depth actually written (after lossless reductions).
    pub bit_depth: u8,
    /// Colour type actually written (never `Auto`).
    pub color_mode: ColorMode,
    pub palette_colors: Option<u16>,
    /// Total scale applied relative to the input (explicit scale × target-size scale).
    pub scale: f32,
    /// `Some(true/false)` when a target size was requested.
    pub met_target: Option<bool>,
    pub notes: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("export cancelled")]
    Cancelled,
    #[error("PNG encoding failed: {0}")]
    Encode(String),
    #[error("colour quantization failed: {0}")]
    Quantize(String),
}

pub(crate) fn validate(pixels: &[f32], width: u32, height: u32) -> Result<(), ExportError> {
    if width == 0 || height == 0 {
        return Err(ExportError::InvalidInput(format!("empty image {width}x{height}")));
    }
    if width > (1 << 24) || height > (1 << 24) {
        return Err(ExportError::InvalidInput(format!("image too large {width}x{height}")));
    }
    let expected = width as usize * height as usize * 4;
    if pixels.len() != expected {
        return Err(ExportError::InvalidInput(format!(
            "pixel buffer has {} floats, expected {expected} for {width}x{height} RGBA",
            pixels.len()
        )));
    }
    Ok(())
}

pub(crate) fn scaled_dims(width: u32, height: u32, scale: f32) -> (u32, u32) {
    let s = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
    let nw = ((width as f64 * s as f64).round() as u32).max(1);
    let nh = ((height as f64 * s as f64).round() as u32).max(1);
    (nw, nh)
}

/// Encode `pixels` as PNG.
///
/// `progress(fraction, message)` may be called often; `cancel()` returning true aborts with
/// [`ExportError::Cancelled`].
pub fn encode_png(
    pixels: &[f32],
    width: u32,
    height: u32,
    opts: &PngOptions,
    progress: &mut dyn FnMut(f32, &str),
    cancel: &dyn Fn() -> bool,
) -> Result<ExportResult, ExportError> {
    validate(pixels, width, height)?;
    let check = || if cancel() { Err(ExportError::Cancelled) } else { Ok(()) };
    check()?;
    progress(0.0, "Preparing");

    // Explicit resize first (in linear light, premultiplied).
    let scale = if opts.scale.is_finite() && opts.scale > 0.0 { opts.scale } else { 1.0 };
    let (nw, nh) = scaled_dims(width, height, scale);
    let resized;
    let (src, w, h): (&[f32], u32, u32) = if (nw, nh) != (width, height) {
        progress(0.02, "Resizing");
        resized = resize_rgba(pixels, width, height, nw, nh, opts.resize_filter);
        check()?;
        (&resized, nw, nh)
    } else {
        (pixels, width, height)
    };
    let eff_scale = if (nw, nh) != (width, height) { nw as f32 / width as f32 } else { 1.0 };

    if let Some(t) = &opts.target {
        let mut r = target::search(src, w, h, opts, t, progress, cancel)?;
        r.scale *= eff_scale;
        progress(1.0, "Done");
        return Ok(r);
    }

    let mut notes = Vec::new();
    let cfg = raw::EncodeCfg::from_opts(opts);
    progress(0.08, "Converting colours");
    let built = raw::build_raw(src, w, h, &cfg, &mut notes)?;
    check()?;
    progress(0.35, "Compressing");
    let params = compress::CompressParams::from_opts(opts, &built);
    let bytes = compress::compress(&built.raw, &params)?;
    check()?;
    progress(0.95, "Tagging");
    let out = compress::finalize(bytes, opts, &mut notes)?;
    progress(1.0, "Done");
    Ok(ExportResult {
        width: w,
        height: h,
        bit_depth: out.bit_depth,
        color_mode: out.color_mode,
        palette_colors: out.palette_colors,
        bytes: out.bytes,
        scale: eff_scale,
        met_target: None,
        notes,
    })
}

/// Fast size estimate for live UI feedback (low-effort encode of a tile mosaic sampled from the
/// image, extrapolated to the full pixel count).
pub fn estimate_size(pixels: &[f32], width: u32, height: u32, opts: &PngOptions) -> u64 {
    if validate(pixels, width, height).is_err() {
        return 0;
    }
    let est = target::estimate(pixels, width, height, opts).unwrap_or(0);
    match &opts.target {
        Some(t) => est.min(t.max_bytes),
        None => est,
    }
}
