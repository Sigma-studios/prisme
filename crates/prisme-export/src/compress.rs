//! Compression back ends (oxipng on native, pure-Rust writer on wasm) and colour tagging.

use crate::raw::{Built, Raw};
use crate::{ColorMode, ColorTag, ExportError, FilterStrategy, PngOptions};
use prisme_color::{IccProfile, RgbSpace};

#[derive(Clone, Debug)]
#[cfg_attr(target_arch = "wasm32", allow(dead_code))] // reductions are oxipng-only
pub(crate) struct CompressParams {
    pub effort: u8,
    pub filter: FilterStrategy,
    pub interlace: bool,
    /// Zopfli iterations when enabled.
    pub zopfli: Option<u8>,
    pub optimize_alpha: bool,
    pub allow_depth_reduction: bool,
    pub allow_type_reduction: bool,
    pub allow_gray: bool,
}

impl CompressParams {
    pub fn from_opts(o: &PngOptions, b: &Built) -> Self {
        Self {
            effort: o.effort.min(6),
            filter: o.filter,
            interlace: o.interlace,
            zopfli: o.zopfli.then_some(o.zopfli_iterations.max(1)),
            optimize_alpha: o.clean_transparent,
            allow_depth_reduction: b.allow_depth_reduction,
            allow_type_reduction: b.allow_type_reduction,
            allow_gray: b.allow_gray,
        }
    }
}

/// CICP (colour primaries, transfer characteristics) codes from ITU-T H.273.
pub(crate) fn cicp_codes(space: RgbSpace) -> Option<(u8, u8)> {
    match space {
        RgbSpace::Srgb => Some((1, 13)),
        RgbSpace::LinearSrgb => Some((1, 8)),
        RgbSpace::DisplayP3 => Some((12, 13)),
        RgbSpace::Rec2020 => Some((9, 1)),
        RgbSpace::LinearRec2020 => Some((9, 8)),
        RgbSpace::Rec2100Pq => Some((9, 16)),
        RgbSpace::Rec2100Hlg => Some((9, 18)),
        RgbSpace::AdobeRgb | RgbSpace::ProPhoto => None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn compress(raw: &Raw, p: &CompressParams) -> Result<Vec<u8>, ExportError> {
    use crate::raw::RawColor;
    use oxipng::{BitDepth, ColorType, Deflater, FilterStrategy as F, IndexSet, Options, RGBA8, RowFilter, StripChunks};

    let ct = match &raw.color {
        RawColor::Gray => ColorType::Grayscale { transparent_shade: None },
        RawColor::Rgb => ColorType::RGB { transparent_color: None },
        RawColor::GrayAlpha => ColorType::GrayscaleAlpha,
        RawColor::Rgba => ColorType::RGBA,
        RawColor::Indexed(pal) => {
            ColorType::Indexed { palette: pal.iter().map(|c| RGBA8::new(c[0], c[1], c[2], c[3])).collect() }
        }
    };
    let depth = BitDepth::try_from(raw.depth).map_err(|e| ExportError::Encode(e.to_string()))?;
    let img = oxipng::RawImage::new(raw.w, raw.h, ct, depth, raw.data.clone())
        .map_err(|e| ExportError::Encode(e.to_string()))?;

    let mut o = Options::from_preset(p.effort);
    o.interlace = Some(p.interlace);
    o.optimize_alpha = p.optimize_alpha;
    o.bit_depth_reduction = p.allow_depth_reduction;
    o.color_type_reduction = p.allow_type_reduction;
    o.grayscale_reduction = p.allow_gray;
    o.palette_reduction = true;
    o.scale_16 = false;
    o.strip = StripChunks::All;
    let single = |f: RowFilter| -> IndexSet<F> { [F::Basic(f)].into_iter().collect() };
    match p.filter {
        FilterStrategy::Auto => {}
        FilterStrategy::None => o.filters = single(RowFilter::None),
        FilterStrategy::Sub => o.filters = single(RowFilter::Sub),
        FilterStrategy::Up => o.filters = single(RowFilter::Up),
        FilterStrategy::Average => o.filters = single(RowFilter::Average),
        FilterStrategy::Paeth => o.filters = single(RowFilter::Paeth),
        FilterStrategy::MinSum => o.filters = [F::MinSum].into_iter().collect(),
        FilterStrategy::Entropy => o.filters = [F::Entropy].into_iter().collect(),
        FilterStrategy::Brute => {
            let num_lines = if p.effort >= 5 { 8 } else { 4 };
            let level = if p.effort >= 6 { 5 } else { 1 };
            o.filters = [F::Brute { num_lines, level }].into_iter().collect();
        }
    }
    if let Some(it) = p.zopfli {
        o.deflater = Deflater::Zopfli(oxipng::ZopfliOptions {
            iteration_count: std::num::NonZeroU64::new(it.max(1) as u64).unwrap_or(std::num::NonZeroU64::MIN),
            ..Default::default()
        });
    }
    img.create_optimized_png(&o).map_err(|e| ExportError::Encode(e.to_string()))
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn compress(raw: &Raw, p: &CompressParams) -> Result<Vec<u8>, ExportError> {
    Ok(crate::writer::write_png(raw, p.interlace, p.filter, p.effort, p.zopfli))
}

pub(crate) fn chunk(name: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 12);
    out.extend((data.len() as u32).to_be_bytes());
    out.extend(name);
    out.extend(data);
    let mut h = crc32fast::Hasher::new();
    h.update(name);
    h.update(data);
    out.extend(h.finalize().to_be_bytes());
    out
}

/// Iterate (name, data, start offset, end offset) over the chunks of a PNG file.
pub(crate) fn chunks(png: &[u8]) -> impl Iterator<Item = ([u8; 4], &[u8], usize, usize)> {
    let mut pos = 8usize;
    std::iter::from_fn(move || {
        if pos + 12 > png.len() {
            return None;
        }
        let len = u32::from_be_bytes(png[pos..pos + 4].try_into().ok()?) as usize;
        let name: [u8; 4] = png[pos + 4..pos + 8].try_into().ok()?;
        let end = pos + 12 + len;
        if end > png.len() {
            return None;
        }
        let item = (name, &png[pos + 8..pos + 8 + len], pos, end);
        pos = end;
        Some(item)
    })
}

pub(crate) struct Final {
    pub bytes: Vec<u8>,
    pub bit_depth: u8,
    pub color_mode: ColorMode,
    pub palette_colors: Option<u16>,
}

fn iccp_chunk(space: RgbSpace) -> Option<Vec<u8>> {
    let icc = IccProfile::encode_standard(space)?;
    let mut data = b"ICC profile\0\0".to_vec();
    data.extend(miniz_oxide::deflate::compress_to_vec_zlib(&icc, 9));
    Some(chunk(b"iCCP", &data))
}

/// Inspect the encoded PNG, insert colour-tag (and optional text) chunks after IHDR.
pub(crate) fn finalize(png: Vec<u8>, o: &PngOptions, notes: &mut Vec<String>) -> Result<Final, ExportError> {
    let (_, ihdr, _, ihdr_end) =
        chunks(&png).next().filter(|c| &c.0 == b"IHDR").ok_or_else(|| ExportError::Encode("missing IHDR".into()))?;
    let bit_depth = ihdr[8];
    let ct = ihdr[9];
    let color_mode = match ct {
        0 => ColorMode::Gray,
        2 => ColorMode::Rgb,
        3 => ColorMode::Palette,
        4 => ColorMode::GrayAlpha,
        _ => ColorMode::Rgba,
    };
    let palette_colors =
        chunks(&png).find(|c| &c.0 == b"PLTE").map(|c| (c.1.len() / 3) as u16).filter(|_| ct == 3);
    let is_gray = ct == 0 || ct == 4;
    let space = o.target_space;

    let mut extra = Vec::new();
    let srgb_chunk = || chunk(b"sRGB", &[0]);
    let cicp_chunk = |(p, t): (u8, u8)| chunk(b"cICP", &[p, t, 0, 1]);
    let srgb_like = space.transfer() == prisme_color::Transfer::Srgb;
    match o.color_tag {
        ColorTag::None => {}
        ColorTag::SrgbChunk if space == RgbSpace::Srgb => extra.push(srgb_chunk()),
        ColorTag::Cicp if cicp_codes(space).is_some() => extra.extend(cicp_codes(space).map(cicp_chunk)),
        tag => {
            if tag != ColorTag::Icc {
                notes.push(format!("{tag:?} tag not available for {}; using ICC profile", space.name()));
            }
            if !is_gray {
                match iccp_chunk(space) {
                    Some(c) => extra.push(c),
                    None => notes.push("Could not encode ICC profile; file is untagged".into()),
                }
            } else if srgb_like {
                notes.push("Grayscale output: tagged with sRGB chunk instead of ICC (identical neutral axis)".into());
                extra.push(srgb_chunk());
            } else if let Some(c) = cicp_codes(space) {
                notes.push("Grayscale output: tagged with cICP instead of an RGB ICC profile".into());
                extra.push(cicp_chunk(c));
            } else {
                notes.push(format!("Grayscale output cannot be tagged as {}; file is untagged", space.name()));
            }
        }
    }
    if !o.strip_metadata {
        extra.push(chunk(b"tEXt", b"Software\0Prisme"));
    }
    if extra.is_empty() {
        return Ok(Final { bytes: png, bit_depth, color_mode, palette_colors });
    }
    let mut out = Vec::with_capacity(png.len() + extra.iter().map(Vec::len).sum::<usize>());
    out.extend_from_slice(&png[..ihdr_end]);
    extra.iter().for_each(|c| out.extend(c));
    out.extend_from_slice(&png[ihdr_end..]);
    Ok(Final { bytes: out, bit_depth, color_mode, palette_colors })
}
