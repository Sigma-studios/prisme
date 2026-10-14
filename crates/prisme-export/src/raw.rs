//! Choosing the PNG colour type / bit depth and producing packed raw scanline data.

use crate::compress::cicp_codes;
use crate::convert::{encode_pixels, exact_palette, stats, to_u8, to_u16};
use crate::quantize::{gray_levels, quantize_palette};
use crate::{BitDepth, ColorMode, ColorTag, ExportError, PngOptions};
use prisme_color::{RgbSpace, Transfer};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RawColor {
    Gray,
    GrayAlpha,
    Rgb,
    Rgba,
    /// Palette entries (straight alpha); entries with alpha < 255 come first.
    Indexed(Vec<[u8; 4]>),
}

impl RawColor {
    pub fn channels(&self) -> usize {
        match self {
            RawColor::Gray | RawColor::Indexed(_) => 1,
            RawColor::GrayAlpha => 2,
            RawColor::Rgb => 3,
            RawColor::Rgba => 4,
        }
    }

    pub fn png_code(&self) -> u8 {
        match self {
            RawColor::Gray => 0,
            RawColor::Rgb => 2,
            RawColor::Indexed(_) => 3,
            RawColor::GrayAlpha => 4,
            RawColor::Rgba => 6,
        }
    }
}

/// Packed, unfiltered PNG scanlines (no filter bytes), 16-bit samples big-endian.
#[derive(Clone, Debug)]
pub(crate) struct Raw {
    pub w: u32,
    pub h: u32,
    pub color: RawColor,
    pub depth: u8,
    pub data: Vec<u8>,
}

impl Raw {
    pub fn row_bytes(&self) -> usize {
        (self.w as usize * self.color.channels() * self.depth as usize).div_ceil(8)
    }
}

pub(crate) struct Built {
    pub raw: Raw,
    pub allow_depth_reduction: bool,
    pub allow_type_reduction: bool,
    pub allow_gray: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct EncodeCfg {
    pub bit_depth: BitDepth,
    pub color_mode: ColorMode,
    pub palette_colors: u16,
    pub dither: f32,
    pub space: RgbSpace,
    pub tag: ColorTag,
    pub clean_transparent: bool,
    pub effort: u8,
}

impl EncodeCfg {
    pub fn from_opts(o: &PngOptions) -> Self {
        Self {
            bit_depth: o.bit_depth,
            color_mode: o.color_mode,
            palette_colors: o.palette_colors.clamp(2, 256),
            dither: o.dither,
            space: o.target_space,
            tag: o.color_tag,
            clean_transparent: o.clean_transparent,
            effort: o.effort.min(6),
        }
    }
}

/// Whether a grayscale PNG can still be tagged correctly with `tag` for `space`.
/// Gray PNGs cannot carry an RGB ICC profile; for sRGB-transfer spaces (sRGB, Display P3, whose
/// neutral axes are identical) we can use an sRGB chunk instead, and cICP works for gray.
pub(crate) fn gray_tag_ok(tag: ColorTag, space: RgbSpace) -> bool {
    let srgb_like = space.transfer() == Transfer::Srgb;
    match tag {
        ColorTag::None => true,
        ColorTag::Cicp => cicp_codes(space).is_some() || srgb_like,
        ColorTag::SrgbChunk | ColorTag::Icc => srgb_like,
    }
}

fn pack_bits(samples: &[u8], w: u32, depth: u8) -> Vec<u8> {
    let w = w as usize;
    let d = depth as usize;
    let row_bytes = (w * d).div_ceil(8);
    let rows = samples.len() / w.max(1);
    let mut out = vec![0u8; row_bytes * rows];
    for (srow, orow) in samples.chunks(w).zip(out.chunks_mut(row_bytes)) {
        for (x, &s) in srow.iter().enumerate() {
            let bit = x * d;
            orow[bit / 8] |= s << (8 - d - bit % 8);
        }
    }
    out
}

fn min_depth_for(n: usize) -> u8 {
    match n {
        0..=2 => 1,
        3..=4 => 2,
        5..=16 => 4,
        _ => 8,
    }
}

#[allow(clippy::too_many_arguments)]
fn palette_raw(
    q8: &[[u8; 4]],
    w: u32,
    h: u32,
    k: u16,
    dither: f32,
    effort: u8,
    depth: Option<u8>,
    notes: &mut Vec<String>,
) -> Result<Raw, ExportError> {
    let (pal, idx) = match exact_palette(q8, k as usize) {
        Some(p) => p,
        None => {
            let r = quantize_palette(q8, w, h, k, dither, effort)?;
            notes.push(format!("Quantized to {} colours (dither {:.2})", r.0.len(), dither.clamp(0.0, 1.0)));
            r
        }
    };
    Ok(indexed_raw(pal, idx, w, h, depth))
}

fn indexed_raw(pal: Vec<[u8; 4]>, idx: Vec<u8>, w: u32, h: u32, depth: Option<u8>) -> Raw {
    // Put translucent entries first so tRNS stays short.
    let mut order: Vec<usize> = (0..pal.len()).collect();
    order.sort_by_key(|&i| (pal[i][3] == 255) as u8);
    let mut remap = vec![0u8; pal.len().max(1)];
    for (new, &old) in order.iter().enumerate() {
        remap[old] = new as u8;
    }
    let pal: Vec<[u8; 4]> = order.iter().map(|&i| pal[i]).collect();
    let idx: Vec<u8> = idx.iter().map(|&i| remap[i as usize]).collect();
    let depth = depth.unwrap_or_else(|| min_depth_for(pal.len())).max(min_depth_for(pal.len()));
    let data = if depth == 8 { idx } else { pack_bits(&idx, w, depth) };
    Raw { w, h, color: RawColor::Indexed(pal), depth, data }
}

fn pack8(q: &[[u8; 4]], w: u32, h: u32, color: RawColor) -> Raw {
    let ch = color.channels();
    let mut data = Vec::with_capacity(q.len() * ch);
    match color {
        RawColor::Gray => data.extend(q.iter().map(|p| p[0])),
        RawColor::GrayAlpha => q.iter().for_each(|p| data.extend([p[0], p[3]])),
        RawColor::Rgb => q.iter().for_each(|p| data.extend([p[0], p[1], p[2]])),
        _ => q.iter().for_each(|p| data.extend(*p)),
    }
    Raw { w, h, color, depth: 8, data }
}

fn pack16(q: &[[u16; 4]], w: u32, h: u32, color: RawColor) -> Raw {
    let sel: &[usize] = match color {
        RawColor::Gray => &[0],
        RawColor::GrayAlpha => &[0, 3],
        RawColor::Rgb => &[0, 1, 2],
        _ => &[0, 1, 2, 3],
    };
    let mut data = Vec::with_capacity(q.len() * sel.len() * 2);
    for p in q {
        for &c in sel {
            data.extend(p[c].to_be_bytes());
        }
    }
    Raw { w, h, color, depth: 16, data }
}

/// Build the raw image for the given settings.
pub(crate) fn build_raw(
    px: &[f32],
    w: u32,
    h: u32,
    cfg: &EncodeCfg,
    notes: &mut Vec<String>,
) -> Result<Built, ExportError> {
    let mut depth = cfg.bit_depth;
    let mode = cfg.color_mode;
    let low = matches!(depth, BitDepth::One | BitDepth::Two | BitDepth::Four);
    if depth == BitDepth::Sixteen && mode == ColorMode::Palette {
        notes.push("Palette images cannot be 16-bit; using 8-bit".into());
        depth = BitDepth::Eight;
    }
    if low && matches!(mode, ColorMode::Rgb | ColorMode::Rgba | ColorMode::GrayAlpha) {
        notes.push(format!("{mode:?} does not support bit depths below 8; using 8-bit"));
        depth = BitDepth::Eight;
    }
    let gray_ok = gray_tag_ok(cfg.tag, cfg.space);
    if matches!(mode, ColorMode::Gray | ColorMode::GrayAlpha) && !gray_ok {
        notes.push(format!(
            "Grayscale PNG cannot carry an RGB ICC profile for {}; the colour tag will fall back",
            cfg.space.name()
        ));
    }
    let flatten = matches!(mode, ColorMode::Rgb | ColorMode::Gray);
    let to_gray = matches!(mode, ColorMode::Gray | ColorMode::GrayAlpha);
    let enc = encode_pixels(px, w, cfg.space, to_gray, flatten);
    if flatten && px.chunks_exact(4).any(|p| p[3] < 1.0) {
        notes.push("Transparency flattened onto white".into());
    }
    let built = |raw: Raw| Built {
        raw,
        allow_depth_reduction: matches!(cfg.bit_depth, BitDepth::Auto | BitDepth::Sixteen),
        allow_type_reduction: mode == ColorMode::Auto,
        allow_gray: mode == ColorMode::Auto && gray_ok,
    };

    if depth == BitDepth::Sixteen {
        let q = to_u16(&enc, cfg.clean_transparent);
        drop(enc);
        let st = stats(&q, u16::MAX);
        let color = match mode {
            ColorMode::Auto if st.gray && gray_ok => {
                if st.opaque { RawColor::Gray } else { RawColor::GrayAlpha }
            }
            ColorMode::Auto => {
                if st.opaque { RawColor::Rgb } else { RawColor::Rgba }
            }
            ColorMode::Rgba => RawColor::Rgba,
            ColorMode::Rgb => RawColor::Rgb,
            ColorMode::GrayAlpha => RawColor::GrayAlpha,
            _ => RawColor::Gray,
        };
        return Ok(built(pack16(&q, w, h, color)));
    }

    let q8 = to_u8(&enc, cfg.clean_transparent);
    let st = stats(&q8, 255u8);

    if let BitDepth::One | BitDepth::Two | BitDepth::Four = depth {
        let d: u8 = match depth {
            BitDepth::One => 1,
            BitDepth::Two => 2,
            _ => 4,
        };
        let levels = 1u32 << d;
        let gray_path = mode == ColorMode::Gray || (mode == ColorMode::Auto && st.gray && st.opaque && gray_ok);
        if gray_path {
            let g: Vec<f32> = enc.chunks_exact(4).map(|p| p[0]).collect();
            let lv = gray_levels(&g, w, levels, cfg.dither);
            return Ok(built(Raw { w, h, color: RawColor::Gray, depth: d, data: pack_bits(&lv, w, d) }));
        }
        let k = if mode == ColorMode::Palette { cfg.palette_colors.min(levels as u16) } else { levels as u16 };
        let raw = palette_raw(&q8, w, h, k, cfg.dither, cfg.effort, Some(d), notes)?;
        return Ok(built(raw));
    }

    let explicit8 = depth == BitDepth::Eight;
    let raw = match mode {
        ColorMode::Auto => {
            if st.gray && st.opaque && gray_ok {
                // Exact low bit-depth gray when every value is representable.
                let mut d = 8u8;
                if !explicit8 {
                    for cand in [1u8, 2, 4] {
                        let step = 255 / ((1u16 << cand) - 1) as u8;
                        if q8.iter().all(|p| p[0] % step == 0) {
                            d = cand;
                            break;
                        }
                    }
                }
                if d < 8 {
                    let step = 255 / ((1u16 << d) - 1) as u8;
                    let lv: Vec<u8> = q8.iter().map(|p| p[0] / step).collect();
                    Raw { w, h, color: RawColor::Gray, depth: d, data: pack_bits(&lv, w, d) }
                } else {
                    pack8(&q8, w, h, RawColor::Gray)
                }
            } else if let Some((pal, idx)) = exact_palette(&q8, 256) {
                indexed_raw(pal, idx, w, h, if explicit8 { Some(8) } else { None })
            } else if st.gray && gray_ok {
                pack8(&q8, w, h, RawColor::GrayAlpha)
            } else if st.opaque {
                pack8(&q8, w, h, RawColor::Rgb)
            } else {
                pack8(&q8, w, h, RawColor::Rgba)
            }
        }
        ColorMode::Rgba => pack8(&q8, w, h, RawColor::Rgba),
        ColorMode::Rgb => pack8(&q8, w, h, RawColor::Rgb),
        ColorMode::GrayAlpha => pack8(&q8, w, h, RawColor::GrayAlpha),
        ColorMode::Gray => pack8(&q8, w, h, RawColor::Gray),
        ColorMode::Palette => palette_raw(
            &q8,
            w,
            h,
            cfg.palette_colors,
            cfg.dither,
            cfg.effort,
            if explicit8 { Some(8) } else { None },
            notes,
        )?,
    };
    Ok(built(raw))
}
