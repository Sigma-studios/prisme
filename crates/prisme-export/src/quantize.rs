//! Palette quantization (quantette, Oklab) and low bit-depth gray dithering.

use crate::ExportError;
use palette::{Alpha, Oklab, Srgb};
use quantette::{
    ImageRef, PaletteBuf, PaletteSize, Pipeline, QuantizeMethod,
    color_map::{IndexedColorMap, NearestNeighborColorMap},
    color_space::{oklab_to_srgb8, srgb8_to_oklab},
    dither::FloydSteinberg,
    kmeans::{Kmeans, KmeansOptions},
    wu::{BinnerF32x3, WuF32x3},
};

type Oklaba = Alpha<Oklab<f32>, f32>;

fn qerr(e: impl std::fmt::Debug) -> ExportError {
    ExportError::Quantize(format!("{e:?}"))
}

fn ditherer(dither: f32) -> Option<FloydSteinberg> {
    let d = if dither.is_finite() { dither.clamp(0.0, 1.0) } else { 0.0 };
    if d <= 0.0 { None } else { FloydSteinberg::with_error_diffusion(d * FloydSteinberg::DEFAULT_ERROR_DIFFUSION) }
}

/// Quantize RGBA8 pixels to at most `k` palette entries (k in 2..=256).
/// Returns (palette RGBA8, indices).
pub(crate) fn quantize_palette(
    px: &[[u8; 4]],
    w: u32,
    h: u32,
    k: u16,
    dither: f32,
    effort: u8,
) -> Result<(Vec<[u8; 4]>, Vec<u8>), ExportError> {
    let k = k.clamp(2, 256);
    let has_alpha = px.iter().any(|p| p[3] != 255);
    if !has_alpha {
        quantize_opaque(px, w, h, k, dither, effort)
    } else {
        quantize_alpha(px, w, h, k, dither)
    }
}

fn quantize_opaque(
    px: &[[u8; 4]],
    w: u32,
    h: u32,
    k: u16,
    dither: f32,
    effort: u8,
) -> Result<(Vec<[u8; 4]>, Vec<u8>), ExportError> {
    let rgb: Vec<Srgb<u8>> = px.iter().map(|p| Srgb::new(p[0], p[1], p[2])).collect();
    let img = ImageRef::new(w, h, rgb.as_slice()).map_err(qerr)?;
    let method = if effort >= 4 { QuantizeMethod::kmeans() } else { QuantizeMethod::Wu };
    let pipeline = Pipeline::new()
        .palette_size(PaletteSize::try_from(k).map_err(qerr)?)
        .ditherer(ditherer(dither))
        .quantize_method(method);
    #[cfg(not(target_arch = "wasm32"))]
    let pipeline = pipeline.parallel(true);
    let indexed = pipeline.input_image(img).output_srgb8_indexed_image();
    let (pal, idx) = indexed.into_parts();
    let pal = pal.into_iter().map(|c| [c.red, c.green, c.blue, 255]).collect();
    Ok((pal, idx))
}

/// RGBA quantization: k-means in premultiplied Oklab + alpha, seeded by Wu's method, mapped with
/// 4D Floyd–Steinberg. Fully transparent pixels get a dedicated transparent entry.
fn quantize_alpha(
    px: &[[u8; 4]],
    w: u32,
    h: u32,
    k: u16,
    dither: f32,
) -> Result<(Vec<[u8; 4]>, Vec<u8>), ExportError> {
    let rgb: Vec<Srgb<u8>> = px.iter().map(|p| Srgb::new(p[0], p[1], p[2])).collect();
    let lab = srgb8_to_oklab(&rgb);
    drop(rgb);
    let feats: Vec<Oklaba> = lab
        .iter()
        .zip(px)
        .map(|(l, p)| {
            let a = p[3] as f32 / 255.0;
            Alpha { color: Oklab::new(l.l * a, l.a * a, l.b * a), alpha: a }
        })
        .collect();
    drop(lab);
    let has_transparent = px.iter().any(|p| p[3] == 0);
    let visible: Vec<Oklaba> = feats.iter().zip(px).filter(|(_, p)| p[3] != 0).map(|(f, _)| *f).collect();

    let mut palette4: Vec<Oklaba> = Vec::new();
    if !visible.is_empty() {
        let kv = (k - has_transparent as u16).max(1);
        let size = PaletteSize::try_from(kv).map_err(qerr)?;
        let prem_lab: Vec<Oklab> = visible.iter().map(|f| f.color).collect();
        let wu = WuF32x3::run_slice(&prem_lab, BinnerF32x3::oklab_from_srgb8()).map_err(qerr)?;
        let cmap = wu.color_map(size);
        let n = cmap.palette().len();
        let mut acc = vec![(0.0f64, 0u64); n];
        for (c, f) in prem_lab.iter().zip(&visible) {
            let i = cmap.palette_index(c) as usize;
            acc[i].0 += f.alpha as f64;
            acc[i].1 += 1;
        }
        let init: Vec<Oklaba> = cmap
            .palette()
            .iter()
            .zip(&acc)
            .map(|(c, (s, n))| Alpha { color: *c, alpha: if *n > 0 { (*s / *n as f64) as f32 } else { 1.0 } })
            .collect();
        drop(prem_lab);
        let init = PaletteBuf::new(init).map_err(qerr)?;
        let opts = KmeansOptions::new().sampling_factor(0.5).max_samples(1 << 18);
        let km = Kmeans::<Oklaba, f32, 4>::run_slice(&visible, init, opts).map_err(qerr)?;
        palette4 = km.into_palette().into_vec();
    }
    let transparent_index = if has_transparent || palette4.is_empty() {
        palette4.push(Alpha { color: Oklab::new(0.0, 0.0, 0.0), alpha: 0.0 });
        Some((palette4.len() - 1) as u8)
    } else {
        None
    };
    drop(visible);

    let map = NearestNeighborColorMap::new(PaletteBuf::new(palette4.clone()).map_err(qerr)?);
    let mut indices = match ditherer(dither) {
        Some(fs) => {
            let img = ImageRef::new(w, h, feats.as_slice()).map_err(qerr)?;
            fs.dither_image_to_indexed(img, &map).into_parts().1
        }
        None => map.map_to_indices(&feats),
    };
    if let Some(t) = transparent_index {
        for (i, p) in indices.iter_mut().zip(px) {
            if p[3] == 0 {
                *i = t;
            }
        }
    }

    // Back to straight-alpha sRGB8-encoded entries.
    let unprem: Vec<Oklab> = palette4
        .iter()
        .map(|f| {
            let a = f.alpha.max(1e-6);
            Oklab::new(f.color.l / a, f.color.a / a, f.color.b / a)
        })
        .collect();
    let rgb8 = oklab_to_srgb8(&unprem);
    let pal = rgb8
        .iter()
        .zip(&palette4)
        .map(|(c, f)| {
            let a = (f.alpha.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            if a == 0 { [0, 0, 0, 0] } else { [c.red, c.green, c.blue, a] }
        })
        .collect();
    Ok((pal, indices))
}

/// Floyd–Steinberg (serpentine) quantization of encoded gray values to `levels` levels.
pub(crate) fn gray_levels(gray: &[f32], w: u32, levels: u32, dither: f32) -> Vec<u8> {
    let w = w as usize;
    let maxl = (levels - 1) as f32;
    let strength = if dither.is_finite() { dither.clamp(0.0, 1.0) } else { 0.0 };
    let mut out = vec![0u8; gray.len()];
    if strength <= 0.0 {
        for (o, g) in out.iter_mut().zip(gray) {
            *o = (g.clamp(0.0, 1.0) * maxl + 0.5) as u8;
        }
        return out;
    }
    let mut cur = vec![0.0f32; w + 2];
    let mut next = vec![0.0f32; w + 2];
    for (y, (row, orow)) in gray.chunks(w).zip(out.chunks_mut(w)).enumerate() {
        let ltr = y % 2 == 0;
        for i in 0..w {
            let x = if ltr { i } else { w - 1 - i };
            let v = (row[x] + cur[x + 1]).clamp(0.0, 1.0);
            let q = (v * maxl + 0.5).floor().min(maxl);
            orow[x] = q as u8;
            let err = (v - q / maxl) * strength;
            let (fwd, back) = if ltr { (x + 2, x) } else { (x, x + 2) };
            cur[fwd] += err * 7.0 / 16.0;
            next[back] += err * 3.0 / 16.0;
            next[x + 1] += err * 5.0 / 16.0;
            next[fwd] += err * 1.0 / 16.0;
        }
        std::mem::swap(&mut cur, &mut next);
        next.iter_mut().for_each(|v| *v = 0.0);
    }
    out
}
