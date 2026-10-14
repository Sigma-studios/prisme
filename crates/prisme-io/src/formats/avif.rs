//! AVIF through `avif-parse` (container) and `rusty_av1d` (pure-Rust rav1d fork, no
//! assembly), so it also works on wasm32. Colour comes from the `colr` box (ICC or
//! nclx) or the AV1 sequence header; `irot`/`imir` are applied.

use super::{ftyp_brands, ImageDecoder};
use crate::color::{self, cicp, ColorSource, Layout, Samples};
use crate::{Decoded, DecodedImage, IoError};
use rusty_av1d::{Decoder, PixelLayout, PlanarImageComponent, Picture, Rav1dError, Settings};

pub struct Avif;

impl ImageDecoder for Avif {
    fn name(&self) -> &'static str {
        "AVIF"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["avif", "avifs"]
    }

    fn sniff(&self, h: &[u8]) -> bool {
        ftyp_brands(h).iter().any(|b| b == b"avif" || b == b"avis")
    }

    fn decode(&self, bytes: &[u8]) -> Result<Decoded, IoError> {
        decode_avif(bytes).map(Decoded::Raster)
    }
}

// --------------------------------------------------------------------------------
// Minimal ISO-BMFF property scan (colr / irot / imir of the item properties)
// --------------------------------------------------------------------------------

#[derive(Default)]
pub(crate) struct ItemProps {
    pub nclx: Option<(u8, u8, u8, bool)>,
    pub icc: Option<Vec<u8>>,
    /// Anti-clockwise rotation in quarter turns.
    pub irot: u8,
    /// Mirror axis: 0 = vertical axis (left/right), 1 = horizontal axis (top/bottom).
    pub imir: Option<u8>,
}

/// Iterate `(type, payload)` of the boxes in `data`.
fn boxes(mut data: &[u8]) -> impl Iterator<Item = ([u8; 4], &[u8])> {
    std::iter::from_fn(move || {
        if data.len() < 8 {
            return None;
        }
        let size32 = u32::from_be_bytes(data[..4].try_into().ok()?) as u64;
        let ty: [u8; 4] = data[4..8].try_into().ok()?;
        let (header, size) = match size32 {
            0 => (8, data.len() as u64),
            1 => (16, u64::from_be_bytes(data.get(8..16)?.try_into().ok()?)),
            s => (8, s),
        };
        if size < header as u64 || size > data.len() as u64 {
            return None;
        }
        let payload = &data[header..size as usize];
        data = &data[size as usize..];
        Some((ty, payload))
    })
}

fn child<'a>(data: &'a [u8], ty: &[u8; 4]) -> Option<&'a [u8]> {
    boxes(data).find(|(t, _)| t == ty).map(|(_, p)| p)
}

pub(crate) fn item_props(file: &[u8]) -> ItemProps {
    let mut props = ItemProps::default();
    let Some(ipco) = child(file, b"meta")
        .and_then(|m| m.get(4..))
        .and_then(|m| child(m, b"iprp"))
        .and_then(|p| child(p, b"ipco"))
    else {
        return props;
    };
    for (ty, p) in boxes(ipco) {
        match &ty {
            b"colr" if p.len() >= 4 => match &p[..4] {
                b"nclx" if p.len() >= 11 && props.nclx.is_none() => {
                    let prim = u16::from_be_bytes([p[4], p[5]]) as u8;
                    let tf = u16::from_be_bytes([p[6], p[7]]) as u8;
                    let mc = u16::from_be_bytes([p[8], p[9]]) as u8;
                    props.nclx = Some((prim, tf, mc, p[10] & 0x80 != 0));
                }
                b"prof" | b"rICC" if props.icc.is_none() => props.icc = Some(p[4..].to_vec()),
                _ => {}
            },
            b"irot" if !p.is_empty() => props.irot = p[0] & 3,
            b"imir" if !p.is_empty() => props.imir = Some(p[0] & 1),
            _ => {}
        }
    }
    props
}

/// Apply HEIF `irot` then `imir` to an image.
pub(crate) fn apply_transforms(img: DecodedImage, props: &ItemProps) -> DecodedImage {
    let rot = match props.irot {
        1 => Some(8), // 90° anti-clockwise
        2 => Some(3),
        3 => Some(6),
        _ => None,
    };
    let mir = props.imir.map(|axis| if axis == 0 { 2 } else { 4 });
    img.oriented(rot).oriented(mir)
}

// --------------------------------------------------------------------------------
// AV1 decoding
// --------------------------------------------------------------------------------

fn av1_err(e: Rav1dError) -> IoError {
    IoError::Decode(format!("AV1: {e}"))
}

fn decode_av1(data: &[u8]) -> Result<Picture, IoError> {
    let mut settings = Settings::new();
    #[cfg(target_arch = "wasm32")]
    settings.set_n_threads(1);
    settings.set_max_frame_delay(1);
    let mut dec = Decoder::with_settings(&settings).map_err(av1_err)?;
    match dec.send_data(data.to_vec().into_boxed_slice(), None, None, None) {
        Ok(()) | Err(Rav1dError::TryAgain) => {}
        Err(e) => return Err(av1_err(e)),
    }
    for _ in 0..1024 {
        match dec.get_picture() {
            Ok(p) => return Ok(p),
            Err(Rav1dError::TryAgain) => match dec.send_pending_data() {
                Ok(()) | Err(Rav1dError::TryAgain) => {}
                Err(e) => return Err(av1_err(e)),
            },
            Err(e) => return Err(av1_err(e)),
        }
    }
    Err(IoError::Decode("AV1 decoder produced no picture".into()))
}

/// Normalised plane sampler (handles 8-bit and 16-bit storage).
struct Plane<'a> {
    data: &'a [u8],
    stride: usize,
    wide: bool,
    w: usize,
    h: usize,
}

impl Plane<'_> {
    fn new(p: &Picture, c: PlanarImageComponent, w: usize, h: usize) -> Plane<'_> {
        Plane { data: p.plane(c), stride: p.stride(c) as usize, wide: p.bit_depth() > 8, w, h }
    }

    #[inline]
    fn at(&self, x: usize, y: usize) -> f32 {
        let (x, y) = (x.min(self.w - 1), y.min(self.h - 1));
        if self.wide {
            let o = y * self.stride + x * 2;
            match self.data.get(o..o + 2) {
                Some(b) => u16::from_ne_bytes([b[0], b[1]]) as f32,
                None => 0.0,
            }
        } else {
            self.data.get(y * self.stride + x).copied().unwrap_or(0) as f32
        }
    }

    /// Bilinear sample at a continuous position (sample centres at integers).
    #[inline]
    fn bilinear(&self, fx: f32, fy: f32) -> f32 {
        let (fx, fy) = (fx.max(0.0), fy.max(0.0));
        let (x0, y0) = (fx as usize, fy as usize);
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let a = self.at(x0, y0) * (1.0 - tx) + self.at(x0 + 1, y0) * tx;
        let b = self.at(x0, y0 + 1) * (1.0 - tx) + self.at(x0 + 1, y0 + 1) * tx;
        a * (1.0 - ty) + b * ty
    }
}

/// YUV -> encoded R'G'B' (normalised floats), interleaved with `ch` channels.
fn yuv_to_rgb(p: &Picture, mc: u8, full_range: bool, out: &mut [f32], ch: usize) -> Result<(), IoError> {
    let (w, h) = (p.width() as usize, p.height() as usize);
    let bits = p.bits_per_component().map_or(8, |b| b.0) as i32;
    let max = ((1 << bits) - 1) as f32;
    let scale = (1 << (bits - 8)) as f32;
    let (y_off, y_scale, c_scale) =
        if full_range { (0.0, 1.0 / max, 1.0 / max) } else { (16.0 * scale, 1.0 / (219.0 * scale), 1.0 / (224.0 * scale)) };
    let c_off = 128.0 * scale;
    let layout = p.pixel_layout();
    let (sx, sy) = match layout {
        PixelLayout::I420 => (2usize, 2usize),
        PixelLayout::I422 => (2, 1),
        _ => (1, 1),
    };
    let yp = Plane::new(p, PlanarImageComponent::Y, w, h);
    let mono = layout == PixelLayout::I400;
    let (cw, chh) = (w.div_ceil(sx), h.div_ceil(sy));
    let (up, vp) = if mono {
        (None, None)
    } else {
        (Some(Plane::new(p, PlanarImageComponent::U, cw, chh)), Some(Plane::new(p, PlanarImageComponent::V, cw, chh)))
    };
    let (kr, kb) = match mc {
        1 => (0.2126f32, 0.0722f32),
        9 | 10 => (0.2627, 0.0593),
        4 => (0.30, 0.11),
        7 => (0.212, 0.087),
        _ => (0.299, 0.114),
    };
    let kg = 1.0 - kr - kb;
    let identity = mc == 0;
    color::for_row_blocks(out, w * ch, |row0, block| {
        for (j, px) in block.chunks_exact_mut(ch).enumerate() {
            let (x, y) = ((row0 * w + j) % w, (row0 * w + j) / w);
            let yv = (yp.at(x, y) - y_off) * y_scale;
            let rgb = match (&up, &vp) {
                (Some(u), Some(v)) => {
                    let (fx, fy) = ((x as f32 + 0.5) / sx as f32 - 0.5, (y as f32 + 0.5) / sy as f32 - 0.5);
                    let (fx, fy) = if sx == 1 && sy == 1 { (x as f32, y as f32) } else { (fx, fy) };
                    let (cb, cr) = if identity {
                        // GBR: U = B, V = R stored like luma.
                        ((u.bilinear(fx, fy) - y_off) * y_scale, (v.bilinear(fx, fy) - y_off) * y_scale)
                    } else {
                        ((u.bilinear(fx, fy) - c_off) * c_scale, (v.bilinear(fx, fy) - c_off) * c_scale)
                    };
                    if identity {
                        [cr, yv, cb]
                    } else {
                        let r = yv + (2.0 - 2.0 * kr) * cr;
                        let b = yv + (2.0 - 2.0 * kb) * cb;
                        let g = (yv - kr * r - kb * b) / kg;
                        [r, g, b]
                    }
                }
                _ => [yv; 3],
            };
            px[..3].copy_from_slice(&rgb);
        }
    });
    Ok(())
}

fn decode_avif(bytes: &[u8]) -> Result<DecodedImage, IoError> {
    let avif = avif_parse::read_avif(&mut std::io::Cursor::new(bytes)).map_err(|e| IoError::Decode(format!("AVIF: {e:?}")))?;
    let props = item_props(bytes);
    let pic = decode_av1(&avif.primary_item)?;
    let (w, h) = (pic.width(), pic.height());
    let bits = pic.bits_per_component().map_or(8, |b| b.0);
    let alpha_pic = match &avif.alpha_item {
        Some(a) => match decode_av1(a) {
            Ok(p) if p.width() == w && p.height() == h => Some(p),
            Ok(_) => {
                log::warn!("AVIF alpha plane size mismatch; ignoring alpha");
                None
            }
            Err(e) => {
                log::warn!("AVIF alpha decode failed: {e}");
                None
            }
        },
        None => None,
    };
    let (prim, tf, mc, full) = props.nclx.unwrap_or((
        pic.color_primaries() as u8,
        pic.transfer_characteristic() as u8,
        pic.matrix_coefficients() as u8,
        pic.color_range() == rusty_av1d::pixel::YUVRange::Full,
    ));
    let ch = if alpha_pic.is_some() { 4 } else { 3 };
    let mut rgb = vec![0.0f32; w as usize * h as usize * ch];
    yuv_to_rgb(&pic, mc, full, &mut rgb, ch)?;
    if let Some(ap) = &alpha_pic {
        let abits = ap.bits_per_component().map_or(8, |b| b.0) as i32;
        let amax = ((1 << abits) - 1) as f32;
        let a_full = ap.color_range() == rusty_av1d::pixel::YUVRange::Full;
        let s = (1 << (abits - 8)) as f32;
        let plane = Plane::new(ap, PlanarImageComponent::Y, w as usize, h as usize);
        let wu = w as usize;
        color::for_row_blocks(&mut rgb, wu * 4, |row0, block| {
            for (j, px) in block.chunks_exact_mut(4).enumerate() {
                let i = row0 * wu + j;
                let v = plane.at(i % wu, i / wu);
                px[3] = if a_full { v / amax } else { ((v - 16.0 * s) / (219.0 * s)).clamp(0.0, 1.0) };
            }
        });
    }
    let (source, desc) = match props.icc.as_deref().and_then(ColorSource::from_icc) {
        Some(s) => s,
        None => match cicp(prim, tf) {
            Some((a, d)) => (ColorSource::Analytic(a), d),
            None => color::untagged_srgb(),
        },
    };
    if avif.premultiplied_alpha && ch == 4 {
        // Premultiplication happened on the encoded values.
        color::unpremultiply(&mut rgb, w);
    }
    let layout = if ch == 4 { Layout::Rgba } else { Layout::Rgb };
    let pixels = color::to_working(Samples::F32(&rgb), w, h, layout, &source);
    let img = DecodedImage::new(w, h, pixels, desc, bits, false);
    Ok(apply_transforms(img, &props))
}
