//! Working space -> encoded target space conversion and image statistics.

use crate::resize::for_each_row;
use prisme_color::{RgbSpace, SpaceConverter, working_luminance};

#[inline]
#[allow(clippy::manual_clamp)] // intentional: maps NaN to 0, unlike f32::clamp
fn clamp01(v: f32) -> f32 {
    // `max` first so NaN maps to 0.
    v.max(0.0).min(1.0)
}

/// Convert working-space straight-alpha RGBA to encoded target-space RGBA in [0,1].
///
/// * `gray`: replace colour by the encoded relative luminance (R=G=B).
/// * `flatten`: composite over white (in linear light) and make opaque.
pub(crate) fn encode_pixels(px: &[f32], w: u32, space: RgbSpace, gray: bool, flatten: bool) -> Vec<f32> {
    let conv = SpaceConverter::new(space);
    let transfer = space.transfer();
    let mut out = vec![0.0f32; px.len()];
    let row_len = (w as usize * 4).max(4);
    for_each_row(&mut out, row_len, |y, row| {
        let src = &px[y * row_len..y * row_len + row.len()];
        for (d, s) in row.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
            let mut a = clamp01(s[3]);
            let fin = |v: f32| if v.is_finite() { v } else { 0.0 };
            let mut rgb = [fin(s[0]), fin(s[1]), fin(s[2])];
            if flatten {
                rgb = rgb.map(|c| c * a + (1.0 - a));
                a = 1.0;
            }
            let enc = if gray {
                let y = transfer.from_linear(working_luminance(rgb).max(0.0));
                [y, y, y]
            } else {
                conv.from_working(rgb)
            };
            d[0] = clamp01(enc[0]);
            d[1] = clamp01(enc[1]);
            d[2] = clamp01(enc[2]);
            d[3] = a;
        }
    });
    out
}

/// Quantize encoded RGBA to 8-bit (plain rounding).
pub(crate) fn to_u8(enc: &[f32], clean_transparent: bool) -> Vec<[u8; 4]> {
    enc.chunks_exact(4)
        .map(|p| {
            let q = |v: f32| (v * 255.0 + 0.5) as u8;
            let a = q(p[3]);
            if clean_transparent && a == 0 { [0, 0, 0, 0] } else { [q(p[0]), q(p[1]), q(p[2]), a] }
        })
        .collect()
}

/// Quantize encoded RGBA to 16-bit (exact rounding).
pub(crate) fn to_u16(enc: &[f32], clean_transparent: bool) -> Vec<[u16; 4]> {
    enc.chunks_exact(4)
        .map(|p| {
            let q = |v: f32| (v as f64 * 65535.0 + 0.5) as u16;
            let a = q(p[3]);
            if clean_transparent && a == 0 { [0, 0, 0, 0] } else { [q(p[0]), q(p[1]), q(p[2]), a] }
        })
        .collect()
}

pub(crate) struct Stats {
    pub opaque: bool,
    pub gray: bool,
}

pub(crate) fn stats<T: Copy + PartialEq>(px: &[[T; 4]], max: T) -> Stats {
    let mut opaque = true;
    let mut gray = true;
    for p in px {
        opaque &= p[3] == max;
        gray &= p[0] == p[1] && p[1] == p[2];
        if !opaque && !gray {
            break;
        }
    }
    Stats { opaque, gray }
}

/// Exact palette when the image has at most `max` distinct RGBA8 colours.
/// Returns (palette, indices).
pub(crate) fn exact_palette(px: &[[u8; 4]], max: usize) -> Option<(Vec<[u8; 4]>, Vec<u8>)> {
    // Small open-addressing table (no external hasher: works on wasm without getrandom).
    const SIZE: usize = 1024;
    let mut keys = [0u32; SIZE];
    let mut vals = [0u16; SIZE];
    let mut used = [false; SIZE];
    let mut palette: Vec<[u8; 4]> = Vec::new();
    let mut indices = Vec::with_capacity(px.len());
    let mut last: Option<(u32, u8)> = None;
    for p in px {
        let key = u32::from_le_bytes(*p);
        if let Some((k, i)) = last
            && k == key
        {
            indices.push(i);
            continue;
        }
        let mut slot = (key.wrapping_mul(0x9E37_79B1) >> 22) as usize & (SIZE - 1);
        let idx = loop {
            if !used[slot] {
                if palette.len() >= max {
                    return None;
                }
                used[slot] = true;
                keys[slot] = key;
                vals[slot] = palette.len() as u16;
                palette.push(*p);
                break vals[slot];
            }
            if keys[slot] == key {
                break vals[slot];
            }
            slot = (slot + 1) & (SIZE - 1);
        };
        indices.push(idx as u8);
        last = Some((key, idx as u8));
    }
    Some((palette, indices))
}
