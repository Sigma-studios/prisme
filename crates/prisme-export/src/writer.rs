//! Minimal pure-Rust PNG writer (filters, Adam7, zlib via miniz_oxide or zopfli).
//!
//! Used on wasm32, where oxipng's libdeflate (C) is unavailable. Compiled everywhere so it is
//! covered by the native test-suite.
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

use crate::FilterStrategy;
use crate::compress::chunk;
use crate::raw::{Raw, RawColor};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Strat {
    Fixed(u8),
    MinSum,
    Entropy,
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i16 + b as i16 - c as i16;
    let (pa, pb, pc) = ((p - a as i16).abs(), (p - b as i16).abs(), (p - c as i16).abs());
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

fn filter_row(ft: u8, bpp: usize, row: &[u8], prev: &[u8], out: &mut [u8]) {
    for i in 0..row.len() {
        let a = if i >= bpp { row[i - bpp] } else { 0 };
        let b = prev[i];
        let c = if i >= bpp { prev[i - bpp] } else { 0 };
        out[i] = match ft {
            0 => row[i],
            1 => row[i].wrapping_sub(a),
            2 => row[i].wrapping_sub(b),
            3 => row[i].wrapping_sub(((a as u16 + b as u16) / 2) as u8),
            _ => row[i].wrapping_sub(paeth(a, b, c)),
        };
    }
}

fn score_minsum(d: &[u8]) -> u64 {
    d.iter().map(|&v| (v as i8).unsigned_abs() as u64).sum()
}

fn score_entropy(d: &[u8]) -> f64 {
    let mut hist = [0u32; 256];
    d.iter().for_each(|&v| hist[v as usize] += 1);
    let n = d.len().max(1) as f64;
    hist.iter().filter(|&&c| c > 0).map(|&c| {
        let p = c as f64 / n;
        -p * p.log2()
    }).sum()
}

/// Filter one (sub-)image; appends `[filter byte, data...]` rows to `out`.
fn filter_image(data: &[u8], row_bytes: usize, bpp: usize, strat: Strat, out: &mut Vec<u8>) {
    if row_bytes == 0 {
        return;
    }
    let zero = vec![0u8; row_bytes];
    let mut bufs = vec![vec![0u8; row_bytes]; 5];
    let mut prev: &[u8] = &zero;
    for row in data.chunks(row_bytes) {
        let ft = match strat {
            Strat::Fixed(f) => {
                filter_row(f, bpp, row, prev, &mut bufs[f as usize]);
                f
            }
            Strat::MinSum | Strat::Entropy => {
                let mut best = (f64::INFINITY, 0u8);
                for f in 0..5u8 {
                    filter_row(f, bpp, row, prev, &mut bufs[f as usize]);
                    let s = if strat == Strat::MinSum {
                        score_minsum(&bufs[f as usize]) as f64
                    } else {
                        score_entropy(&bufs[f as usize])
                    };
                    if s < best.0 {
                        best = (s, f);
                    }
                }
                best.1
            }
        };
        out.push(ft);
        out.extend_from_slice(&bufs[ft as usize]);
        prev = row;
    }
}

const ADAM7: [(usize, usize, usize, usize); 7] =
    [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)];

/// Extract one Adam7 pass as a packed sub-image. Returns (width, height, data).
fn extract_pass(raw: &Raw, pass: (usize, usize, usize, usize)) -> (usize, usize, Vec<u8>) {
    let (x0, y0, dx, dy) = pass;
    let (w, h) = (raw.w as usize, raw.h as usize);
    let pw = if w > x0 { (w - x0).div_ceil(dx) } else { 0 };
    let ph = if h > y0 { (h - y0).div_ceil(dy) } else { 0 };
    if pw == 0 || ph == 0 {
        return (pw, ph, Vec::new());
    }
    let bits = raw.color.channels() * raw.depth as usize;
    let src_rb = raw.row_bytes();
    let dst_rb = (pw * bits).div_ceil(8);
    let mut out = vec![0u8; dst_rb * ph];
    for py in 0..ph {
        let srow = &raw.data[(y0 + py * dy) * src_rb..][..src_rb];
        let drow = &mut out[py * dst_rb..][..dst_rb];
        if bits >= 8 {
            let b = bits / 8;
            for px in 0..pw {
                let sx = x0 + px * dx;
                drow[px * b..px * b + b].copy_from_slice(&srow[sx * b..sx * b + b]);
            }
        } else {
            let mask = (1u8 << bits) - 1;
            for px in 0..pw {
                let sbit = (x0 + px * dx) * bits;
                let v = (srow[sbit / 8] >> (8 - bits - sbit % 8)) & mask;
                let dbit = px * bits;
                drow[dbit / 8] |= v << (8 - bits - dbit % 8);
            }
        }
    }
    (pw, ph, out)
}

fn filtered_stream(raw: &Raw, interlace: bool, strat: Strat) -> Vec<u8> {
    let bits = raw.color.channels() * raw.depth as usize;
    let bpp = (bits / 8).max(1);
    // Palette / sub-byte images rarely benefit from filtering.
    let strat = match strat {
        Strat::Fixed(_) => strat,
        _ if matches!(raw.color, RawColor::Indexed(_)) || raw.depth < 8 => Strat::Fixed(0),
        _ => strat,
    };
    let mut out = Vec::with_capacity(raw.data.len() + raw.h as usize * 2);
    if interlace {
        for pass in ADAM7 {
            let (pw, _, data) = extract_pass(raw, pass);
            filter_image(&data, (pw * bits).div_ceil(8), bpp, strat, &mut out);
        }
    } else {
        filter_image(&raw.data, raw.row_bytes(), bpp, strat, &mut out);
    }
    out
}

fn zlib(data: &[u8], level: u8, zopfli: Option<u8>) -> Vec<u8> {
    if let Some(it) = zopfli {
        let opts = zopfli::Options {
            iteration_count: std::num::NonZeroU64::new(it.max(1) as u64).unwrap_or(std::num::NonZeroU64::MIN),
            ..Default::default()
        };
        let mut out = Vec::new();
        if zopfli::compress(opts, zopfli::Format::Zlib, data, &mut out).is_ok() {
            return out;
        }
    }
    miniz_oxide::deflate::compress_to_vec_zlib(data, level)
}

/// Encode `raw` as a complete PNG file (no colour chunks; those are added by `finalize`).
pub(crate) fn write_png(raw: &Raw, interlace: bool, filter: FilterStrategy, effort: u8, zopfli: Option<u8>) -> Vec<u8> {
    let level = match effort {
        0 => 1,
        1 => 3,
        2 => 6,
        3 => 7,
        4 => 8,
        _ => 9,
    };
    let candidates: Vec<Strat> = match filter {
        FilterStrategy::None => vec![Strat::Fixed(0)],
        FilterStrategy::Sub => vec![Strat::Fixed(1)],
        FilterStrategy::Up => vec![Strat::Fixed(2)],
        FilterStrategy::Average => vec![Strat::Fixed(3)],
        FilterStrategy::Paeth => vec![Strat::Fixed(4)],
        FilterStrategy::MinSum => vec![Strat::MinSum],
        FilterStrategy::Entropy => vec![Strat::Entropy],
        FilterStrategy::Auto => match effort {
            0 | 1 => vec![Strat::MinSum],
            2 | 3 => vec![Strat::Fixed(0), Strat::MinSum],
            _ => vec![Strat::Fixed(0), Strat::MinSum, Strat::Entropy, Strat::Fixed(1)],
        },
        FilterStrategy::Brute => (0..5).map(Strat::Fixed).chain([Strat::MinSum, Strat::Entropy]).collect(),
    };
    let idat = if candidates.len() == 1 {
        zlib(&filtered_stream(raw, interlace, candidates[0]), level, zopfli)
    } else {
        // Pick the best strategy with a fast compression level, then compress for real.
        let trial = |s: Strat| zlib(&filtered_stream(raw, interlace, s), level.min(3), None).len();
        let best = candidates.iter().copied().min_by_key(|&s| trial(s)).unwrap_or(Strat::MinSum);
        zlib(&filtered_stream(raw, interlace, best), level, zopfli)
    };

    let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend(raw.w.to_be_bytes());
    ihdr.extend(raw.h.to_be_bytes());
    ihdr.extend([raw.depth, raw.color.png_code(), 0, 0, interlace as u8]);
    png.extend(chunk(b"IHDR", &ihdr));
    if let RawColor::Indexed(pal) = &raw.color {
        let plte: Vec<u8> = pal.iter().flat_map(|c| [c[0], c[1], c[2]]).collect();
        png.extend(chunk(b"PLTE", &plte));
        let n_trns = pal.iter().rposition(|c| c[3] != 255).map_or(0, |i| i + 1);
        if n_trns > 0 {
            let trns: Vec<u8> = pal[..n_trns].iter().map(|c| c[3]).collect();
            png.extend(chunk(b"tRNS", &trns));
        }
    }
    // Split IDAT into 1 MiB chunks.
    for part in idat.chunks(1 << 20) {
        png.extend(chunk(b"IDAT", part));
    }
    if idat.is_empty() {
        png.extend(chunk(b"IDAT", &[]));
    }
    png.extend(chunk(b"IEND", &[]));
    png
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(bytes: &[u8]) -> (png::ColorType, u8, bool, Vec<u8>) {
        let mut reader = png::Decoder::new(std::io::Cursor::new(bytes)).read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        buf.truncate(info.buffer_size());
        let i = reader.info();
        (i.color_type, i.bit_depth as u8, i.interlaced, buf)
    }

    #[test]
    fn roundtrip_all_types() {
        let (w, h) = (23u32, 17u32);
        let mut seed = 1u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 24) as u8
        };
        let cases: Vec<(RawColor, u8)> = vec![
            (RawColor::Gray, 1),
            (RawColor::Gray, 4),
            (RawColor::Gray, 8),
            (RawColor::Gray, 16),
            (RawColor::GrayAlpha, 8),
            (RawColor::Rgb, 8),
            (RawColor::Rgb, 16),
            (RawColor::Rgba, 8),
            (RawColor::Rgba, 16),
            (RawColor::Indexed(vec![[1, 2, 3, 0], [4, 5, 6, 128], [7, 8, 9, 255], [9, 9, 9, 255]]), 2),
            (RawColor::Indexed((0..=255).map(|i| [i as u8, 255 - i as u8, 0, 255]).collect()), 8),
        ];
        for (color, depth) in cases {
            let row_bytes = (w as usize * color.channels() * depth as usize).div_ceil(8);
            let pal_len = if let RawColor::Indexed(p) = &color { p.len() } else { 0 };
            let mut data: Vec<u8> = (0..row_bytes * h as usize).map(|_| rnd()).collect();
            if pal_len > 0 && pal_len < (1 << depth) {
                data.iter_mut().for_each(|v| *v %= pal_len as u8);
            }
            // Zero padding bits at row ends for sub-byte depths.
            let bits = w as usize * color.channels() * depth as usize;
            if !bits.is_multiple_of(8) {
                let mask = 0xFFu8 << (8 - bits % 8);
                for r in 0..h as usize {
                    data[r * row_bytes + row_bytes - 1] &= mask;
                }
            }
            let raw = Raw { w, h, color: color.clone(), depth, data: data.clone() };
            for interlace in [false, true] {
                for (filter, effort, z) in [
                    (FilterStrategy::Auto, 2, None),
                    (FilterStrategy::Paeth, 0, None),
                    (FilterStrategy::Brute, 4, Some(2)),
                ] {
                    let png_bytes = write_png(&raw, interlace, filter, effort, z);
                    let (ct, d, il, out) = decode(&png_bytes);
                    assert_eq!(ct as u8, color.png_code());
                    assert_eq!(d, depth);
                    assert_eq!(il, interlace);
                    assert_eq!(out, data, "{color:?} depth {depth} interlace {interlace} {filter:?}");
                }
            }
        }
    }
}
