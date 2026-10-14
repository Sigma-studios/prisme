//! Target file-size search and fast size estimation.

use crate::compress::{self, Final, chunks};
use crate::raw::{self, EncodeCfg};
use crate::{
    BitDepth, ColorMode, ExportError, ExportResult, PngOptions, TargetPriority, TargetSize, resize_rgba, scaled_dims,
};

/// Effort used while searching (the winner is re-encoded at the requested effort).
const SEARCH_EFFORT: u8 = 1;

/// Encode `src` with `opts` (ignoring `scale` and `target`).
fn encode_once(src: &[f32], w: u32, h: u32, opts: &PngOptions, notes: &mut Vec<String>) -> Result<Final, ExportError> {
    let cfg = EncodeCfg::from_opts(opts);
    let built = raw::build_raw(src, w, h, &cfg, notes)?;
    let params = compress::CompressParams::from_opts(opts, &built);
    let bytes = compress::compress(&built.raw, &params)?;
    compress::finalize(bytes, opts, notes)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Cand {
    scale: f32,
    /// `None` = lossless at the requested settings.
    colors: Option<u16>,
    /// Force 8-bit (used when the request was 16-bit).
    eight: bool,
}

struct Outcome {
    fin: Final,
    w: u32,
    h: u32,
    cand: Cand,
    notes: Vec<String>,
}

impl Outcome {
    fn len(&self) -> u64 {
        self.fin.bytes.len() as u64
    }
}

struct Search<'a> {
    src: &'a [f32],
    w: u32,
    h: u32,
    opts: &'a PngOptions,
    t: &'a TargetSize,
    min_scale: f32,
    cache: Option<(u32, u32, Vec<f32>)>,
    smallest: Option<Outcome>,
    probes: usize,
    progress: &'a mut dyn FnMut(f32, &str),
    cancel: &'a dyn Fn() -> bool,
}

impl Search<'_> {
    fn fits(&self, o: &Outcome) -> bool {
        o.len() <= self.t.max_bytes
    }

    fn scaled(&mut self, scale: f32) -> (u32, u32) {
        let (nw, nh) = scaled_dims(self.w, self.h, scale);
        if (nw, nh) == (self.w, self.h) {
            return (nw, nh);
        }
        if !matches!(&self.cache, Some((cw, ch, _)) if (*cw, *ch) == (nw, nh)) {
            let px = resize_rgba(self.src, self.w, self.h, nw, nh, self.opts.resize_filter);
            self.cache = Some((nw, nh, px));
        }
        (nw, nh)
    }

    fn cand_opts(&self, c: Cand, effort: u8, zopfli: bool) -> PngOptions {
        let mut o = self.opts.clone();
        o.target = None;
        o.effort = effort;
        o.zopfli = zopfli;
        if c.eight && o.bit_depth == BitDepth::Sixteen {
            o.bit_depth = BitDepth::Auto;
        }
        if let Some(n) = c.colors {
            o.color_mode = ColorMode::Palette;
            o.palette_colors = n;
            if !matches!(o.bit_depth, BitDepth::One | BitDepth::Two | BitDepth::Four) {
                o.bit_depth = BitDepth::Auto;
            }
            if o.dither <= 0.0 {
                o.dither = 1.0;
            }
        }
        o
    }

    fn probe(&mut self, c: Cand, effort: u8, zopfli: bool) -> Result<Outcome, ExportError> {
        if (self.cancel)() {
            return Err(ExportError::Cancelled);
        }
        self.probes += 1;
        let frac = (0.05 + self.probes as f32 * 0.04).min(0.9);
        let msg = match c.colors {
            Some(n) => format!("Trying {n} colours at {:.0}%", c.scale * 100.0),
            None => format!("Trying lossless at {:.0}%", c.scale * 100.0),
        };
        (self.progress)(frac, &msg);
        let o = self.cand_opts(c, effort, zopfli);
        let (nw, nh) = self.scaled(c.scale);
        let mut notes = Vec::new();
        let fin = if (nw, nh) == (self.w, self.h) {
            encode_once(self.src, nw, nh, &o, &mut notes)?
        } else {
            let px = &self.cache.as_ref().map(|c| &c.2).ok_or_else(|| ExportError::Encode("resize cache".into()))?;
            encode_once(px, nw, nh, &o, &mut notes)?
        };
        let out = Outcome { fin, w: nw, h: nh, cand: c, notes };
        log::debug!("target probe {c:?} effort {effort}: {} bytes", out.len());
        if self.smallest.as_ref().is_none_or(|s| out.len() < s.len()) {
            self.smallest = Some(Outcome {
                fin: Final {
                    bytes: out.fin.bytes.clone(),
                    bit_depth: out.fin.bit_depth,
                    color_mode: out.fin.color_mode,
                    palette_colors: out.fin.palette_colors,
                },
                w: out.w,
                h: out.h,
                cand: out.cand,
                notes: out.notes.clone(),
            });
        }
        Ok(out)
    }

    fn fast(&mut self, c: Cand) -> Result<Outcome, ExportError> {
        self.probe(c, SEARCH_EFFORT, false)
    }

    /// Largest colour count in `lo..=hi` that fits at `scale` with `eight` bit flag.
    fn search_colors(&mut self, scale: f32, lo: u16, hi: u16, eight: bool) -> Result<Option<Outcome>, ExportError> {
        let lo = lo.clamp(2, 256);
        let hi = hi.clamp(lo, 256);
        let r = self.fast(Cand { scale, colors: Some(hi), eight })?;
        if self.fits(&r) {
            return Ok(Some(r));
        }
        if lo == hi {
            return Ok(None);
        }
        let r_lo = self.fast(Cand { scale, colors: Some(lo), eight })?;
        if !self.fits(&r_lo) {
            return Ok(None);
        }
        let (mut good, mut bad, mut best) = (lo, hi, r_lo);
        while bad - good > (good / 20).max(1) {
            let mid = ((good as u32 + bad as u32) / 2) as u16;
            let r = self.fast(Cand { scale, colors: Some(mid), eight })?;
            if self.fits(&r) {
                good = mid;
                best = r;
            } else {
                bad = mid;
            }
        }
        Ok(Some(best))
    }

    /// Largest scale in `[min_scale, hi)` that fits (assumes `hi` does not).
    fn search_scale(&mut self, colors: Option<u16>, hi: f32, eight: bool) -> Result<Option<Outcome>, ExportError> {
        let lo = self.min_scale;
        if lo >= hi {
            return Ok(None);
        }
        let r_lo = self.fast(Cand { scale: lo, colors, eight })?;
        if !self.fits(&r_lo) {
            return Ok(None);
        }
        let (mut good, mut bad, mut best) = (lo, hi, r_lo);
        for _ in 0..10 {
            if bad - good < 0.01 || scaled_dims(self.w, self.h, good) == scaled_dims(self.w, self.h, bad) {
                break;
            }
            let mid = (good + bad) * 0.5;
            let r = self.fast(Cand { scale: mid, colors, eight })?;
            if self.fits(&r) {
                good = mid;
                best = r;
            } else {
                bad = mid;
            }
        }
        Ok(Some(best))
    }

    /// Joint search along a path from (scale 1, max colours) to (min_scale, 2 colours).
    fn search_balanced(&mut self, maxc: u16, eight: bool) -> Result<Option<Outcome>, ExportError> {
        let min_s = self.min_scale;
        let at = |t: f32| {
            let scale = 1.0 - t * (1.0 - min_s);
            let c = ((maxc as f32).ln() * (1.0 - t) + 2f32.ln() * t).exp().round() as u16;
            Cand { scale, colors: Some(c.clamp(2, maxc)), eight }
        };
        let r1 = self.fast(at(1.0))?;
        if !self.fits(&r1) {
            return Ok(None);
        }
        let (mut good, mut bad, mut best) = (1.0f32, 0.0f32, r1);
        for _ in 0..9 {
            if good - bad < 0.01 {
                break;
            }
            let mid = (good + bad) * 0.5;
            let r = self.fast(at(mid))?;
            if self.fits(&r) {
                good = mid;
                best = r;
            } else {
                bad = mid;
            }
        }
        Ok(Some(best))
    }

    fn notch_down(&self, c: Cand) -> Option<Cand> {
        match c.colors {
            Some(n) if n > 2 => Some(Cand { colors: Some(((n as f32 * 0.9) as u16).min(n - 1).max(2)), ..c }),
            _ if c.scale > self.min_scale + 1e-4 => Some(Cand { scale: (c.scale - 0.03).max(self.min_scale), ..c }),
            _ => None,
        }
    }
}

fn describe(o: &Outcome) -> String {
    let mut s = match o.cand.colors {
        Some(_) => format!("palette {} colours", o.fin.palette_colors.unwrap_or(0)),
        None => "lossless".to_string(),
    };
    if o.cand.eight {
        s.push_str(", 8-bit");
    }
    if o.cand.scale < 1.0 {
        s.push_str(&format!(", scale {:.3} ({}x{})", o.cand.scale, o.w, o.h));
    }
    s
}

fn into_result(o: Outcome, met: bool, mut notes: Vec<String>) -> ExportResult {
    notes.extend(o.notes);
    ExportResult {
        width: o.w,
        height: o.h,
        bit_depth: o.fin.bit_depth,
        color_mode: o.fin.color_mode,
        palette_colors: o.fin.palette_colors,
        scale: if o.cand.scale < 1.0 { o.cand.scale } else { 1.0 },
        bytes: o.fin.bytes,
        met_target: Some(met),
        notes,
    }
}

pub(crate) fn search(
    src: &[f32],
    w: u32,
    h: u32,
    opts: &PngOptions,
    t: &TargetSize,
    progress: &mut dyn FnMut(f32, &str),
    cancel: &dyn Fn() -> bool,
) -> Result<ExportResult, ExportError> {
    let min_scale = if t.min_scale.is_finite() { t.min_scale.clamp(0.05, 1.0) } else { 1.0 };
    let mut s = Search { src, w, h, opts, t, min_scale, cache: None, smallest: None, probes: 0, progress, cancel };
    let max = t.max_bytes;
    let effort = opts.effort.min(6);
    let zopfli = opts.zopfli;
    let mut notes = vec![];
    let ok_note = |o: &Outcome| format!("Target {max} bytes met: {} → {} bytes", describe(o), o.len());

    // (1) Lossless at the requested settings and effort.
    let full = Cand { scale: 1.0, colors: None, eight: false };
    let r = s.probe(full, effort, zopfli)?;
    if s.fits(&r) {
        notes.push(ok_note(&r));
        return Ok(into_result(r, true, notes));
    }
    if effort < 6 && !zopfli && (r.len() as f64) < max as f64 * 1.15 && (w as u64 * h as u64) <= 4_000_000 {
        let r = s.probe(full, 6, false)?;
        if s.fits(&r) {
            notes.push(ok_note(&r));
            return Ok(into_result(r, true, notes));
        }
    }
    // (1b) 16-bit request: try 8-bit lossless.
    let mut eight = false;
    if opts.bit_depth == BitDepth::Sixteen {
        eight = true;
        let c = Cand { eight: true, ..full };
        let r = s.probe(c, effort, zopfli)?;
        if s.fits(&r) {
            notes.push(ok_note(&r));
            return Ok(into_result(r, true, notes));
        }
    }

    // (2)/(3) Lossy search with a fast effort.
    let mut maxc: u16 = if opts.color_mode == ColorMode::Palette { opts.palette_colors.clamp(2, 256) } else { 256 };
    match opts.bit_depth {
        BitDepth::One => maxc = maxc.min(2),
        BitDepth::Two => maxc = maxc.min(4),
        BitDepth::Four => maxc = maxc.min(16),
        _ => {}
    }
    // If the request already was a palette, step (1) covered `maxc` colours.
    let pal_top = if opts.color_mode == ColorMode::Palette { maxc.saturating_sub(1).max(2) } else { maxc };
    let p = t.allow_lossy_palette;
    let rz = t.allow_resize && min_scale < 1.0;
    let winner = match (p, rz) {
        (false, false) => None,
        (true, false) => s.search_colors(1.0, 2, pal_top, eight)?,
        (false, true) => s.search_scale(None, 1.0, eight)?,
        (true, true) => match t.priority {
            TargetPriority::ColorsFirst => {
                let floor = 32.min(pal_top);
                match s.search_colors(1.0, floor, pal_top, eight)? {
                    Some(o) => Some(o),
                    None => match s.search_scale(Some(floor), 1.0, eight)? {
                        Some(o) => Some(o),
                        None if floor > 2 => s.search_colors(min_scale, 2, floor - 1, eight)?,
                        None => None,
                    },
                }
            }
            TargetPriority::ResolutionFirst => {
                let r = s.fast(Cand { scale: 1.0, colors: Some(pal_top), eight })?;
                if s.fits(&r) {
                    Some(r)
                } else {
                    match s.search_scale(Some(pal_top), 1.0, eight)? {
                        Some(o) => Some(o),
                        None if pal_top > 2 => s.search_colors(min_scale, 2, pal_top - 1, eight)?,
                        None => None,
                    }
                }
            }
            TargetPriority::Balanced => {
                let r = s.fast(Cand { scale: 1.0, colors: Some(pal_top), eight })?;
                if s.fits(&r) { Some(r) } else { s.search_balanced(pal_top, eight)? }
            }
        },
    };

    let Some(win) = winner else {
        let smallest = s.smallest.take().ok_or_else(|| ExportError::Encode("no candidate encoded".into()))?;
        notes.push(format!(
            "Target {max} bytes not reachable with the allowed reductions; smallest result: {} → {} bytes",
            describe(&smallest),
            smallest.len()
        ));
        return Ok(into_result(smallest, false, notes));
    };

    // Re-encode the winner at the requested effort.
    if effort <= SEARCH_EFFORT && !zopfli {
        notes.push(ok_note(&win));
        return Ok(into_result(win, true, notes));
    }
    (s.progress)(0.92, "Final encode");
    let r = s.probe(win.cand, effort, zopfli)?;
    if s.fits(&r) {
        notes.push(ok_note(&r));
        return Ok(into_result(r, true, notes));
    }
    if let Some(c) = s.notch_down(win.cand) {
        let r = s.probe(c, effort, zopfli)?;
        if s.fits(&r) {
            notes.push(ok_note(&r));
            return Ok(into_result(r, true, notes));
        }
    }
    notes.push(format!("Final effort exceeded the target; kept the effort-{SEARCH_EFFORT} encoding"));
    notes.push(ok_note(&win));
    Ok(into_result(win, true, notes))
}

/// Effort factor relative to an effort-0 encode (empirical).
fn effort_factor(effort: u8, zopfli: bool) -> f64 {
    let f = match effort {
        0 => 1.0,
        1 => 0.97,
        2 => 0.95,
        3 => 0.94,
        4 => 0.93,
        5 => 0.925,
        _ => 0.92,
    };
    if zopfli { f * 0.96 } else { f }
}

const SAMPLE_GRID: u32 = 8;
const SAMPLE_TILE: u32 = 64;

pub(crate) fn estimate(px: &[f32], w: u32, h: u32, opts: &PngOptions) -> Result<u64, ExportError> {
    let scale = if opts.scale.is_finite() && opts.scale > 0.0 { opts.scale } else { 1.0 };
    let (ow, oh) = scaled_dims(w, h, scale);
    let out_px = ow as f64 * oh as f64;
    let mut o = opts.clone();
    o.target = None;
    o.effort = 0;
    o.zopfli = false;
    if o.filter == crate::FilterStrategy::Brute {
        o.filter = crate::FilterStrategy::Auto;
    }
    let mut notes = Vec::new();
    let sample_px = (SAMPLE_GRID * SAMPLE_TILE) as f64 * (SAMPLE_GRID * SAMPLE_TILE) as f64;

    let (fin, enc_px) = if out_px <= sample_px * 2.0 {
        let fin = if (ow, oh) != (w, h) {
            let r = resize_rgba(px, w, h, ow, oh, opts.resize_filter);
            encode_once(&r, ow, oh, &o, &mut notes)?
        } else {
            encode_once(px, w, h, &o, &mut notes)?
        };
        (fin, out_px)
    } else {
        // Tile mosaic sampled across the image at full source resolution, then scaled.
        let ts_w = ((SAMPLE_TILE as f32 / scale).ceil() as u32).clamp(1, w);
        let ts_h = ((SAMPLE_TILE as f32 / scale).ceil() as u32).clamp(1, h);
        let (mw, mh) = (ts_w * SAMPLE_GRID, ts_h * SAMPLE_GRID);
        let mut mosaic = vec![0.0f32; mw as usize * mh as usize * 4];
        for gy in 0..SAMPLE_GRID {
            let sy = (((gy as f64 + 0.5) * h as f64 / SAMPLE_GRID as f64) as i64 - ts_h as i64 / 2)
                .clamp(0, (h - ts_h) as i64) as usize;
            for gx in 0..SAMPLE_GRID {
                let sx = (((gx as f64 + 0.5) * w as f64 / SAMPLE_GRID as f64) as i64 - ts_w as i64 / 2)
                    .clamp(0, (w - ts_w) as i64) as usize;
                for y in 0..ts_h as usize {
                    let s0 = ((sy + y) * w as usize + sx) * 4;
                    let d0 = ((gy * ts_h) as usize + y) * mw as usize * 4 + (gx * ts_w) as usize * 4;
                    mosaic[d0..d0 + ts_w as usize * 4].copy_from_slice(&px[s0..s0 + ts_w as usize * 4]);
                }
            }
        }
        let (sw, sh) = scaled_dims(mw, mh, scale);
        let fin = if (sw, sh) != (mw, mh) {
            let r = resize_rgba(&mosaic, mw, mh, sw, sh, opts.resize_filter);
            encode_once(&r, sw, sh, &o, &mut notes)?
        } else {
            encode_once(&mosaic, mw, mh, &o, &mut notes)?
        };
        (fin, sw as f64 * sh as f64)
    };
    let idat: usize = chunks(&fin.bytes).filter(|c| &c.0 == b"IDAT").map(|c| c.1.len()).sum();
    let overhead = fin.bytes.len().saturating_sub(idat) as f64;
    let est = overhead + idat as f64 * (out_px / enc_px.max(1.0)) * effort_factor(opts.effort.min(6), opts.zopfli);
    Ok(est.round() as u64)
}
