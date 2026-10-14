//! Sparse, copy-on-write tiled pixel storage.
//!
//! A [`TileGrid`] maps tile coordinates to `Arc<Tile>`. Cloning a grid only bumps
//! reference counts, so snapshotting a document for undo is cheap; mutating goes
//! through `Arc::make_mut`, which clones a tile only when it is shared. Every tile
//! carries a process-unique `id` that changes whenever its content may have changed,
//! which GPU caches use as a content key.

use crate::geom::IRect;
use half::f16;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub const TILE_SIZE: u32 = 256;
pub const TILE_PIXELS: usize = (TILE_SIZE * TILE_SIZE) as usize;
const TS: i32 = TILE_SIZE as i32;

static NEXT_TILE_ID: AtomicU64 = AtomicU64::new(1);

fn next_tile_id() -> u64 {
    NEXT_TILE_ID.fetch_add(1, Ordering::Relaxed)
}

/// Storage precision of a tile. All math is done in f32; storage keeps memory low
/// while staying lossless for the source (f16 is lossless for 8-bit sources, f32 is
/// required for 16-bit integer and float sources).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum PixelFormat {
    #[default]
    F16,
    F32,
}

impl PixelFormat {
    pub fn max(self, o: PixelFormat) -> PixelFormat {
        if self == PixelFormat::F32 || o == PixelFormat::F32 { PixelFormat::F32 } else { PixelFormat::F16 }
    }

    pub fn bytes_per_channel(self) -> usize {
        match self {
            PixelFormat::F16 => 2,
            PixelFormat::F32 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TileCoord {
    pub x: i32,
    pub y: i32,
}

impl TileCoord {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    pub fn of_pixel(px: i32, py: i32) -> Self {
        Self { x: px.div_euclid(TS), y: py.div_euclid(TS) }
    }

    pub fn rect(&self) -> IRect {
        IRect::new(self.x * TS, self.y * TS, (self.x + 1) * TS, (self.y + 1) * TS)
    }
}

#[derive(Debug)]
pub enum TileBuf {
    F16(Vec<f16>),
    F32(Vec<f32>),
}

/// A `TILE_SIZE`² block of pixels with 1 (mask) or 4 (RGBA, straight alpha) channels.
#[derive(Debug)]
pub struct Tile {
    id: u64,
    channels: u8,
    buf: TileBuf,
}

impl Clone for Tile {
    fn clone(&self) -> Self {
        let buf = match &self.buf {
            TileBuf::F16(v) => TileBuf::F16(v.clone()),
            TileBuf::F32(v) => TileBuf::F32(v.clone()),
        };
        Tile { id: next_tile_id(), channels: self.channels, buf }
    }
}

impl Tile {
    pub fn new(format: PixelFormat, channels: u8) -> Self {
        let n = TILE_PIXELS * channels as usize;
        let buf = match format {
            PixelFormat::F16 => TileBuf::F16(vec![f16::ZERO; n]),
            PixelFormat::F32 => TileBuf::F32(vec![0.0; n]),
        };
        Tile { id: next_tile_id(), channels, buf }
    }

    pub fn from_f32(format: PixelFormat, channels: u8, data: &[f32]) -> Self {
        assert_eq!(data.len(), TILE_PIXELS * channels as usize);
        let buf = match format {
            PixelFormat::F16 => TileBuf::F16(data.iter().map(|&v| f16::from_f32(v)).collect()),
            PixelFormat::F32 => TileBuf::F32(data.to_vec()),
        };
        Tile { id: next_tile_id(), channels, buf }
    }

    /// Build from raw little-endian bytes in the given format (used by the file container).
    pub fn from_bytes(format: PixelFormat, channels: u8, bytes: &[u8]) -> Option<Self> {
        let n = TILE_PIXELS * channels as usize;
        if bytes.len() != n * format.bytes_per_channel() {
            return None;
        }
        let buf = match format {
            PixelFormat::F16 => TileBuf::F16(
                bytes.chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect(),
            ),
            PixelFormat::F32 => TileBuf::F32(
                bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            ),
        };
        Some(Tile { id: next_tile_id(), channels, buf })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        match &self.buf {
            TileBuf::F16(v) => v.iter().flat_map(|h| h.to_bits().to_le_bytes()).collect(),
            TileBuf::F32(v) => v.iter().flat_map(|f| f.to_le_bytes()).collect(),
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn channels(&self) -> u8 {
        self.channels
    }

    pub fn format(&self) -> PixelFormat {
        match self.buf {
            TileBuf::F16(_) => PixelFormat::F16,
            TileBuf::F32(_) => PixelFormat::F32,
        }
    }

    pub fn buf(&self) -> &TileBuf {
        &self.buf
    }

    pub fn byte_size(&self) -> usize {
        TILE_PIXELS * self.channels as usize * self.format().bytes_per_channel()
    }

    /// Read-only f32 view (borrowed for f32 tiles, converted for f16 tiles).
    pub fn f32_view(&self) -> Cow<'_, [f32]> {
        match &self.buf {
            TileBuf::F32(v) => Cow::Borrowed(v.as_slice()),
            TileBuf::F16(v) => {
                let mut out = vec![0.0f32; v.len()];
                half::slice::HalfFloatSliceExt::convert_to_f32_slice(v.as_slice(), &mut out);
                Cow::Owned(out)
            }
        }
    }

    /// Mutate the tile through an f32 slice. f16 tiles are converted in and out.
    pub fn with_f32_mut<R>(&mut self, f: impl FnOnce(&mut [f32]) -> R) -> R {
        self.id = next_tile_id();
        match &mut self.buf {
            TileBuf::F32(v) => f(v.as_mut_slice()),
            TileBuf::F16(v) => {
                let mut tmp = vec![0.0f32; v.len()];
                half::slice::HalfFloatSliceExt::convert_to_f32_slice(v.as_slice(), &mut tmp);
                let r = f(&mut tmp);
                half::slice::HalfFloatSliceExt::convert_from_f32_slice(v.as_mut_slice(), &tmp);
                r
            }
        }
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32) -> [f32; 4] {
        let c = self.channels as usize;
        let i = (y * TILE_SIZE + x) as usize * c;
        let read = |k: usize| -> f32 {
            match &self.buf {
                TileBuf::F16(v) => v[i + k].to_f32(),
                TileBuf::F32(v) => v[i + k],
            }
        };
        if c == 1 {
            let v = read(0);
            [v, v, v, v]
        } else {
            [read(0), read(1), read(2), read(3)]
        }
    }

    /// True when every pixel is fully transparent (alpha == 0, or value == 0 for masks).
    pub fn is_clear(&self) -> bool {
        let c = self.channels as usize;
        let off = if c == 4 { 3 } else { 0 };
        match &self.buf {
            TileBuf::F16(v) => v.iter().skip(off).step_by(c).all(|a| a.to_f32() == 0.0),
            TileBuf::F32(v) => v.iter().skip(off).step_by(c).all(|&a| a == 0.0),
        }
    }

    /// Fill every pixel with a constant.
    pub fn fill(&mut self, px: [f32; 4]) {
        let c = self.channels as usize;
        self.with_f32_mut(|d| {
            for p in d.chunks_exact_mut(c) {
                p.copy_from_slice(&px[..c]);
            }
        });
    }

    /// 2x box downsample of four child tiles (TL, TR, BL, BR) into one tile.
    /// Colour is averaged with alpha weighting so transparent pixels don't bleed.
    pub fn downsample(children: [Option<&Tile>; 4], format: PixelFormat, channels: u8) -> Tile {
        let c = channels as usize;
        let mut out = vec![0.0f32; TILE_PIXELS * c];
        let half_ts = (TILE_SIZE / 2) as usize;
        for (q, child) in children.iter().enumerate() {
            let Some(child) = child else { continue };
            let src = child.f32_view();
            let ox = (q % 2) * half_ts;
            let oy = (q / 2) * half_ts;
            for y in 0..half_ts {
                for x in 0..half_ts {
                    let o = ((oy + y) * TILE_SIZE as usize + ox + x) * c;
                    let s00 = ((2 * y) * TILE_SIZE as usize + 2 * x) * c;
                    let s01 = s00 + c;
                    let s10 = s00 + TILE_SIZE as usize * c;
                    let s11 = s10 + c;
                    if c == 1 {
                        out[o] = 0.25 * (src[s00] + src[s01] + src[s10] + src[s11]);
                    } else {
                        let mut acc = [0.0f32; 4];
                        for s in [s00, s01, s10, s11] {
                            let a = src[s + 3];
                            acc[0] += src[s] * a;
                            acc[1] += src[s + 1] * a;
                            acc[2] += src[s + 2] * a;
                            acc[3] += a;
                        }
                        if acc[3] > 0.0 {
                            out[o] = acc[0] / acc[3];
                            out[o + 1] = acc[1] / acc[3];
                            out[o + 2] = acc[2] / acc[3];
                            out[o + 3] = acc[3] * 0.25;
                        }
                    }
                }
            }
        }
        Tile::from_f32(format, channels, &out)
    }
}

/// Sparse grid of tiles. Missing tiles are fully transparent (or zero for masks).
#[derive(Clone, Debug)]
pub struct TileGrid {
    pub format: PixelFormat,
    pub channels: u8,
    pub tiles: HashMap<TileCoord, Arc<Tile>>,
}

impl TileGrid {
    pub fn new(format: PixelFormat, channels: u8) -> Self {
        Self { format, channels, tiles: HashMap::new() }
    }

    pub fn rgba(format: PixelFormat) -> Self {
        Self::new(format, 4)
    }

    pub fn mask() -> Self {
        Self::new(PixelFormat::F16, 1)
    }

    pub fn get_tile(&self, c: TileCoord) -> Option<&Arc<Tile>> {
        self.tiles.get(&c)
    }

    /// Mutable access to a tile, creating it (transparent) if missing and cloning it if shared.
    pub fn tile_mut(&mut self, c: TileCoord) -> &mut Tile {
        let (format, channels) = (self.format, self.channels);
        let arc = self.tiles.entry(c).or_insert_with(|| Arc::new(Tile::new(format, channels)));
        Arc::make_mut(arc)
    }

    pub fn insert(&mut self, c: TileCoord, t: Tile) {
        self.tiles.insert(c, Arc::new(t));
    }

    pub fn remove_clear_tiles(&mut self) {
        self.tiles.retain(|_, t| !t.is_clear());
    }

    pub fn get_pixel(&self, x: i32, y: i32) -> [f32; 4] {
        let c = TileCoord::of_pixel(x, y);
        match self.tiles.get(&c) {
            Some(t) => t.get((x - c.x * TS) as u32, (y - c.y * TS) as u32),
            None => [0.0; 4],
        }
    }

    /// Bilinear sample at a continuous position (pixel centres at +0.5).
    pub fn sample_bilinear(&self, x: f32, y: f32) -> [f32; 4] {
        let fx = x - 0.5;
        let fy = y - 0.5;
        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let p00 = self.get_pixel(x0, y0);
        let p10 = self.get_pixel(x0 + 1, y0);
        let p01 = self.get_pixel(x0, y0 + 1);
        let p11 = self.get_pixel(x0 + 1, y0 + 1);
        if self.channels == 1 {
            let v = (p00[0] * (1.0 - tx) + p10[0] * tx) * (1.0 - ty) + (p01[0] * (1.0 - tx) + p11[0] * tx) * ty;
            return [v; 4];
        }
        // premultiplied interpolation
        let w = [(1.0 - tx) * (1.0 - ty), tx * (1.0 - ty), (1.0 - tx) * ty, tx * ty];
        let mut acc = [0.0f32; 4];
        for (p, w) in [p00, p10, p01, p11].iter().zip(w) {
            let a = p[3] * w;
            acc[0] += p[0] * a;
            acc[1] += p[1] * a;
            acc[2] += p[2] * a;
            acc[3] += a;
        }
        if acc[3] > 1e-8 {
            [acc[0] / acc[3], acc[1] / acc[3], acc[2] / acc[3], acc[3]]
        } else {
            [0.0; 4]
        }
    }

    pub fn set_pixel(&mut self, x: i32, y: i32, px: [f32; 4]) {
        let c = TileCoord::of_pixel(x, y);
        let ch = self.channels as usize;
        let lx = (x - c.x * TS) as usize;
        let ly = (y - c.y * TS) as usize;
        self.tile_mut(c).with_f32_mut(|d| {
            let i = (ly * TILE_SIZE as usize + lx) * ch;
            d[i..i + ch].copy_from_slice(&px[..ch]);
        });
    }

    /// Bounding box of all allocated tiles (tile granularity).
    pub fn tile_bounds(&self) -> IRect {
        self.tiles.keys().fold(IRect::EMPTY, |acc, c| acc.union(&c.rect()))
    }

    /// Tight bounding box of non-transparent pixels.
    pub fn content_bounds(&self) -> IRect {
        let mut r = IRect::EMPTY;
        let ch = self.channels as usize;
        let ai = if ch == 4 { 3 } else { 0 };
        for (c, t) in &self.tiles {
            let d = t.f32_view();
            let (mut x0, mut y0, mut x1, mut y1) = (TS, TS, -1, -1);
            for y in 0..TS {
                let row = &d[(y as usize * TILE_SIZE as usize) * ch..((y as usize + 1) * TILE_SIZE as usize) * ch];
                for x in 0..TS {
                    if row[x as usize * ch + ai] > 0.0 {
                        x0 = x0.min(x);
                        x1 = x1.max(x);
                        y0 = y0.min(y);
                        y1 = y1.max(y);
                    }
                }
            }
            if x1 >= 0 {
                r = r.union(&IRect::new(c.x * TS + x0, c.y * TS + y0, c.x * TS + x1 + 1, c.y * TS + y1 + 1));
            }
        }
        r
    }

    /// Tile coordinates overlapping a pixel rect.
    pub fn coords_in(rect: &IRect) -> impl Iterator<Item = TileCoord> + use<> {
        let r = *rect;
        let (tx0, ty0) = (r.x0.div_euclid(TS), r.y0.div_euclid(TS));
        let (tx1, ty1) = ((r.x1 - 1).div_euclid(TS), (r.y1 - 1).div_euclid(TS));
        let empty = r.is_empty();
        (ty0..=ty1).flat_map(move |y| (tx0..=tx1).map(move |x| TileCoord::new(x, y))).filter(move |_| !empty)
    }

    /// Read a rectangle into a dense f32 buffer (`channels` per pixel, row-major).
    pub fn read_rect(&self, rect: &IRect) -> Vec<f32> {
        let ch = self.channels as usize;
        let w = rect.width() as usize;
        let mut out = vec![0.0f32; w * rect.height() as usize * ch];
        for c in Self::coords_in(rect) {
            let Some(t) = self.tiles.get(&c) else { continue };
            let tr = c.rect().intersect(rect);
            if tr.is_empty() {
                continue;
            }
            let d = t.f32_view();
            for y in tr.y0..tr.y1 {
                let ly = (y - c.y * TS) as usize;
                let lx0 = (tr.x0 - c.x * TS) as usize;
                let n = tr.width() as usize * ch;
                let src = &d[(ly * TILE_SIZE as usize + lx0) * ch..][..n];
                let oy = (y - rect.y0) as usize;
                let ox = (tr.x0 - rect.x0) as usize;
                out[(oy * w + ox) * ch..][..n].copy_from_slice(src);
            }
        }
        out
    }

    /// Write a dense f32 buffer into a rectangle. Tiles that end up clear are dropped.
    pub fn write_rect(&mut self, rect: &IRect, data: &[f32]) {
        let ch = self.channels as usize;
        let w = rect.width() as usize;
        assert_eq!(data.len(), w * rect.height() as usize * ch);
        for c in Self::coords_in(rect).collect::<Vec<_>>() {
            let tr = c.rect().intersect(rect);
            if tr.is_empty() {
                continue;
            }
            // Skip creating tiles for fully transparent input.
            if !self.tiles.contains_key(&c) {
                let mut any = false;
                'outer: for y in tr.y0..tr.y1 {
                    let oy = (y - rect.y0) as usize;
                    for x in tr.x0..tr.x1 {
                        let ox = (x - rect.x0) as usize;
                        let a = data[(oy * w + ox) * ch + if ch == 4 { 3 } else { 0 }];
                        if a != 0.0 {
                            any = true;
                            break 'outer;
                        }
                    }
                }
                if !any {
                    continue;
                }
            }
            self.tile_mut(c).with_f32_mut(|d| {
                for y in tr.y0..tr.y1 {
                    let ly = (y - c.y * TS) as usize;
                    let lx0 = (tr.x0 - c.x * TS) as usize;
                    let n = tr.width() as usize * ch;
                    let oy = (y - rect.y0) as usize;
                    let ox = (tr.x0 - rect.x0) as usize;
                    d[(ly * TILE_SIZE as usize + lx0) * ch..][..n].copy_from_slice(&data[(oy * w + ox) * ch..][..n]);
                }
            });
            if self.tiles[&c].is_clear() {
                self.tiles.remove(&c);
            }
        }
    }

    /// Build a grid from a dense f32 image placed at the origin.
    pub fn from_dense(format: PixelFormat, channels: u8, width: u32, height: u32, data: &[f32]) -> Self {
        let mut g = TileGrid::new(format, channels);
        g.write_rect(&IRect::from_size(width, height), data);
        g
    }

    /// Apply `f` to every pixel of `rect` (creating tiles as needed). `f(x, y, px)`.
    pub fn map_rect(&mut self, rect: &IRect, mut f: impl FnMut(i32, i32, &mut [f32])) {
        let ch = self.channels as usize;
        for c in Self::coords_in(rect).collect::<Vec<_>>() {
            let tr = c.rect().intersect(rect);
            if tr.is_empty() {
                continue;
            }
            self.tile_mut(c).with_f32_mut(|d| {
                for y in tr.y0..tr.y1 {
                    let ly = (y - c.y * TS) as usize;
                    for x in tr.x0..tr.x1 {
                        let lx = (x - c.x * TS) as usize;
                        let i = (ly * TILE_SIZE as usize + lx) * ch;
                        f(x, y, &mut d[i..i + ch]);
                    }
                }
            });
        }
    }

    /// Unique bytes held (each Arc counted once by this grid).
    pub fn byte_size(&self) -> usize {
        self.tiles.values().map(|t| t.byte_size()).sum()
    }

    /// Translate the whole grid by an integer offset (re-tiles when not tile-aligned).
    pub fn translated(&self, dx: i32, dy: i32) -> TileGrid {
        if dx % TS == 0 && dy % TS == 0 {
            let mut g = TileGrid::new(self.format, self.channels);
            for (c, t) in &self.tiles {
                g.tiles.insert(TileCoord::new(c.x + dx / TS, c.y + dy / TS), t.clone());
            }
            return g;
        }
        let b = self.tile_bounds();
        let data = self.read_rect(&b);
        let mut g = TileGrid::new(self.format, self.channels);
        g.write_rect(&b.translate(dx, dy), &data);
        g
    }

    pub fn convert_format(&self, format: PixelFormat) -> TileGrid {
        if format == self.format {
            return self.clone();
        }
        let mut g = TileGrid::new(format, self.channels);
        for (c, t) in &self.tiles {
            g.tiles.insert(*c, Arc::new(Tile::from_f32(format, self.channels, &t.f32_view())));
        }
        g
    }
}

// ---------------------------------------------------------------------------------
// Serialization: tiles can be routed to an out-of-band store (deduplicated by tile id)
// by the file container; without a store they serialize inline.
// ---------------------------------------------------------------------------------

pub mod store {
    use super::*;
    use std::cell::RefCell;

    #[derive(Default)]
    pub struct TileSink {
        pub index_of: HashMap<u64, u32>,
        pub tiles: Vec<Arc<Tile>>,
    }

    impl TileSink {
        pub fn add(&mut self, t: &Arc<Tile>) -> u32 {
            if let Some(&i) = self.index_of.get(&t.id()) {
                return i;
            }
            let i = self.tiles.len() as u32;
            self.tiles.push(t.clone());
            self.index_of.insert(t.id(), i);
            i
        }
    }

    pub trait TileSource {
        fn get(&self, index: u32) -> Option<Arc<Tile>>;
    }

    thread_local! {
        pub(super) static SINK: RefCell<Option<TileSink>> = const { RefCell::new(None) };
        pub(super) static SOURCE: RefCell<Option<Box<dyn TileSource>>> = RefCell::new(None);
    }

    /// Run `f` with an active tile sink; tiles serialized inside are stored by reference.
    pub fn with_sink<R>(f: impl FnOnce() -> R) -> (R, TileSink) {
        SINK.with(|s| *s.borrow_mut() = Some(TileSink::default()));
        let r = f();
        let sink = SINK.with(|s| s.borrow_mut().take()).unwrap_or_default();
        (r, sink)
    }

    /// Run `f` with an active tile source used to resolve tile references.
    pub fn with_source<R>(src: Box<dyn TileSource>, f: impl FnOnce() -> R) -> R {
        SOURCE.with(|s| *s.borrow_mut() = Some(src));
        let r = f();
        SOURCE.with(|s| *s.borrow_mut() = None);
        r
    }
}

#[derive(Serialize, Deserialize)]
enum TileRepr {
    Ref(u32),
    Inline { format: PixelFormat, channels: u8, bytes: Vec<u8> },
}

#[derive(Serialize, Deserialize)]
struct GridRepr {
    format: PixelFormat,
    channels: u8,
    tiles: Vec<(TileCoord, TileRepr)>,
}

impl Serialize for TileGrid {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut coords: Vec<_> = self.tiles.keys().copied().collect();
        coords.sort();
        let tiles = coords
            .into_iter()
            .map(|c| {
                let t = &self.tiles[&c];
                let repr = store::SINK.with(|sink| match sink.borrow_mut().as_mut() {
                    Some(sink) => TileRepr::Ref(sink.add(t)),
                    None => TileRepr::Inline { format: t.format(), channels: t.channels(), bytes: t.to_bytes() },
                });
                (c, repr)
            })
            .collect();
        GridRepr { format: self.format, channels: self.channels, tiles }.serialize(s)
    }
}

impl<'de> Deserialize<'de> for TileGrid {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let repr = GridRepr::deserialize(d)?;
        let mut g = TileGrid::new(repr.format, repr.channels);
        for (c, t) in repr.tiles {
            let tile = match t {
                TileRepr::Ref(i) => store::SOURCE
                    .with(|s| s.borrow().as_ref().and_then(|s| s.get(i)))
                    .ok_or_else(|| serde::de::Error::custom(format!("missing tile ref {i}")))?,
                TileRepr::Inline { format, channels, bytes } => Arc::new(
                    Tile::from_bytes(format, channels, &bytes)
                        .ok_or_else(|| serde::de::Error::custom("bad inline tile"))?,
                ),
            };
            g.tiles.insert(c, tile);
        }
        Ok(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_roundtrip_across_tiles() {
        let mut g = TileGrid::rgba(PixelFormat::F32);
        let r = IRect::new(-10, 200, 300, 600);
        let data: Vec<f32> = (0..r.area() as usize * 4).map(|i| (i % 97) as f32 / 97.0 + 0.01).collect();
        g.write_rect(&r, &data);
        assert_eq!(g.read_rect(&r), data);
        assert_eq!(g.content_bounds(), r);
    }

    #[test]
    fn cow_changes_id() {
        let mut g = TileGrid::rgba(PixelFormat::F16);
        g.set_pixel(1, 1, [1.0, 0.5, 0.25, 1.0]);
        let snap = g.clone();
        let id0 = snap.tiles[&TileCoord::new(0, 0)].id();
        g.set_pixel(2, 2, [1.0; 4]);
        assert_ne!(g.tiles[&TileCoord::new(0, 0)].id(), id0);
        assert_eq!(snap.get_pixel(2, 2), [0.0; 4]);
        assert_eq!(g.get_pixel(1, 1), [1.0, 0.5, 0.25, 1.0]);
    }

    #[test]
    fn f16_is_lossless_for_8bit_srgb() {
        for v in 0..=255u32 {
            let s = v as f32 / 255.0;
            let lin = if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) };
            let h = f16::from_f32(lin).to_f32();
            let back = if h <= 0.0031308 { h * 12.92 } else { 1.055 * h.powf(1.0 / 2.4) - 0.055 };
            assert_eq!((back * 255.0).round() as u32, v);
        }
    }
}
