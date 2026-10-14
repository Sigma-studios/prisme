//! The `.prisme` project container.
//!
//! Layout (all integers little-endian):
//!
//! ```text
//! magic  "PRISME\0\x01"            8 bytes
//! version u32                      currently 1
//! count   u32                      number of chunks
//! TOC     count x { tag [u8;4], offset u64, length u64 }
//! chunks  in order PREV (optional), DOC, TILE
//! ```
//!
//! * `PREV`: flattened sRGB8 RGBA preview: `w u32, h u32, lz4(w*h*4 bytes)`. It sits right
//!   after the TOC so a thumbnail / instant first paint only reads the head of the file.
//! * `DOC `: postcard-serialized [`History`]; every tile is a reference into `TILE`.
//! * `TILE`: `count u32`, then per tile `format u8, channels u8, flags u8, reserved u8,
//!   len u32, data`. Data is LZ4 of the little-endian samples, byte-plane shuffled when
//!   `flags & 1` (much better ratios on float data). Tiles are deduplicated by tile id,
//!   so history states that share tiles store them once.

use crate::IoError;
use prisme_core::history::HistoryEntry;
use prisme_core::tile::store::{with_sink, with_source, TileSource};
use prisme_core::tile::{Tile, TileGrid, TILE_PIXELS};
use prisme_core::{Document, History, PixelFormat};
use std::sync::Arc;

const MAGIC: &[u8; 8] = b"PRISME\0\x01";
const VERSION: u32 = 1;
const TAG_PREVIEW: [u8; 4] = *b"PREV";
const TAG_DOC: [u8; 4] = *b"DOC ";
const TAG_TILES: [u8; 4] = *b"TILE";
const HEADER_LEN: usize = 16;
const TOC_ENTRY_LEN: usize = 20;
const FLAG_SHUFFLED: u8 = 1;

fn corrupt(msg: impl Into<String>) -> IoError {
    IoError::Corrupt(msg.into())
}

/// Serialize a whole history (all states, structurally shared tiles deduplicated) or
/// only the current state if `include_history` is false.
pub fn save(history: &History, include_history: bool) -> Result<Vec<u8>, IoError> {
    save_with_preview(history, include_history, None)
}

/// Like [`save`], embedding a caller-provided flattened preview (w, h, sRGB8 RGBA) at
/// the front of the file.
pub fn save_with_preview(
    history: &History,
    include_history: bool,
    preview: Option<(u32, u32, &[u8])>,
) -> Result<Vec<u8>, IoError> {
    let current_only;
    let history = if include_history {
        history
    } else {
        let e = &history.entries()[history.cursor()];
        let entry = HistoryEntry { label: e.label.clone(), doc: e.doc.clone(), merge_key: None };
        let mut h = History::from_parts(vec![entry], 0);
        h.budget_bytes = history.budget_bytes;
        current_only = h;
        &current_only
    };

    let (doc, sink) = with_sink(|| postcard::to_allocvec(history));
    let doc = doc.map_err(|e| IoError::Encode(format!("document serialization: {e}")))?;
    let tiles = encode_tiles(&sink.tiles);

    let mut chunks: Vec<([u8; 4], Vec<u8>)> = Vec::with_capacity(3);
    if let Some((w, h, rgba)) = preview {
        if rgba.len() != w as usize * h as usize * 4 {
            return Err(IoError::Encode("preview size does not match its dimensions".into()));
        }
        let mut p = Vec::with_capacity(8 + rgba.len() / 2);
        p.extend_from_slice(&w.to_le_bytes());
        p.extend_from_slice(&h.to_le_bytes());
        p.extend_from_slice(&lz4_flex::compress(rgba));
        chunks.push((TAG_PREVIEW, p));
    }
    chunks.push((TAG_DOC, doc));
    chunks.push((TAG_TILES, tiles));

    let toc_len = chunks.len() * TOC_ENTRY_LEN;
    let total = HEADER_LEN + toc_len + chunks.iter().map(|c| c.1.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
    let mut offset = (HEADER_LEN + toc_len) as u64;
    for (tag, data) in &chunks {
        out.extend_from_slice(tag);
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&(data.len() as u64).to_le_bytes());
        offset += data.len() as u64;
    }
    for (_, data) in &chunks {
        out.extend_from_slice(data);
    }
    Ok(out)
}

/// Load a history saved with [`save`] / [`save_with_preview`].
pub fn load(bytes: &[u8]) -> Result<History, IoError> {
    let toc = read_toc(bytes)?;
    let find = |tag: [u8; 4]| toc.iter().find(|(t, _)| *t == tag).map(|(_, d)| *d);
    let doc = find(TAG_DOC).ok_or_else(|| corrupt("missing DOC chunk"))?;
    let tiles = find(TAG_TILES).ok_or_else(|| corrupt("missing TILE chunk"))?;
    let tiles = decode_tiles(tiles)?;

    struct Source(Vec<Arc<Tile>>);
    impl TileSource for Source {
        fn get(&self, index: u32) -> Option<Arc<Tile>> {
            self.0.get(index as usize).cloned()
        }
    }
    let history: History = with_source(Box::new(Source(tiles)), || postcard::from_bytes(doc))
        .map_err(|e| corrupt(format!("document: {e}")))?;

    if history.entries().is_empty() {
        return Err(corrupt("history has no states"));
    }
    for e in history.entries() {
        validate_document(&e.doc)?;
    }
    // Rebuild through `from_parts` so an out-of-range cursor is clamped.
    let mut h = History::from_parts(history.entries().to_vec(), history.cursor());
    h.budget_bytes = history.budget_bytes;
    Ok(h)
}

/// Read only the embedded flattened preview (fast path for instant open/thumbnail).
/// Returns (w, h, sRGB8 RGBA bytes).
pub fn read_preview(bytes: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let toc = read_toc(bytes).ok()?;
    let p = toc.iter().find(|(t, _)| *t == TAG_PREVIEW)?.1;
    let w = u32::from_le_bytes(p.get(0..4)?.try_into().ok()?);
    let h = u32::from_le_bytes(p.get(4..8)?.try_into().ok()?);
    let n = (w as usize).checked_mul(h as usize)?.checked_mul(4)?;
    // A preview larger than 64 MiB is certainly corrupt; don't allocate for it.
    if n > 64 << 20 {
        return None;
    }
    let rgba = lz4_flex::decompress(&p[8..], n).ok()?;
    (rgba.len() == n).then_some((w, h, rgba))
}

/// A chunk: tag and payload.
type Chunk<'a> = ([u8; 4], &'a [u8]);

/// Parse the header and table of contents, returning each chunk's bytes.
fn read_toc(bytes: &[u8]) -> Result<Vec<Chunk<'_>>, IoError> {
    if bytes.len() < HEADER_LEN || &bytes[..8] != MAGIC {
        return Err(corrupt("not a Prisme project file"));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes"));
    if version != VERSION {
        return Err(IoError::Unsupported(format!("Prisme project version {version} (this build reads {VERSION})")));
    }
    let count = u32::from_le_bytes(bytes[12..16].try_into().expect("4 bytes")) as usize;
    let toc_end = count
        .checked_mul(TOC_ENTRY_LEN)
        .and_then(|l| l.checked_add(HEADER_LEN))
        .filter(|&e| e <= bytes.len())
        .ok_or_else(|| corrupt("truncated table of contents"))?;
    let mut out = Vec::with_capacity(count);
    for e in bytes[HEADER_LEN..toc_end].chunks_exact(TOC_ENTRY_LEN) {
        let tag: [u8; 4] = e[..4].try_into().expect("4 bytes");
        let off = u64::from_le_bytes(e[4..12].try_into().expect("8 bytes"));
        let len = u64::from_le_bytes(e[12..20].try_into().expect("8 bytes"));
        let end = off.checked_add(len).filter(|&end| end <= bytes.len() as u64 && off >= toc_end as u64);
        let end = end.ok_or_else(|| corrupt(format!("chunk {} out of bounds", String::from_utf8_lossy(&tag))))?;
        out.push((tag, &bytes[off as usize..end as usize]));
    }
    Ok(out)
}

// --------------------------------------------------------------------------------
// Tiles
// --------------------------------------------------------------------------------

/// Transpose `bytes` (elements of `width` bytes) into byte planes.
fn shuffle(bytes: &[u8], width: usize) -> Vec<u8> {
    let n = bytes.len() / width;
    let mut out = vec![0u8; bytes.len()];
    for (i, el) in bytes.chunks_exact(width).enumerate() {
        for (b, &v) in el.iter().enumerate() {
            out[b * n + i] = v;
        }
    }
    out
}

fn unshuffle(planes: &[u8], width: usize) -> Vec<u8> {
    let n = planes.len() / width;
    let mut out = vec![0u8; planes.len()];
    for (b, plane) in planes.chunks_exact(n.max(1)).enumerate().take(width) {
        for (i, &v) in plane.iter().enumerate() {
            out[i * width + b] = v;
        }
    }
    out
}

fn encode_tile(t: &Tile) -> Vec<u8> {
    let width = t.format().bytes_per_channel();
    let data = lz4_flex::compress(&shuffle(&t.to_bytes(), width));
    let mut out = Vec::with_capacity(8 + data.len());
    out.push(match t.format() {
        PixelFormat::F16 => 0,
        PixelFormat::F32 => 1,
    });
    out.push(t.channels());
    out.push(FLAG_SHUFFLED);
    out.push(0);
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    out
}

fn encode_tiles(tiles: &[Arc<Tile>]) -> Vec<u8> {
    #[cfg(not(target_arch = "wasm32"))]
    let encoded: Vec<Vec<u8>> = {
        use rayon::prelude::*;
        tiles.par_iter().map(|t| encode_tile(t)).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let encoded: Vec<Vec<u8>> = tiles.iter().map(|t| encode_tile(t)).collect();
    let mut out = Vec::with_capacity(4 + encoded.iter().map(Vec::len).sum::<usize>());
    out.extend_from_slice(&(tiles.len() as u32).to_le_bytes());
    for e in encoded {
        out.extend_from_slice(&e);
    }
    out
}

struct RawTile<'a> {
    format: PixelFormat,
    channels: u8,
    flags: u8,
    data: &'a [u8],
}

fn decode_tile(r: &RawTile<'_>) -> Result<Arc<Tile>, IoError> {
    let width = r.format.bytes_per_channel();
    let size = TILE_PIXELS * r.channels as usize * width;
    let bytes = lz4_flex::decompress(r.data, size).map_err(|e| corrupt(format!("tile data: {e}")))?;
    if bytes.len() != size {
        return Err(corrupt("tile data has the wrong size"));
    }
    let bytes = if r.flags & FLAG_SHUFFLED != 0 { unshuffle(&bytes, width) } else { bytes };
    Tile::from_bytes(r.format, r.channels, &bytes).map(Arc::new).ok_or_else(|| corrupt("bad tile"))
}

fn decode_tiles(chunk: &[u8]) -> Result<Vec<Arc<Tile>>, IoError> {
    let count = u32::from_le_bytes(chunk.get(..4).ok_or_else(|| corrupt("truncated TILE chunk"))?.try_into().expect("4"));
    // Every tile takes at least 8 header bytes; reject impossible counts before allocating.
    if count as usize > chunk.len() / 8 {
        return Err(corrupt("tile count exceeds chunk size"));
    }
    let mut raw = Vec::with_capacity(count as usize);
    let mut o = 4usize;
    for _ in 0..count {
        let h = chunk.get(o..o + 8).ok_or_else(|| corrupt("truncated tile header"))?;
        let format = match h[0] {
            0 => PixelFormat::F16,
            1 => PixelFormat::F32,
            f => return Err(corrupt(format!("unknown tile format {f}"))),
        };
        let channels = h[1];
        if channels != 1 && channels != 4 {
            return Err(corrupt(format!("unsupported tile channel count {channels}")));
        }
        let len = u32::from_le_bytes(h[4..8].try_into().expect("4")) as usize;
        let data = chunk.get(o + 8..o + 8 + len).ok_or_else(|| corrupt("truncated tile data"))?;
        raw.push(RawTile { format, channels, flags: h[2], data });
        o += 8 + len;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        raw.par_iter().map(decode_tile).collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        raw.iter().map(decode_tile).collect()
    }
}

/// Check that every tile matches its grid's channel count (corrupt files could
/// otherwise make later pixel access go out of bounds).
fn validate_document(doc: &Document) -> Result<(), IoError> {
    let check = |g: &TileGrid| -> Result<(), IoError> {
        if g.channels != 1 && g.channels != 4 {
            return Err(corrupt("grid with unsupported channel count"));
        }
        if g.tiles.values().any(|t| t.channels() != g.channels) {
            return Err(corrupt("tile channel count does not match its grid"));
        }
        Ok(())
    };
    for l in doc.all_layers() {
        if let Some(g) = l.raster() {
            check(g)?;
        }
        if let Some(m) = &l.mask {
            check(&m.grid)?;
        }
    }
    if let Some(s) = &doc.selection {
        check(s)?;
    }
    for m in &doc.masks {
        check(&m.mask)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prisme_core::{Layer, LayerKind};
    use rand::{Rng, SeedableRng};

    fn raster_layers(doc: &Document) -> Vec<Vec<f32>> {
        let b = doc.bounds();
        doc.all_layers().iter().filter_map(|l| l.raster().map(|g| g.read_rect(&b))).collect()
    }

    fn random_history(seed: u64) -> History {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut doc = Document::with_background("t", 700, 500, Some([0.2, 0.3, 0.4, 1.0]));
        // A second, high-precision layer with values f16 can't represent.
        let id = doc.alloc_layer_id();
        let mut g = TileGrid::rgba(PixelFormat::F32);
        g.set_pixel(3, 4, [0.123_456_79, 2.5, -0.001, 0.5]);
        doc.layers.push(Layer::new(id, "precise", LayerKind::Raster(Arc::new(g))));
        let bg = doc.layers[0].id;
        let mut h = History::new(doc);
        for i in 0..12 {
            let mut d = h.current().clone();
            let target = if rng.random_bool(0.5) { bg } else { id };
            let g = d.layer_mut(target).unwrap().raster_mut().unwrap();
            for _ in 0..40 {
                let (x, y) = (rng.random_range(0..700), rng.random_range(0..500));
                g.set_pixel(x, y, [rng.random(), rng.random(), rng.random(), rng.random()]);
            }
            if rng.random_bool(0.2) {
                d.orthogonal_transform(1);
            }
            h.push(format!("edit {i}"), d);
        }
        h.undo();
        h.undo();
        h
    }

    #[test]
    fn roundtrip_full_history_bit_exact() {
        let h = random_history(42);
        let bytes = save(&h, true).unwrap();
        let back = load(&bytes).unwrap();
        assert_eq!(back.cursor(), h.cursor());
        assert_eq!(back.entries().len(), h.entries().len());
        for (a, b) in h.entries().iter().zip(back.entries()) {
            assert_eq!(a.label, b.label);
            assert_eq!(a.doc.info.width, b.doc.info.width);
            let (la, lb) = (raster_layers(&a.doc), raster_layers(&b.doc));
            assert_eq!(la.len(), lb.len());
            for (x, y) in la.iter().zip(&lb) {
                assert!(x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()), "pixels differ in '{}'", a.label);
            }
        }
        // Shared tiles must stay shared after loading (dedup also in memory).
        assert_eq!(back.memory_bytes(), h.memory_bytes());
    }

    #[test]
    fn current_only_keeps_one_state() {
        let h = random_history(7);
        let full = save(&h, true).unwrap();
        let bytes = save(&h, false).unwrap();
        assert!(bytes.len() < full.len());
        let back = load(&bytes).unwrap();
        assert_eq!(back.entries().len(), 1);
        assert_eq!(back.cursor(), 0);
        assert_eq!(back.entries()[0].label, h.entries()[h.cursor()].label);
        assert_eq!(raster_layers(back.current()), raster_layers(h.current()));
    }

    #[test]
    fn preview_roundtrip() {
        let h = History::new(Document::with_background("p", 10, 10, None));
        let rgba: Vec<u8> = (0..4 * 6 * 4).map(|i| i as u8).collect();
        let bytes = save_with_preview(&h, true, Some((4, 6, &rgba))).unwrap();
        assert_eq!(read_preview(&bytes), Some((4, 6, rgba)));
        assert!(read_preview(&save(&h, true).unwrap()).is_none());
        assert!(load(&bytes).is_ok());
    }

    #[test]
    fn corrupt_files_error_without_panicking() {
        let h = random_history(3);
        let bytes = save_with_preview(&h, true, Some((1, 1, &[1, 2, 3, 4]))).unwrap();
        // Every truncation must fail cleanly.
        for len in (0..bytes.len()).step_by((bytes.len() / 200).max(1)) {
            assert!(load(&bytes[..len]).is_err());
            let _ = read_preview(&bytes[..len]);
        }
        // Random byte flips must never panic (they may or may not be detected).
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        for _ in 0..300 {
            let mut b = bytes.clone();
            for _ in 0..rng.random_range(1..8) {
                let i = rng.random_range(0..b.len());
                b[i] = rng.random();
            }
            let _ = load(&b);
            let _ = read_preview(&b);
        }
        assert!(load(b"PRISME\0\x01\x02\0\0\0\0\0\0\0").is_err());
        assert!(load(b"garbage").is_err());
    }
}
