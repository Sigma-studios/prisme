//! Prisme image IO.
//!
//! Decodes image files into straight-alpha, scene-linear working-space (linear
//! Rec.2020) RGBA, builds documents from them, and reads/writes the chunked `.prisme`
//! project container (see [`container`]).
//!
//! Decoders are kept in a small registry ([`decoders`]); the decoder is chosen by magic
//! bytes first, then by file extension.

mod color;
pub mod container;
mod formats;

use prisme_core::geom::{Affine2, IRect};
use prisme_core::tile::{Tile, TileCoord, TileGrid, TILE_SIZE};
use prisme_core::{Document, Layer, LayerId, LayerKind, PixelFormat, VectorLayer};
use std::sync::Arc;

pub use formats::{decoders, ImageDecoder};

/// A decoded raster image in the working space.
#[derive(Clone, Debug)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    /// Straight-alpha linear working-space RGBA, row-major, len = w*h*4.
    pub pixels: Vec<f32>,
    pub format: PixelFormat,
    /// Human readable source colour description, e.g. "sRGB (untagged)", ICC description, "Linear Rec.709".
    pub color_info: String,
    pub source_bit_depth: u8,
}

impl DecodedImage {
    /// Storage precision for a source: f16 is lossless for <= 8-bit integer sources,
    /// everything else (16-bit integer, float) needs f32.
    pub fn format_for(source_bit_depth: u8, float: bool) -> PixelFormat {
        if float || source_bit_depth > 8 { PixelFormat::F32 } else { PixelFormat::F16 }
    }

    /// Image with the storage format derived from the source depth.
    pub(crate) fn new(width: u32, height: u32, pixels: Vec<f32>, color_info: String, bits: u8, float: bool) -> Self {
        debug_assert_eq!(pixels.len(), width as usize * height as usize * 4);
        Self { width, height, pixels, format: Self::format_for(bits, float), color_info, source_bit_depth: bits }
    }

    /// Apply an EXIF orientation (1..=8) in place.
    pub(crate) fn oriented(mut self, orientation: Option<u16>) -> Self {
        if let Some(o) = orientation {
            let px = std::mem::take(&mut self.pixels);
            let (px, w, h) = color::apply_orientation(px, self.width, self.height, o);
            self.pixels = px;
            self.width = w;
            self.height = h;
        }
        self
    }

    /// Downscaled copy whose longest side is <= `max_side`.
    pub(crate) fn downscaled(self, max_side: u32) -> Self {
        if self.width.max(self.height) <= max_side {
            return self;
        }
        let (pixels, width, height) = color::downscale(&self.pixels, self.width, self.height, max_side);
        Self { width, height, pixels, ..self }
    }
}

/// Result of decoding a file.
#[derive(Clone, Debug)]
pub enum Decoded {
    Raster(DecodedImage),
    /// Multi-layer sources (multi-page TIFF). (name, image)
    Layers(Vec<(String, DecodedImage)>),
    /// SVG kept as vector source; size in user units.
    Vector { svg: String, width: f32, height: f32 },
}

#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("unsupported format: {0}")]
    Unsupported(String),
    #[error("decode error: {0}")]
    Decode(String),
    #[error("encode error: {0}")]
    Encode(String),
    #[error("invalid project file: {0}")]
    Corrupt(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl IoError {
    pub(crate) fn decode(e: impl std::fmt::Display) -> Self {
        IoError::Decode(e.to_string())
    }
}

/// Lowercased extension of a file name (without the dot).
fn extension(file_name: &str) -> String {
    std::path::Path::new(file_name)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default()
}

/// File name without directories and extension.
fn file_stem(file_name: &str) -> String {
    std::path::Path::new(file_name)
        .file_stem()
        .and_then(|e| e.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("Untitled")
        .to_string()
}

/// Candidate decoders in priority order: magic + extension, magic only, extension only.
fn candidates(bytes: &[u8], file_name: &str) -> Vec<&'static dyn ImageDecoder> {
    let ext = extension(file_name);
    let header = &bytes[..bytes.len().min(64)];
    let all = decoders();
    let mut out: Vec<&'static dyn ImageDecoder> = Vec::new();
    let mut push = |d: &'static dyn ImageDecoder| {
        if !out.iter().any(|o| o.name() == d.name()) {
            out.push(d);
        }
    };
    all.iter().filter(|d| d.sniff(header) && d.extensions().contains(&ext.as_str())).for_each(|d| push(*d));
    all.iter().filter(|d| d.sniff(header)).for_each(|d| push(*d));
    all.iter().filter(|d| d.extensions().contains(&ext.as_str())).for_each(|d| push(*d));
    out
}

/// Decode an image file. The format is detected from magic bytes, then the extension.
pub fn decode(bytes: &[u8], file_name: &str) -> Result<Decoded, IoError> {
    let cands = candidates(bytes, file_name);
    if cands.is_empty() {
        return Err(IoError::Unsupported(format!("unrecognised file type: {file_name}")));
    }
    let mut first_err = None;
    for d in cands {
        match d.decode(bytes) {
            Ok(r) => return Ok(r),
            Err(e) => {
                log::debug!("{} decoder failed on {file_name}: {e}", d.name());
                first_err.get_or_insert(e);
            }
        }
    }
    Err(first_err.expect("at least one candidate"))
}

/// Fast low-resolution preview for instant display while the full decode runs (JPEG:
/// DCT-domain downscale; camera RAW: embedded JPEG preview; others: `None`).
/// Longest side <= max_side.
pub fn decode_preview(bytes: &[u8], file_name: &str, max_side: u32) -> Option<DecodedImage> {
    let d = candidates(bytes, file_name).into_iter().next()?;
    d.decode_preview(bytes, max_side.max(1)).map(|p| p.downscaled(max_side.max(1)))
}

/// Lowercase extensions (without dot) of all decodable formats, plus `prisme`.
pub fn supported_extensions() -> &'static [&'static str] {
    static EXTS: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    EXTS.get_or_init(|| {
        let mut v: Vec<&'static str> = vec!["prisme"];
        for d in decoders() {
            for e in d.extensions() {
                if !v.contains(e) {
                    v.push(e);
                }
            }
        }
        v
    })
}

/// True for Prisme project files (`.prisme`).
pub fn is_project_file(file_name: &str) -> bool {
    extension(file_name) == "prisme"
}

// --------------------------------------------------------------------------------
// Document building
// --------------------------------------------------------------------------------

/// Build a tile grid from a dense RGBA image placed at (`ox`, `oy`). Tiles are built in
/// parallel on native; fully transparent tiles are skipped.
fn grid_from_dense(img: &DecodedImage, ox: i32, oy: i32) -> TileGrid {
    let rect = IRect::new(ox, oy, ox + img.width as i32, oy + img.height as i32);
    let coords: Vec<TileCoord> = TileGrid::coords_in(&rect).collect();
    let format = img.format;
    let w = img.width as usize;
    let build = |c: &TileCoord| -> Option<(TileCoord, Tile)> {
        let tr = c.rect().intersect(&rect);
        if tr.is_empty() {
            return None;
        }
        let ts = TILE_SIZE as usize;
        let mut buf = vec![0.0f32; ts * ts * 4];
        let mut any = false;
        let n = tr.width() as usize * 4;
        for y in tr.y0..tr.y1 {
            let src = &img.pixels[(((y - oy) as usize) * w + (tr.x0 - ox) as usize) * 4..][..n];
            let dst = &mut buf[(((y - c.y * TILE_SIZE as i32) as usize) * ts + (tr.x0 - c.x * TILE_SIZE as i32) as usize) * 4..][..n];
            dst.copy_from_slice(src);
            any |= src.iter().skip(3).step_by(4).any(|&a| a != 0.0);
        }
        any.then(|| (*c, Tile::from_f32(format, 4, &buf)))
    };
    #[cfg(not(target_arch = "wasm32"))]
    let tiles: Vec<(TileCoord, Tile)> = {
        use rayon::prelude::*;
        coords.par_iter().filter_map(build).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let tiles: Vec<(TileCoord, Tile)> = coords.iter().filter_map(build).collect();
    let mut g = TileGrid::rgba(format);
    g.tiles.reserve(tiles.len());
    for (c, t) in tiles {
        g.insert(c, t);
    }
    g
}

/// Layers to create for a decoded source: (name, kind, format, size).
fn decoded_layers(file_name: &str, decoded: Decoded, doc_size: Option<(u32, u32)>) -> Vec<(String, LayerKind, PixelFormat)> {
    let stem = file_stem(file_name);
    let center = |w: u32, h: u32| -> (i32, i32) {
        match doc_size {
            Some((dw, dh)) => ((dw as i32 - w as i32) / 2, (dh as i32 - h as i32) / 2),
            None => (0, 0),
        }
    };
    let raster = |name: String, img: DecodedImage| {
        let (ox, oy) = center(img.width, img.height);
        let g = grid_from_dense(&img, ox, oy);
        (name, LayerKind::Raster(Arc::new(g)), img.format)
    };
    match decoded {
        Decoded::Raster(img) => vec![raster(stem, img)],
        Decoded::Layers(v) => v.into_iter().map(|(n, img)| raster(n, img)).collect(),
        Decoded::Vector { svg, width, height } => {
            let (ox, oy) = match doc_size {
                Some((dw, dh)) => ((dw as f32 - width) * 0.5, (dh as f32 - height) * 0.5),
                None => (0.0, 0.0),
            };
            let transform = if doc_size.is_some() { Affine2::translate(ox.round(), oy.round()) } else { Affine2::IDENTITY };
            let v = VectorLayer { svg: Arc::new(svg), size: [width, height], transform };
            vec![(stem, LayerKind::Vector(v), PixelFormat::F16)]
        }
    }
}

fn decoded_size(decoded: &Decoded) -> (u32, u32) {
    match decoded {
        Decoded::Raster(i) => (i.width, i.height),
        Decoded::Layers(v) => v.iter().fold((0, 0), |(w, h), (_, i)| (w.max(i.width), h.max(i.height))),
        Decoded::Vector { width, height, .. } => (width.ceil() as u32, height.ceil() as u32),
    }
}

fn decoded_color_info(decoded: &Decoded) -> Option<String> {
    match decoded {
        Decoded::Raster(i) => Some(i.color_info.clone()),
        Decoded::Layers(v) => v.first().map(|(_, i)| i.color_info.clone()),
        Decoded::Vector { .. } => Some("sRGB (SVG)".into()),
    }
}

/// Build a new document from decoded content: raster -> one raster layer named after
/// the file; Layers -> one layer each (bottom = first); Vector -> a vector layer on a
/// transparent document of the SVG size (rounded up).
pub fn document_from_decoded(file_name: &str, decoded: Decoded) -> Document {
    let (w, h) = decoded_size(&decoded);
    let profile = decoded_color_info(&decoded);
    let mut doc = Document::new(file_stem(file_name), w.max(1), h.max(1), PixelFormat::F16);
    doc.info.source_profile = profile;
    let mut format = PixelFormat::F16;
    for (name, kind, f) in decoded_layers(file_name, decoded, None) {
        format = format.max(f);
        let id = doc.alloc_layer_id();
        doc.layers.push(Layer::new(id, name, kind));
        doc.active_layer = Some(id);
    }
    doc.info.format = format;
    doc
}

/// Add decoded content as new layer(s) on top of an existing document (used for
/// "Place"/paste/drag-drop into an open doc). Content is centred on the canvas.
pub fn add_decoded_as_layers(doc: &mut Document, file_name: &str, decoded: Decoded) -> Vec<LayerId> {
    let prepared = prepare_layers(file_name, decoded, (doc.info.width, doc.info.height));
    add_prepared_layers(doc, prepared)
}

/// Layers built from decoded content for a document of a given size, ready to be added with
/// [`add_prepared_layers`]. Building the tile grids is the expensive part of placing an image,
/// so this split lets it run off the UI thread (it is serializable through
/// [`container::to_transport`] for Web Workers).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PreparedLayers {
    pub color_info: Option<String>,
    /// (name, content, storage format), bottom first.
    pub layers: Vec<(String, LayerKind, PixelFormat)>,
}

/// Build the layers for placing `decoded` centred on a `doc_size` canvas.
pub fn prepare_layers(file_name: &str, decoded: Decoded, doc_size: (u32, u32)) -> PreparedLayers {
    let color_info = decoded_color_info(&decoded);
    PreparedLayers { color_info, layers: decoded_layers(file_name, decoded, Some(doc_size)) }
}

/// Add layers built by [`prepare_layers`] on top of `doc`; returns the new layer ids.
pub fn add_prepared_layers(doc: &mut Document, prepared: PreparedLayers) -> Vec<LayerId> {
    if doc.info.source_profile.is_none() {
        doc.info.source_profile = prepared.color_info;
    }
    let mut ids = vec![];
    for (name, kind, f) in prepared.layers {
        doc.info.format = doc.info.format.max(f);
        let id = doc.alloc_layer_id();
        doc.insert_layer_above(Layer::new(id, name, kind), None);
        doc.active_layer = Some(id);
        ids.push(id);
    }
    ids
}

#[cfg(test)]
mod tests;
