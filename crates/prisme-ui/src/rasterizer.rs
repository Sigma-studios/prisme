//! Bridges text and vector layers into the GPU engine.

use prisme_core::layer::{Layer, LayerKind};
use prisme_core::{Document, IRect, TileGrid};
use prisme_gpu::LayerRasterizer;
use prisme_text::{FontDb, TextRenderer};
use prisme_vector::VectorRenderer;
use std::sync::Arc;

pub struct Rasterizer {
    pub text: TextRenderer,
    pub vector: VectorRenderer,
}

impl Rasterizer {
    pub fn new(fonts: Arc<FontDb>) -> Self {
        Self { text: TextRenderer::new(fonts), vector: VectorRenderer::new() }
    }

    /// Document-space bounds of a text/vector layer's rendering (None for other kinds).
    pub fn layer_bounds(&self, layer: &Layer) -> Option<IRect> {
        let b = match &layer.kind {
            LayerKind::Text(t) => self.text.bounds(t),
            LayerKind::Vector(v) => VectorRenderer::bounds(v),
            _ => return None,
        };
        Some(IRect::from_f32(b[0], b[1], b[2], b[3]).expand(1))
    }
}

impl LayerRasterizer for Rasterizer {
    fn rasterize(&self, layer: &Layer, _doc: &Document, mip: u32) -> Option<TileGrid> {
        match &layer.kind {
            LayerKind::Text(t) => self.text.rasterize(t, mip),
            LayerKind::Vector(v) => self.vector.rasterize(v, mip),
            _ => None,
        }
    }
}
