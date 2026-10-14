//! Bridges text and vector layers into the GPU engine.

use prisme_core::layer::{Layer, LayerKind};
use prisme_core::{Document, TileGrid};
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
