//! Hand-off of dab strokes to the host's GPU stroke buffer.
//!
//! With [`ToolCx::strokes`](crate::ToolCx) set, dab tools leave the tiles alone while painting
//! and emit [`StrokeOp`]s instead (dynamics and dab placement stay on the CPU). The host
//! rasterizes, accumulates and composites on the GPU, reads the result back when the stroke
//! ends and lands it with [`write_stroke`] as one history edit. [`replay_stroke`] applies the
//! same stroke with the CPU engine (no GPU, headless, or a readback that can't be awaited).

use crate::liquify::{LiquifyField, LiquifySpec};
use crate::paint::RetouchCanvas;
use prisme_brush::{Dab, StrokeCanvas, StrokeMode};
use prisme_core::tile::{Tile, TileCoord, TileGrid, TILE_SIZE};
use prisme_core::{Document, IRect, LayerId, MaskId};
use std::collections::HashSet;
use std::sync::Arc;

/// Where painting goes: a layer's pixels, its mask, or a saved (document-level) mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaintTarget {
    pub layer: LayerId,
    pub mask: bool,
    /// Painting on a saved (document-level) mask instead of a layer.
    pub saved: Option<MaskId>,
}

pub fn target_grid(doc: &mut Document, t: PaintTarget) -> Option<&mut TileGrid> {
    if let Some(mid) = t.saved {
        return doc.masks.iter_mut().find(|m| m.id == mid).map(|m| Arc::make_mut(&mut m.mask));
    }
    let l = doc.layer_mut(t.layer)?;
    if t.mask { l.mask.as_mut().map(|m| Arc::make_mut(&mut m.grid)) } else { l.raster_mut() }
}

/// Stroke start state: target, its snapshot, and what the dabs do.
#[derive(Clone, Debug)]
pub struct StrokeSpec {
    pub target: PaintTarget,
    /// Target pixels at stroke start (the working copy keeps these until the stroke lands).
    pub base: Arc<TileGrid>,
    /// Value of missing base tiles (1.0 for reveal-all masks).
    pub base_default: f32,
    pub selection: Option<Arc<TileGrid>>,
    pub mode: StrokeMode,
}

#[derive(Clone, Debug)]
pub enum StrokeOp {
    Begin(StrokeSpec),
    Dabs(Vec<Dab>),
    /// Stroke finished: land it as one history edit with this label.
    End(String),
    /// Start a Liquify session: a stroke lasting until `LiquifyEnd` / `Abort` whose preview is
    /// the target warped through a displacement field (see [`crate::liquify`]).
    LiquifyBegin(LiquifySpec),
    /// Field nodes changed: `nodes` rect with its row-major values.
    LiquifyField { nodes: IRect, data: Vec<[f32; 4]> },
    /// Bake the session: land the warped target as one history edit with this label.
    LiquifyEnd { label: String, field: Arc<LiquifyField> },
    /// Discard the in-progress stroke / session.
    Abort,
}

/// Apply a whole stroke with the CPU engine (target must still hold `spec.base`).
pub fn replay_stroke(doc: &mut Document, spec: &StrokeSpec, dabs: &[Dab]) {
    let Some(g) = target_grid(doc, spec.target) else { return };
    match spec.mode {
        StrokeMode::Paint { target, opacity, lock_alpha } => {
            let mut c = StrokeCanvas::new(spec.base.clone(), target, opacity, spec.selection.clone(), lock_alpha, spec.base_default);
            c.apply(dabs, g);
            c.finish(g);
        }
        StrokeMode::Retouch(m) => RetouchCanvas::new(m, spec.base.clone(), spec.selection.clone()).apply(dabs, g),
    }
}

/// Add the tiles `dabs` touch to `out`.
pub fn dab_tiles(dabs: &[Dab], out: &mut HashSet<TileCoord>) {
    for d in dabs {
        let b = d.bounds();
        out.extend(TileGrid::coords_in(&IRect::new(b[0], b[1], b[2], b[3])));
    }
}

/// Land read-back stroke pixels: `px` covers `rect` (`channels` per pixel, row-major); only
/// `tiles` are written so the stroke's bounding box doesn't materialize untouched tiles.
/// Writes go through the tiles' own format (f16/f32). Returns false on a layout mismatch.
pub fn write_stroke(doc: &mut Document, target: PaintTarget, base_default: f32, rect: IRect, px: &[f32], channels: usize, tiles: impl IntoIterator<Item = TileCoord>) -> bool {
    let Some(g) = target_grid(doc, target) else { return true };
    let ch = g.channels as usize;
    let w = rect.width() as usize;
    if ch != channels || px.len() != w * rect.height() as usize * ch {
        return false;
    }
    let def = base_default;
    for c in tiles {
        let tr = c.rect().intersect(&rect);
        if tr.is_empty() {
            continue;
        }
        if !g.tiles.contains_key(&c) {
            let mut t = Tile::new(g.format, g.channels);
            if def != 0.0 {
                t.fill([def; 4]);
            }
            g.insert(c, t);
        }
        g.tile_mut(c).with_f32_mut(|d| {
            let n = tr.width() as usize * ch;
            for y in tr.y0..tr.y1 {
                let dst = ((y - c.rect().y0) as usize * TILE_SIZE as usize + (tr.x0 - c.rect().x0) as usize) * ch;
                let src = ((y - rect.y0) as usize * w + (tr.x0 - rect.x0) as usize) * ch;
                d[dst..dst + n].copy_from_slice(&px[src..src + n]);
            }
        });
        if def == 0.0 && g.tiles[&c].is_clear() {
            g.tiles.remove(&c);
        }
    }
    true
}
