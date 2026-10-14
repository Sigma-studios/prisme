//! GPU brush strokes: dab tools hand their strokes to the engine (`prisme_gpu::stroke`)
//! instead of writing tiles. The canvas shows the stroke from the GPU; when it ends the
//! result is read back asynchronously and lands in the tiles as one history edit.
//!
//! Until it lands, anything that needs the document's pixels — a new stroke or tool
//! interaction, undo/redo, history jumps, any edit, save/export/close — first calls
//! [`DocState::finish_stroke`]. On native that blocks on the readback (a GPU→CPU copy of
//! the stroke rect: well under a frame). The web can't block, so there it replays the
//! recorded dabs with the CPU engine instead (bit-for-bit the reference semantics), and
//! input is never queued. Normally neither happens: [`poll`] lands the readback a frame or
//! two after pointer-up.

use crate::app::AppState;
use crate::doc::DocState;
use prisme_brush::Dab;
use prisme_core::tile::TileCoord;
use prisme_core::Document;
use prisme_gpu::{Engine, GpuStroke, Readback, UploadKey};
use prisme_tools::stroke::{dab_tiles, replay_stroke, write_stroke};
use prisme_tools::{PaintTarget, StrokeOp, StrokeSpec};
use std::collections::HashSet;

pub struct DocStroke {
    pub gpu: GpuStroke,
    spec: StrokeSpec,
    /// Every dab so far (CPU replay fallback).
    dabs: Vec<Dab>,
    tiles: HashSet<TileCoord>,
    /// Set once the tool ended the stroke; it then waits for `readback`.
    label: Option<String>,
    readback: Option<Readback>,
    device: prisme_gpu::wgpu::Device,
}

fn upload_key(t: PaintTarget) -> UploadKey {
    match t.saved {
        Some(m) => UploadKey::SavedMask(m.0),
        None if t.mask => UploadKey::LayerMask(t.layer.0),
        None => UploadKey::Content(t.layer.0),
    }
}

/// Apply a tool's stroke ops to the document's GPU stroke.
pub fn feed(d: &mut DocState, engine: &mut Engine, ops: Vec<StrokeOp>) {
    for op in ops {
        match op {
            StrokeOp::Begin(spec) => {
                d.finish_stroke();
                let gpu = engine.stroke_begin(upload_key(spec.target), spec.base.clone(), spec.base_default, spec.selection.clone(), spec.mode);
                let device = engine.ctx.device.clone();
                d.stroke = Some(DocStroke { gpu, spec, dabs: vec![], tiles: HashSet::new(), label: None, readback: None, device });
            }
            StrokeOp::Dabs(v) => {
                if let Some(s) = d.stroke.as_mut().filter(|s| s.label.is_none()) {
                    engine.stroke_dabs(&mut s.gpu, &v);
                    dab_tiles(&v, &mut s.tiles);
                    s.dabs.extend(v);
                    d.touch();
                }
            }
            StrokeOp::End(label) => {
                if let Some(s) = d.stroke.as_mut().filter(|s| s.label.is_none()) {
                    s.readback = engine.stroke_readback(&s.gpu);
                    s.label = Some(label);
                }
            }
        }
    }
}

/// Per frame: land finished strokes whose readback arrived, drop strokes whose tool was
/// reset mid-drag. Requests repaints while a readback is outstanding.
pub fn poll(s: &mut AppState, ctx: &egui::Context) {
    let busy = s.tools.get(&s.tool).is_some_and(|t| t.busy());
    let active = s.active;
    let mut pending = false;
    for (i, d) in s.docs.iter_mut().enumerate() {
        if d.stroke.as_ref().is_some_and(|st| st.label.is_none()) && !(busy && i == active) {
            d.stroke = None;
            d.touch();
        }
        pending |= d.poll_stroke();
    }
    if pending {
        ctx.request_repaint();
    }
}

impl DocState {
    /// A finished stroke is waiting for its readback.
    pub fn stroke_pending(&self) -> bool {
        self.stroke.as_ref().is_some_and(|s| s.label.is_some())
    }

    /// Land a finished stroke now: native waits for the readback, the web replays the dabs
    /// on the CPU if it hasn't arrived. No-op without a finished stroke.
    pub fn finish_stroke(&mut self) {
        if !self.stroke_pending() {
            return;
        }
        let s = self.stroke.take().unwrap();
        #[cfg(not(target_arch = "wasm32"))]
        let px = s.readback.as_ref().map(|rb| rb.wait(&s.device));
        #[cfg(target_arch = "wasm32")]
        let px = s.readback.as_ref().and_then(|rb| rb.try_take(&s.device));
        self.land(s, px);
    }

    /// Non-blocking: land the finished stroke if its readback arrived. True while pending.
    fn poll_stroke(&mut self) -> bool {
        let Some(s) = self.stroke.as_ref().filter(|s| s.label.is_some()) else { return false };
        let px = match &s.readback {
            Some(rb) => match rb.try_take(&s.device) {
                Some(px) => Some(px),
                None => return true,
            },
            None => None,
        };
        let s = self.stroke.take().unwrap();
        self.land(s, px);
        false
    }

    /// Write the stroke into the tiles as one history entry (and into an open working copy).
    fn land(&mut self, s: DocStroke, px: Option<Vec<f32>>) {
        let apply = |doc: &mut Document| {
            let written = match (&px, &s.readback) {
                (Some(px), Some(rb)) if !px.is_empty() => write_stroke(doc, &s.spec, rb.rect, px, s.spec.base.channels as usize, &s.tiles),
                _ => false,
            };
            if !written {
                replay_stroke(doc, &s.spec, &s.dabs);
            }
        };
        let mut doc = self.history.current().clone();
        apply(&mut doc);
        if let Some(w) = &mut self.working {
            apply(w);
        }
        self.history.push(s.label.clone().unwrap_or_default(), doc);
        self.touch();
    }
}
