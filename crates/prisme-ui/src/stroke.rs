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
//!
//! A Liquify session is a long-lived stroke of the same kind: the tool streams field updates
//! while it's open, and its bake (a bicubic GPU render of the changed area) lands the same
//! way; the CPU fallback re-warps with `prisme_tools::liquify::apply_cpu`.
//!
//! A Free Transform preview is a stroke that never lands: it lives while the tool's session
//! is open, and applying it resamples the layer on the CPU in the tool.
//!
//! Strokes on f16 targets (the usual case) come back as finished f16 tiles converted on the
//! GPU (`prisme_gpu::pack`, bit-identical to converting an f32 read-back here): half the bytes,
//! no conversion, and the same tile goes into the history and the working copy. Their GPU
//! bits are then handed to the tile store ([`DocState::adopt`], taken by the canvas before it
//! renders), so the landed tiles aren't uploaded again.

use crate::app::AppState;
use crate::doc::DocState;
use prisme_brush::Dab;
use prisme_core::tile::{TileCoord, TileGrid};
use prisme_core::Document;
use prisme_core::tile::Tile;
use prisme_core::PixelFormat;
use prisme_gpu::{Engine, GpuStroke, PackedTiles, Readback, UploadKey};
use prisme_tools::liquify::{apply_cpu, LiquifyField};
use prisme_tools::stroke::{dab_tiles, replay_stroke, target_grid, write_stroke};
use prisme_tools::{LiquifySpec, PaintTarget, StrokeOp, StrokeSpec, ToolKey, ToolKind};
use std::collections::HashSet;
use std::sync::Arc;

/// What lands the stroke when there's no readback (CPU reference).
enum Replay {
    /// Every dab so far.
    Dabs(StrokeSpec, Vec<Dab>),
    /// The final field (set by `LiquifyEnd`).
    Liquify(LiquifySpec, Option<Arc<LiquifyField>>),
    /// A preview that never lands (Free Transform applies itself on the CPU).
    None,
}

/// How a finished stroke comes back from the GPU.
enum Back {
    /// f32 pixels per tile region (f32 targets), or the Liquify bake.
    F32(Vec<Readback>),
    /// Whole tiles already in f16 (f16 targets).
    F16(PackedTiles),
}

/// What arrived.
enum Landed {
    F32(Vec<Vec<f32>>),
    F16(Vec<Tile>),
}

impl Back {
    fn ready(&self, device: &prisme_gpu::wgpu::Device) -> bool {
        match self {
            Back::F32(rbs) => rbs.iter().all(|rb| rb.ready(device)),
            Back::F16(p) => p.ready(device),
        }
    }

    fn take(&self, device: &prisme_gpu::wgpu::Device) -> Option<Landed> {
        match self {
            Back::F32(rbs) => rbs.iter().map(|rb| rb.try_take(device)).collect::<Option<Vec<_>>>().map(Landed::F32),
            Back::F16(p) => p.take(device).map(Landed::F16),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self, device: &prisme_gpu::wgpu::Device) -> Option<Landed> {
        match self {
            Back::F32(rbs) => Some(Landed::F32(rbs.iter().map(|rb| rb.wait(device)).collect())),
            Back::F16(p) => {
                p.wait(device);
                p.take(device).map(Landed::F16)
            }
        }
    }
}

pub struct DocStroke {
    pub gpu: GpuStroke,
    target: PaintTarget,
    base_default: f32,
    channels: usize,
    replay: Replay,
    tiles: HashSet<TileCoord>,
    /// Set once the tool ended the stroke; it then waits for `readback`.
    label: Option<String>,
    readback: Option<Back>,
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
                let (target, base_default, channels) = (spec.target, spec.base_default, spec.base.channels as usize);
                let replay = Replay::Dabs(spec, vec![]);
                d.stroke = Some(DocStroke { gpu, target, base_default, channels, replay, tiles: HashSet::new(), label: None, readback: None, device: engine.ctx.device.clone() });
            }
            StrokeOp::Dabs(v) => {
                if let Some(s) = d.stroke.as_mut().filter(|s| s.label.is_none())
                    && let Replay::Dabs(_, dabs) = &mut s.replay {
                        engine.stroke_dabs(&mut s.gpu, &v);
                        dab_tiles(&v, &mut s.tiles);
                        dabs.extend(v);
                        d.touch();
                    }
            }
            StrokeOp::End(label) => {
                if let Some(s) = d.stroke.as_mut().filter(|s| s.label.is_none()) {
                    s.readback = engine.stroke_pack(&s.gpu).map(Back::F16).or_else(|| engine.stroke_readback(&s.gpu).map(Back::F32));
                    s.label = Some(label);
                }
            }
            StrokeOp::LiquifyBegin(spec) => {
                d.finish_stroke();
                let gpu = engine.liquify_begin(upload_key(spec.target), spec.base.clone(), spec.selection.clone(), spec.rect, spec.step, spec.nodes);
                let (target, channels) = (spec.target, spec.base.channels as usize);
                let replay = Replay::Liquify(spec, None);
                d.stroke = Some(DocStroke { gpu, target, base_default: 0.0, channels, replay, tiles: HashSet::new(), label: None, readback: None, device: engine.ctx.device.clone() });
                d.touch();
            }
            StrokeOp::LiquifyField { nodes, data } => {
                if let Some(s) = d.stroke.as_mut().filter(|s| s.label.is_none() && s.gpu.is_liquify()) {
                    engine.liquify_update(&mut s.gpu, nodes, &data);
                    d.touch();
                }
            }
            StrokeOp::LiquifyEnd { label, field } => {
                if let Some(s) = d.stroke.as_mut().filter(|s| s.label.is_none() && s.gpu.is_liquify()) {
                    let r = field.active_rect();
                    s.tiles = TileGrid::coords_in(&r).collect();
                    s.readback = engine.liquify_readback(&s.gpu, r, true).map(|rb| Back::F32(vec![rb]));
                    if let Replay::Liquify(_, f) = &mut s.replay {
                        *f = Some(field);
                    }
                    s.label = Some(label);
                }
            }
            StrokeOp::XformBegin(spec) => {
                d.finish_stroke();
                let gpu = engine.xform_begin(UploadKey::Content(spec.layer.0), spec.source.clone(), spec.rect, spec.mask.clone());
                let target = PaintTarget { layer: spec.layer, mask: false, saved: None };
                d.stroke = Some(DocStroke { gpu, target, base_default: 0.0, channels: 4, replay: Replay::None, tiles: HashSet::new(), label: None, readback: None, device: engine.ctx.device.clone() });
                d.touch();
            }
            StrokeOp::XformQuad(h) => {
                if let Some(s) = d.stroke.as_mut().filter(|s| s.label.is_none() && s.gpu.is_xform()) {
                    engine.xform_update(&mut s.gpu, &h);
                    d.touch();
                }
            }
            StrokeOp::Abort => {
                if d.stroke.as_ref().is_some_and(|s| s.label.is_none()) {
                    d.stroke = None;
                    d.touch();
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
    // Liquify: the freeze tint follows its option; a session whose GPU stroke went away
    // (document switched or closed) is dropped.
    let show = s.settings.tools.liquify.show_mask;
    if s.tool == ToolKind::Liquify && busy && s.gpu.is_some() {
        let live = s.docs.get_mut(active).and_then(|d| d.stroke.as_mut()).filter(|st| st.label.is_none() && st.gpu.is_liquify());
        match (live, s.gpu.as_mut()) {
            (Some(st), Some(g)) => {
                if g.engine.liquify_show_mask(&mut st.gpu, show) {
                    s.docs[active].touch();
                }
            }
            _ => {
                if let Some(t) = s.tools.get_mut(&ToolKind::Liquify) {
                    t.reset();
                }
            }
        }
    }
}

/// Apply an open Liquify session (before switching tools, document actions, quitting…).
pub fn commit_liquify(s: &mut AppState) {
    if s.tool == ToolKind::Liquify && s.tools.get(&s.tool).is_some_and(|t| t.busy()) {
        crate::commands::tool_key(s, &ToolKey::Enter);
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
        let px = s.readback.as_ref().and_then(|b| b.wait(&s.device));
        #[cfg(target_arch = "wasm32")]
        let px = s.readback.as_ref().filter(|b| b.ready(&s.device)).and_then(|b| b.take(&s.device));
        self.land(s, px);
    }

    /// Non-blocking: land the finished stroke if its readback arrived. True while pending.
    fn poll_stroke(&mut self) -> bool {
        let Some(s) = self.stroke.as_ref().filter(|s| s.label.is_some()) else { return false };
        let px = match &s.readback {
            Some(b) if !b.ready(&s.device) => return true,
            Some(b) => b.take(&s.device),
            None => None,
        };
        let s = self.stroke.take().unwrap();
        self.land(s, px);
        false
    }

    /// Write the stroke into the tiles as one history entry (and into an open working copy).
    fn land(&mut self, s: DocStroke, px: Option<Landed>) {
        let (px, half) = match px {
            Some(Landed::F32(v)) => (Some(v), None),
            Some(Landed::F16(t)) => (None, Some(t)),
            None => (None, None),
        };
        // All parts must have arrived intact, or the CPU replays the whole stroke.
        let parts = match (&px, &s.readback) {
            (Some(px), Some(Back::F32(rbs))) if px.len() == rbs.len() && px.iter().zip(rbs).all(|(p, rb)| p.len() == rb.rect.area() as usize * s.channels) => {
                Some(rbs.iter().zip(px))
            }
            _ => None,
        };
        // f16 tiles: one shared tile per coordinate (None: the stroke cleared it).
        let tiles: Option<Vec<(TileCoord, Option<Arc<Tile>>)>> = match (half, &s.readback) {
            (Some(t), Some(Back::F16(p))) if t.len() == p.coords.len() && p.channels() as usize == s.channels => Some(
                p.coords
                    .iter()
                    .zip(t)
                    .filter(|(c, _)| s.tiles.contains(c))
                    .map(|(c, t)| (*c, (s.base_default != 0.0 || !t.is_clear()).then(|| Arc::new(t))))
                    .collect(),
            ),
            _ => None,
        };
        let apply = |doc: &mut Document| {
            let written = match (parts.clone(), &tiles) {
                (Some(parts), _) => {
                    for (rb, px) in parts {
                        write_stroke(doc, s.target, s.base_default, rb.rect, px, s.channels, TileGrid::coords_in(&rb.rect).filter(|c| s.tiles.contains(c)));
                    }
                    true
                }
                (None, Some(tiles)) => match target_grid(doc, s.target).filter(|g| g.format == PixelFormat::F16 && g.channels as usize == s.channels) {
                    Some(g) => {
                        for (c, t) in tiles {
                            match t {
                                Some(t) => g.tiles.insert(*c, t.clone()),
                                None => g.tiles.remove(c),
                            };
                        }
                        true
                    }
                    None => false,
                },
                (None, None) => false,
            };
            if !written {
                match &s.replay {
                    Replay::Dabs(spec, dabs) => replay_stroke(doc, spec, dabs),
                    Replay::Liquify(spec, Some(field)) => apply_cpu(doc, spec, field),
                    Replay::Liquify(_, None) | Replay::None => {}
                }
            }
        };
        let mut doc = self.history.current().clone();
        apply(&mut doc);
        if let Some(w) = &mut self.working {
            apply(w);
        }
        let adopt = |doc: &mut Document| target_grid(doc, s.target).is_some_and(|g| g.format == PixelFormat::F16);
        if let (Some(tiles), true) = (&tiles, adopt(&mut doc)) {
            // Hand the landed tiles' GPU bits to the tile store (ids in `coords` order).
            let ids: std::collections::HashMap<TileCoord, u64> = tiles.iter().filter_map(|(c, t)| t.as_ref().map(|t| (*c, t.id()))).collect();
            if let Some(Back::F16(p)) = s.readback {
                let ids = p.coords.iter().map(|c| ids.get(c).copied().unwrap_or(0)).collect();
                self.adopt = Some((p, ids));
            }
        }
        self.history.push(s.label.clone().unwrap_or_default(), doc);
        self.touch();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prisme_core::{IRect, PixelFormat};
    use prisme_tools::{make_tool, Phase, Tool, ToolCx, ToolInput, ToolOptions, ToolOutcome};

    /// Drive a tool like the canvas does (working copy, stroke ops fed to the engine).
    fn input(d: &mut DocState, e: &mut Engine, t: &mut dyn Tool, o: &ToolOptions, ev: Result<ToolInput, ToolKey>) -> ToolOutcome {
        d.working.get_or_insert_with(|| d.history.current().clone());
        let mut ops = vec![];
        let out = {
            let mut cx = ToolCx { doc: d.working.as_mut().unwrap(), opts: o, fg: [0.0; 4], bg: [1.0; 4], strokes: Some(&mut ops), view: d.view_rect };
            match &ev {
                Ok(ev) => t.input(&mut cx, ev),
                Err(k) => t.key(&mut cx, k),
            }
        };
        feed(d, e, ops);
        if let ToolOutcome::Commit(label) = &out {
            d.commit_working(label);
        }
        if !t.busy() {
            d.working = None;
        }
        out
    }

    fn drag(d: &mut DocState, e: &mut Engine, t: &mut dyn Tool, o: &ToolOptions, a: [f32; 2], b: [f32; 2]) {
        for i in 0..=20 {
            let f = i as f32 / 20.0;
            let ph = match i {
                0 => Phase::Down,
                20 => Phase::Up,
                _ => Phase::Move,
            };
            let mut ev = ToolInput::mouse(ph, [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]);
            ev.time = i as f64 * 0.016;
            input(d, e, t, o, Ok(ev));
        }
    }

    fn layer(d: &Document) -> Vec<f32> {
        d.active().unwrap().raster().unwrap().read_rect(&IRect::new(0, 0, 200, 200))
    }

    #[test]
    fn liquify_session_lands_one_history_entry() {
        let Some(ctx) = pollster::block_on(prisme_gpu::GpuContext::headless()) else { return eprintln!("no GPU adapter; skipping") };
        let mut e = Engine::new(ctx);
        let mut doc = Document::with_background("t", 200, 200, Some([1.0; 4]));
        let id = doc.active_layer.unwrap();
        let g = doc.layer_mut(id).unwrap().raster_mut().unwrap();
        g.map_rect(&IRect::new(80, 80, 120, 120), |x, y, p| p.copy_from_slice(&[x as f32 / 200.0, 0.0, y as f32 / 200.0, 1.0]));
        assert_eq!(g.format, PixelFormat::F16);
        let mut d = DocState::new(doc);
        let before = layer(d.history.current());
        let mut o = ToolOptions::default();
        o.liquify.size = 90.0;
        o.liquify.pressure = 1.0;
        let mut t = make_tool(ToolKind::Liquify);
        // Two strokes; the second one is undone inside the session.
        drag(&mut d, &mut e, t.as_mut(), &o, [100.0, 100.0], [125.0, 100.0]);
        drag(&mut d, &mut e, t.as_mut(), &o, [100.0, 60.0], [100.0, 30.0]);
        assert!(d.stroke.as_ref().is_some_and(|s| s.gpu.is_liquify() && s.label.is_none()));
        assert_eq!(input(&mut d, &mut e, t.as_mut(), &o, Err(ToolKey::Undo)), ToolOutcome::Preview);
        assert_eq!(d.history.cursor(), 0, "nothing lands while the session is open");
        // Apply: one "Liquify" entry, the square pushed right, matching the CPU bake.
        assert_eq!(input(&mut d, &mut e, t.as_mut(), &o, Err(ToolKey::Enter)), ToolOutcome::Preview);
        assert!(!t.busy() && d.stroke_pending());
        d.finish_stroke();
        assert_eq!(d.history.cursor(), 1);
        assert_eq!(d.history.entries()[1].label, "Liquify");
        let after = d.history.current().active().unwrap().raster().unwrap().clone();
        assert!(after.get_pixel(118, 100)[1] < 0.1, "pushed right: {:?}", after.get_pixel(118, 100));
        assert!(after.get_pixel(100, 40)[1] > 0.99, "undone stroke left no trace");
        // Undo restores the pixels exactly.
        d.undo();
        assert_eq!(layer(d.history.current()), before);
        // Escape discards a session without touching the history.
        drag(&mut d, &mut e, t.as_mut(), &o, [100.0, 100.0], [125.0, 100.0]);
        assert_eq!(input(&mut d, &mut e, t.as_mut(), &o, Err(ToolKey::Escape)), ToolOutcome::Cancel);
        assert!(d.stroke.is_none() && d.history.cursor() == 0);
        assert_eq!(layer(d.doc()), before);
    }

    #[test]
    fn f16_stroke_lands_as_shared_tiles_and_is_adopted_without_upload() {
        use prisme_gpu::{EvalFormat, EvalRequest, Readback, RenderSource};
        let Some(ctx) = pollster::block_on(prisme_gpu::GpuContext::headless()) else { return eprintln!("no GPU adapter; skipping") };
        let mut e = Engine::new(ctx.clone());
        let mut doc = Document::with_background("t", 600, 400, Some([1.0; 4]));
        let id = doc.active_layer.unwrap();
        assert_eq!(doc.layer(id).unwrap().raster().unwrap().format, PixelFormat::F16);
        doc.layer_mut(id).unwrap().raster_mut().unwrap().map_rect(&IRect::new(50, 50, 300, 300), |x, y, p| p.copy_from_slice(&[x as f32 / 600.0, y as f32 / 400.0, 0.4, 1.0]));
        let mut d = DocState::new(doc);
        let rect = IRect::new(0, 0, 600, 400);
        let render = |e: &mut Engine, doc: &Document| {
            let img = e.evaluate(&EvalRequest { doc, rect, mip: 0, format: EvalFormat::F16, source: RenderSource::LayerStack }).unwrap();
            Readback::start(&e.ctx.device, &e.ctx.queue, &img, rect).wait(&e.ctx.device)
        };
        render(&mut e, d.history.current());
        let mut o = ToolOptions::default();
        o.brush.size = 80.0;
        let mut t = make_tool(ToolKind::Brush);
        drag(&mut d, &mut e, t.as_mut(), &o, [40.0, 60.0], [560.0, 340.0]);
        assert!(d.stroke_pending());
        d.finish_stroke();
        assert_eq!(d.history.cursor(), 1);
        // Landed through the f16 path: the store handoff is queued with the landed tiles' ids.
        let (p, ids) = d.adopt.take().expect("f16 landing");
        let g = d.history.current().layer(id).unwrap().raster().unwrap().clone();
        let landed: Vec<u64> = ids.iter().copied().filter(|&i| i != 0).collect();
        assert!(landed.len() > 4);
        assert!(landed.iter().all(|i| g.tiles.values().any(|t| t.id() == *i)), "ids are the document's tiles");
        // The store takes them over: the next render uploads nothing and matches a fresh engine.
        let before = e.tile_stats();
        e.adopt_tiles(d.history.current().uid, &p, &ids);
        let px = render(&mut e, d.history.current());
        assert_eq!(e.tile_stats().0, before.0, "landed tiles were uploaded again");
        assert_eq!(px, render(&mut Engine::new(ctx), d.history.current()));
    }

    #[test]
    fn transform_previews_on_the_gpu_and_applies_once() {
        let Some(ctx) = pollster::block_on(prisme_gpu::GpuContext::headless()) else { return eprintln!("no GPU adapter; skipping") };
        let mut e = Engine::new(ctx);
        let mut doc = Document::with_background("t", 200, 200, None);
        let id = doc.active_layer.unwrap();
        let g = doc.layer_mut(id).unwrap().raster_mut().unwrap();
        g.map_rect(&IRect::new(40, 40, 80, 80), |_, _, p| p.copy_from_slice(&[1.0, 0.0, 0.0, 1.0]));
        let mut d = DocState::new(doc);
        let o = ToolOptions::default();
        let mut t = make_tool(ToolKind::Transform);
        // Drag the box 50 px right: the preview is a GPU stroke, the working layer is emptied.
        drag(&mut d, &mut e, t.as_mut(), &o, [60.0, 60.0], [110.0, 60.0]);
        assert!(t.busy() && d.stroke.as_ref().is_some_and(|s| s.gpu.is_xform() && s.gpu.generation > 0));
        assert!(d.working.as_ref().unwrap().active().unwrap().raster().unwrap().tiles.is_empty());
        assert_eq!(d.history.cursor(), 0);
        // Enter resamples once and commits; the preview goes away.
        assert_eq!(input(&mut d, &mut e, t.as_mut(), &o, Err(ToolKey::Enter)), ToolOutcome::Commit("Free Transform".into()));
        assert_eq!(d.history.cursor(), 1);
        assert!(d.stroke.is_none() && !t.busy());
        let g = d.history.current().active().unwrap().raster().unwrap().clone();
        assert!(g.get_pixel(110, 60)[3] > 0.99 && g.get_pixel(60, 60)[3] < 0.01, "moved: {:?} {:?}", g.get_pixel(110, 60), g.get_pixel(60, 60));
        // Escape restores the layer.
        drag(&mut d, &mut e, t.as_mut(), &o, [110.0, 60.0], [150.0, 100.0]);
        assert_eq!(input(&mut d, &mut e, t.as_mut(), &o, Err(ToolKey::Escape)), ToolOutcome::Cancel);
        assert!(d.stroke.is_none());
        assert_eq!(d.doc().active().unwrap().raster().unwrap().read_rect(&IRect::new(0, 0, 200, 200)), g.read_rect(&IRect::new(0, 0, 200, 200)));
    }
}
