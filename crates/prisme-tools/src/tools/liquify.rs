use super::*;
use crate::input::Phase;
use crate::liquify::{apply_cpu, warp_rect, working_rect, LiquifyDab, LiquifyField, LiquifyMode, LiquifySpec};
use crate::stroke::{target_grid, PaintTarget, StrokeOp};
use prisme_brush::{BrushSettings, Dab, Dynamic, InputSample, Stroke};
use prisme_core::layer::LayerKind;
use prisme_core::IRect;
use std::sync::Arc;

/// Field nodes before one stroke (in-session undo/redo).
struct Snap {
    nodes: IRect,
    data: Vec<[f32; 4]>,
}

/// Session undo budget (bytes of node snapshots).
const UNDO_BYTES: usize = 256 << 20;

struct Session {
    spec: LiquifySpec,
    field: LiquifyField,
    /// The field as of the last stroke end (undo snapshots are cut from it).
    pre: Vec<[f32; 4]>,
    undo: Vec<Snap>,
    redo: Vec<Snap>,
    /// The host warps on the GPU; otherwise the working copy is re-warped on the CPU.
    gpu: bool,
}

struct Drag {
    stroke: Stroke,
    last: Option<[f32; 2]>,
    /// Nodes changed by this stroke.
    nodes: IRect,
}

/// Liquify: push pixels around with a brush (forward warp, bloat, pucker, twirl, smooth,
/// reconstruct, freeze/thaw). A session starts with the first stroke on a raster layer and
/// keeps going across strokes; Enter bakes it as one "Liquify" edit, Escape discards it,
/// Undo/Redo step through its strokes (see [`crate::liquify`]).
#[derive(Default)]
pub struct LiquifyTool {
    session: Option<Session>,
    drag: Option<Drag>,
}

fn snap_bytes(v: &[Snap]) -> usize {
    v.iter().map(|s| s.data.len() * 16).sum()
}

impl LiquifyTool {
    /// A session is open (strokes not baked yet).
    pub fn active(&self) -> bool {
        self.session.is_some()
    }

    pub fn can_undo(&self) -> (bool, bool) {
        self.session.as_ref().map(|s| (!s.undo.is_empty(), !s.redo.is_empty())).unwrap_or_default()
    }

    fn begin(&mut self, cx: &mut ToolCx) -> Result<(), String> {
        let l = cx.doc.active().ok_or("No active layer")?;
        if l.lock.pixels {
            return Err(format!("“{}” has locked pixels", l.name));
        }
        let LayerKind::Raster(base) = &l.kind else {
            return Err(format!("Liquify works on raster layer pixels — “{}” is a {} layer", l.name, l.kind.type_name().to_lowercase()));
        };
        let rect = working_rect(cx.doc.bounds(), base.content_bounds());
        let field = LiquifyField::new(rect);
        let spec = LiquifySpec {
            target: PaintTarget { layer: l.id, mask: false, saved: None },
            base: base.clone(),
            selection: cx.doc.selection.clone(),
            rect,
            step: field.step,
            nodes: [field.w, field.h],
        };
        let gpu = match cx.strokes.as_deref_mut() {
            Some(ops) => {
                ops.push(StrokeOp::LiquifyBegin(spec.clone()));
                true
            }
            None => false,
        };
        let pre = field.data.clone();
        self.session = Some(Session { spec, field, pre, undo: vec![], redo: vec![], gpu });
        Ok(())
    }

    /// Show field changes: hand the nodes to the GPU host, or re-warp the working copy.
    fn refresh(&self, cx: &mut ToolCx, nodes: IRect) {
        let Some(s) = &self.session else { return };
        if nodes.is_empty() {
            return;
        }
        if s.gpu {
            if let Some(ops) = cx.strokes.as_deref_mut() {
                ops.push(StrokeOp::LiquifyField { nodes, data: s.field.read(nodes) });
            }
            return;
        }
        let r = s.field.doc_rect(nodes);
        let px = warp_rect(&s.spec.base, &s.field, s.spec.selection.as_deref(), r, false);
        if let Some(g) = target_grid(cx.doc, s.spec.target) {
            g.write_rect(&r, &px);
        }
    }

    fn apply(&mut self, cx: &mut ToolCx, dabs: Vec<Dab>) {
        let (Some(s), Some(d)) = (self.session.as_mut(), self.drag.as_mut()) else { return };
        let o = &cx.opts.liquify;
        let mut changed = IRect::EMPTY;
        for dab in dabs {
            let motion = d.last.map(|l| [dab.center[0] - l[0], dab.center[1] - l[1]]).unwrap_or([0.0; 2]);
            d.last = Some(dab.center);
            // `dab.alpha` = pressure (× pen pressure); stationary modes also scale by the rate.
            let p = dab.alpha;
            let amount = match o.mode {
                LiquifyMode::Forward | LiquifyMode::Freeze | LiquifyMode::Thaw => p,
                LiquifyMode::Bloat | LiquifyMode::Pucker => p * o.rate * 0.08,
                LiquifyMode::TwirlCw | LiquifyMode::TwirlCcw => p * o.rate * 0.1,
                LiquifyMode::Reconstruct | LiquifyMode::Smooth => p * o.rate * 0.25,
            };
            if o.mode == LiquifyMode::Forward && motion == [0.0; 2] {
                continue;
            }
            let dab = LiquifyDab { mode: o.mode, center: dab.center, radius: dab.radius, amount, density: o.density, motion };
            changed = changed.union(&s.field.apply_dab(&dab));
        }
        d.nodes = d.nodes.union(&changed);
        self.refresh(cx, changed);
    }

    /// Stroke ended: record its undo step.
    fn end_drag(&mut self) {
        let (Some(s), Some(d)) = (self.session.as_mut(), self.drag.take()) else { return };
        if d.nodes.is_empty() {
            return;
        }
        let n = d.nodes;
        let w = s.field.w;
        let mut old = Vec::with_capacity(n.area() as usize);
        for j in n.y0..n.y1 {
            let row = (j * w + n.x0) as usize..(j * w + n.x1) as usize;
            old.extend_from_slice(&s.pre[row.clone()]);
            s.pre[row.clone()].copy_from_slice(&s.field.data[row]);
        }
        s.undo.push(Snap { nodes: n, data: old });
        s.redo.clear();
        while s.undo.len() > 1 && snap_bytes(&s.undo) > UNDO_BYTES {
            s.undo.remove(0);
        }
    }

    fn step_history(&mut self, cx: &mut ToolCx, redo: bool) -> ToolOutcome {
        if self.drag.is_some() {
            return ToolOutcome::None;
        }
        let Some(s) = self.session.as_mut() else { return ToolOutcome::None };
        let Some(snap) = (if redo { s.redo.pop() } else { s.undo.pop() }) else { return ToolOutcome::None };
        let cur = s.field.read(snap.nodes);
        s.field.write(snap.nodes, &snap.data);
        let w = s.field.w as usize;
        let n = snap.nodes;
        for (k, j) in (n.y0..n.y1).enumerate() {
            let at = j as usize * w + n.x0 as usize;
            s.pre[at..at + n.width() as usize].copy_from_slice(&snap.data[k * n.width() as usize..][..n.width() as usize]);
        }
        let back = Snap { nodes: n, data: cur };
        if redo { s.undo.push(back) } else { s.redo.push(back) }
        self.refresh(cx, n);
        ToolOutcome::Preview
    }

    /// Bake the session (Enter / Apply).
    fn commit(&mut self, cx: &mut ToolCx) -> ToolOutcome {
        self.drag = None;
        let Some(s) = self.session.take() else { return ToolOutcome::None };
        let changed = !s.field.active_rect().is_empty();
        if s.gpu {
            if let Some(ops) = cx.strokes.as_deref_mut() {
                ops.push(if changed { StrokeOp::LiquifyEnd { label: "Liquify".into(), field: Arc::new(s.field) } } else { StrokeOp::Abort });
            }
            // The host lands the GPU render as the history entry; the working copy is untouched.
            return if changed { ToolOutcome::Preview } else { ToolOutcome::Cancel };
        }
        if !changed {
            return ToolOutcome::Cancel;
        }
        apply_cpu(cx.doc, &s.spec, &s.field);
        ToolOutcome::Commit("Liquify".into())
    }

    fn cancel(&mut self, cx: &mut ToolCx) -> ToolOutcome {
        self.drag = None;
        let Some(s) = self.session.take() else { return ToolOutcome::None };
        if s.gpu
            && let Some(ops) = cx.strokes.as_deref_mut() {
                ops.push(StrokeOp::Abort);
            }
        ToolOutcome::Cancel
    }
}

fn brush(cx: &ToolCx) -> BrushSettings {
    let o = &cx.opts.liquify;
    BrushSettings {
        size: o.size.max(1.0),
        hardness: o.density,
        spacing: if o.mode == LiquifyMode::Forward { 0.06 } else { 0.12 },
        flow: o.pressure.clamp(0.0, 1.0),
        size_dynamic: Dynamic::default(),
        flow_dynamic: Dynamic::pressure(0.0),
        airbrush: o.mode.stationary(),
        ..Default::default()
    }
}

fn sample_of(ev: &ToolInput) -> InputSample {
    InputSample { pos: ev.pos, pressure: ev.pressure, tilt: ev.tilt, tilt_dir: ev.tilt_dir, time: ev.time }
}

impl Tool for LiquifyTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Liquify
    }

    fn input(&mut self, cx: &mut ToolCx, ev: &ToolInput) -> ToolOutcome {
        match ev.phase {
            Phase::Down => {
                if self.session.is_none()
                    && let Err(e) = self.begin(cx) {
                        return ToolOutcome::Message(e);
                    }
                let mut stroke = Stroke::new(brush(cx), 5);
                let dabs = stroke.add(sample_of(ev));
                self.drag = Some(Drag { stroke, last: None, nodes: IRect::EMPTY });
                self.apply(cx, dabs);
                ToolOutcome::Preview
            }
            Phase::Move => {
                let Some(d) = self.drag.as_mut() else { return ToolOutcome::None };
                let dabs = d.stroke.add(sample_of(ev));
                if dabs.is_empty() {
                    return ToolOutcome::None;
                }
                self.apply(cx, dabs);
                ToolOutcome::Preview
            }
            Phase::Up => {
                let Some(d) = self.drag.as_mut() else { return ToolOutcome::None };
                let dabs = d.stroke.add(sample_of(ev));
                self.apply(cx, dabs);
                self.end_drag();
                ToolOutcome::Preview
            }
            Phase::Hover => ToolOutcome::None,
        }
    }

    fn tick(&mut self, cx: &mut ToolCx, time: f64) -> ToolOutcome {
        let Some(d) = self.drag.as_mut() else { return ToolOutcome::None };
        let dabs = d.stroke.tick(time);
        if dabs.is_empty() {
            return ToolOutcome::None;
        }
        self.apply(cx, dabs);
        ToolOutcome::Preview
    }

    fn key(&mut self, cx: &mut ToolCx, key: &ToolKey) -> ToolOutcome {
        match key {
            ToolKey::Enter => self.commit(cx),
            ToolKey::Escape => self.cancel(cx),
            ToolKey::Undo => self.step_history(cx, false),
            ToolKey::Redo => self.step_history(cx, true),
            _ => ToolOutcome::None,
        }
    }

    fn cursor(&self) -> ToolCursor {
        ToolCursor::Brush
    }

    fn busy(&self) -> bool {
        self.session.is_some() || self.drag.is_some()
    }

    fn ticking(&self) -> bool {
        self.drag.is_some()
    }

    fn reset(&mut self) {
        self.session = None;
        self.drag = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::ToolOptions;
    use prisme_core::PixelFormat;

    /// White doc with a black 20x20 square at (90..110, 90..110) on a raster layer.
    fn doc() -> Document {
        let mut d = Document::with_background("t", 200, 200, Some([1.0; 4]));
        let id = d.active_layer.unwrap();
        let g = d.layer_mut(id).unwrap().raster_mut().unwrap();
        g.map_rect(&IRect::new(90, 90, 110, 110), |_, _, p| p.copy_from_slice(&[0.0, 0.0, 0.0, 1.0]));
        let _ = PixelFormat::F32;
        d
    }

    fn drag(t: &mut LiquifyTool, d: &mut Document, o: &ToolOptions, pts: &[[f32; 2]]) {
        for (i, p) in pts.iter().enumerate() {
            let ph = if i == 0 { Phase::Down } else if i + 1 == pts.len() { Phase::Up } else { Phase::Move };
            let mut cx = ToolCx { doc: d, opts: o, fg: [0.0; 4], bg: [1.0; 4], strokes: None };
            let mut ev = ToolInput::mouse(ph, *p);
            ev.time = i as f64 * 0.016;
            t.input(&mut cx, &ev);
        }
    }

    fn key(t: &mut LiquifyTool, d: &mut Document, o: &ToolOptions, k: ToolKey) -> ToolOutcome {
        let mut cx = ToolCx { doc: d, opts: o, fg: [0.0; 4], bg: [1.0; 4], strokes: None };
        t.key(&mut cx, &k)
    }

    fn px(d: &Document, x: i32, y: i32) -> [f32; 4] {
        d.active().unwrap().raster().unwrap().get_pixel(x, y)
    }

    fn line(a: [f32; 2], b: [f32; 2], n: usize) -> Vec<[f32; 2]> {
        (0..=n).map(|i| [a[0] + (b[0] - a[0]) * i as f32 / n as f32, a[1] + (b[1] - a[1]) * i as f32 / n as f32]).collect()
    }

    #[test]
    fn forward_warp_moves_feature_and_session_undo() {
        let mut d = doc();
        let mut o = ToolOptions::default();
        o.liquify = crate::LiquifyOptions { size: 80.0, pressure: 1.0, density: 0.8, ..Default::default() };
        let mut t = LiquifyTool::default();
        // Push the square to the right by ~20 px.
        drag(&mut t, &mut d, &o, &line([100.0, 100.0], [120.0, 100.0], 20));
        assert!(t.busy() && t.active());
        assert!(px(&d, 118, 100)[0] < 0.2, "square pushed right: {:?}", px(&d, 118, 100));
        assert!(px(&d, 92, 100)[0] > 0.5, "left edge uncovered: {:?}", px(&d, 92, 100));
        // In-session undo restores the identity, redo re-applies.
        assert_eq!(key(&mut t, &mut d, &o, ToolKey::Undo), ToolOutcome::Preview);
        assert!(px(&d, 118, 100)[0] > 0.99 && px(&d, 92, 100)[0] < 0.01);
        assert_eq!(key(&mut t, &mut d, &o, ToolKey::Undo), ToolOutcome::None);
        assert_eq!(key(&mut t, &mut d, &o, ToolKey::Redo), ToolOutcome::Preview);
        assert!(px(&d, 118, 100)[0] < 0.2);
        // Reconstruct brings it back towards identity.
        o.liquify.mode = LiquifyMode::Reconstruct;
        o.liquify.rate = 1.0;
        for _ in 0..6 {
            drag(&mut t, &mut d, &o, &line([100.0, 100.0], [101.0, 100.0], 6));
        }
        assert!(px(&d, 118, 100)[0] > 0.8, "reconstructed: {:?}", px(&d, 118, 100));
        assert_eq!(key(&mut t, &mut d, &o, ToolKey::Enter), ToolOutcome::Commit("Liquify".into()));
        assert!(!t.busy());
    }

    #[test]
    fn freeze_protects_and_selection_limits() {
        let mut o = ToolOptions::default();
        o.liquify = crate::LiquifyOptions { size: 80.0, pressure: 1.0, density: 0.8, ..Default::default() };
        let mut d = doc();
        let mut t = LiquifyTool::default();
        o.liquify.mode = LiquifyMode::Freeze;
        drag(&mut t, &mut d, &o, &[[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]]);
        o.liquify.mode = LiquifyMode::Forward;
        drag(&mut t, &mut d, &o, &line([100.0, 100.0], [120.0, 100.0], 20));
        assert!(px(&d, 105, 100)[0] < 0.01, "frozen centre stays: {:?}", px(&d, 105, 100));
        key(&mut t, &mut d, &o, ToolKey::Escape);
        // Selection: only the top half may change.
        let mut d = doc();
        d.selection = Some(Arc::new(prisme_core::mask::select_all(IRect::new(0, 0, 200, 100))));
        let mut t = LiquifyTool::default();
        drag(&mut t, &mut d, &o, &line([100.0, 100.0], [120.0, 100.0], 20));
        assert!(px(&d, 115, 95)[0] < 0.2, "selected part moves");
        assert!(px(&d, 115, 105)[0] > 0.99, "unselected part stays");
        assert!(px(&d, 95, 105)[0] < 0.01);
    }

    #[test]
    fn rejects_non_raster_layers() {
        let mut d = doc();
        let id = d.alloc_layer_id();
        d.insert_layer_above(prisme_core::Layer::new(id, "f", LayerKind::Fill(prisme_core::layer::FillLayer { content: prisme_core::layer::FillContent::Solid([1.0; 4]) })), None);
        d.active_layer = Some(id);
        let o = ToolOptions::default();
        let mut t = LiquifyTool::default();
        let mut cx = ToolCx { doc: &mut d, opts: &o, fg: [0.0; 4], bg: [1.0; 4], strokes: None };
        assert!(matches!(t.input(&mut cx, &ToolInput::mouse(Phase::Down, [10.0, 10.0])), ToolOutcome::Message(_)));
        assert!(!t.busy());
    }
}
