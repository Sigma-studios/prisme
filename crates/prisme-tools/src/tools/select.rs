use super::*;
use crate::input::Phase;
use crate::region::{color_range, flood, Sample};

/// Rectangle / ellipse marquee. Shift while dragging (after start) constrains to a square,
/// Alt draws from the centre (modifiers at press time choose add/subtract/intersect).
pub struct MarqueeTool {
    ellipse: bool,
    start: Option<[f32; 2]>,
    cur: [f32; 2],
    op: prisme_core::SelectionOp,
    from_center: bool,
}

impl MarqueeTool {
    pub fn new(ellipse: bool) -> Self {
        Self { ellipse, start: None, cur: [0.0; 2], op: prisme_core::SelectionOp::Replace, from_center: false }
    }

    fn rect(&self, constrain: bool, ratio: f32) -> Option<[f32; 4]> {
        let s = self.start?;
        let mut d = [self.cur[0] - s[0], self.cur[1] - s[1]];
        if constrain || ratio > 0.0 {
            let r = if ratio > 0.0 { ratio } else { 1.0 };
            let m = d[0].abs().max(d[1].abs() * r);
            d = [m * d[0].signum(), m / r * d[1].signum()];
        }
        let r = if self.from_center {
            [s[0] - d[0].abs(), s[1] - d[1].abs(), s[0] + d[0].abs(), s[1] + d[1].abs()]
        } else {
            [s[0].min(s[0] + d[0]), s[1].min(s[1] + d[1]), s[0].max(s[0] + d[0]), s[1].max(s[1] + d[1])]
        };
        Some(r)
    }
}

impl Tool for MarqueeTool {
    fn kind(&self) -> ToolKind {
        if self.ellipse { ToolKind::SelectEllipse } else { ToolKind::SelectRect }
    }

    fn input(&mut self, cx: &mut ToolCx, ev: &ToolInput) -> ToolOutcome {
        match ev.phase {
            Phase::Down => {
                self.start = Some(ev.pos);
                self.cur = ev.pos;
                self.op = sel_op(&ev.mods, cx.opts.sel_op);
                self.from_center = false;
                ToolOutcome::None
            }
            Phase::Move if self.start.is_some() => {
                self.cur = ev.pos;
                // After the press, Alt means "from centre" (Photoshop behaviour).
                if self.op == prisme_core::SelectionOp::Replace {
                    self.from_center = ev.mods.alt;
                }
                ToolOutcome::Preview
            }
            Phase::Up if self.start.is_some() => {
                self.cur = ev.pos;
                let constrain = ev.mods.shift && self.op != prisme_core::SelectionOp::Add;
                let r = self.rect(constrain, cx.opts.sel_fixed_ratio);
                self.start = None;
                let Some(r) = r else { return ToolOutcome::None };
                if (r[2] - r[0]).abs() < 1.0 || (r[3] - r[1]).abs() < 1.0 {
                    // Click without drag: deselect (replace mode only).
                    if self.op == prisme_core::SelectionOp::Replace && cx.doc.selection.is_some() {
                        cx.doc.selection = None;
                        return ToolOutcome::Commit("Deselect".into());
                    }
                    return ToolOutcome::None;
                }
                let shape = if self.ellipse {
                    prisme_core::mask::ellipse(r, cx.opts.sel_antialias)
                } else {
                    prisme_core::mask::rect(r, cx.opts.sel_antialias)
                };
                apply_selection(cx.doc, shape, self.op, cx.opts.sel_feather);
                ToolOutcome::Commit(if self.ellipse { "Ellipse Select" } else { "Rectangle Select" }.into())
            }
            _ => ToolOutcome::None,
        }
    }

    fn overlays(&self, _doc: &Document) -> Vec<Overlay> {
        let Some(r) = self.rect(false, 0.0) else { return vec![] };
        let label = Overlay::Label { pos: [r[2], r[3]], text: format!("{:.0} × {:.0}", r[2] - r[0], r[3] - r[1]) };
        if self.ellipse {
            vec![Overlay::Ellipse { r }, label]
        } else {
            vec![Overlay::Rect { r, dashed: true }, label]
        }
    }

    fn cursor(&self) -> ToolCursor {
        ToolCursor::Crosshair
    }

    fn busy(&self) -> bool {
        self.start.is_some()
    }

    fn reset(&mut self) {
        self.start = None;
    }
}

/// Freehand lasso, or polygonal (click points, double-click/Enter or click the first point to close).
pub struct LassoTool {
    poly: bool,
    points: Vec<[f32; 2]>,
    hover: [f32; 2],
    op: prisme_core::SelectionOp,
    dragging: bool,
    last_click: f64,
}

impl LassoTool {
    pub fn new(poly: bool) -> Self {
        Self { poly, points: vec![], hover: [0.0; 2], op: prisme_core::SelectionOp::Replace, dragging: false, last_click: -1.0 }
    }

    fn finish(&mut self, cx: &mut ToolCx) -> ToolOutcome {
        let pts = std::mem::take(&mut self.points);
        self.dragging = false;
        let area = 0.5 * pts.iter().zip(pts.iter().cycle().skip(1)).map(|(a, b)| a[0] * b[1] - b[0] * a[1]).sum::<f32>();
        if pts.len() < 3 || area.abs() < 4.0 {
            // A click or a degenerate stroke: deselect in replace mode.
            if self.op == prisme_core::SelectionOp::Replace && cx.doc.selection.is_some() {
                cx.doc.selection = None;
                return ToolOutcome::Commit("Deselect".into());
            }
            return ToolOutcome::Cancel;
        }
        let shape = prisme_core::mask::polygon(&pts, cx.opts.sel_antialias);
        apply_selection(cx.doc, shape, self.op, cx.opts.sel_feather);
        ToolOutcome::Commit(if self.poly { "Polygonal Lasso" } else { "Lasso" }.into())
    }
}

impl Tool for LassoTool {
    fn kind(&self) -> ToolKind {
        if self.poly { ToolKind::PolyLasso } else { ToolKind::Lasso }
    }

    fn input(&mut self, cx: &mut ToolCx, ev: &ToolInput) -> ToolOutcome {
        self.hover = ev.pos;
        if self.poly {
            match ev.phase {
                Phase::Down => {
                    if self.points.is_empty() {
                        self.op = sel_op(&ev.mods, cx.opts.sel_op);
                    } else {
                        let first = self.points[0];
                        let close = ((ev.pos[0] - first[0]).powi(2) + (ev.pos[1] - first[1]).powi(2)).sqrt() < ev.tol(8.0);
                        let double = ev.time - self.last_click < 0.35;
                        if close || double {
                            self.last_click = -1.0;
                            return self.finish(cx);
                        }
                    }
                    self.last_click = ev.time;
                    self.points.push(ev.pos);
                    ToolOutcome::Preview
                }
                Phase::Hover | Phase::Move => {
                    if self.points.is_empty() { ToolOutcome::None } else { ToolOutcome::Preview }
                }
                _ => ToolOutcome::None,
            }
        } else {
            match ev.phase {
                Phase::Down => {
                    self.op = sel_op(&ev.mods, cx.opts.sel_op);
                    self.points = vec![ev.pos];
                    self.dragging = true;
                    ToolOutcome::Preview
                }
                Phase::Move if self.dragging => {
                    let last = *self.points.last().unwrap();
                    if (ev.pos[0] - last[0]).abs() + (ev.pos[1] - last[1]).abs() >= ev.tol(1.5) {
                        self.points.push(ev.pos);
                    }
                    ToolOutcome::Preview
                }
                Phase::Up if self.dragging => self.finish(cx),
                _ => ToolOutcome::None,
            }
        }
    }

    fn key(&mut self, cx: &mut ToolCx, key: &ToolKey) -> ToolOutcome {
        match key {
            ToolKey::Enter if !self.points.is_empty() => self.finish(cx),
            ToolKey::Escape if !self.points.is_empty() => {
                self.points.clear();
                ToolOutcome::Preview
            }
            ToolKey::Backspace | ToolKey::Delete if self.poly && !self.points.is_empty() => {
                self.points.pop();
                ToolOutcome::Preview
            }
            _ => ToolOutcome::None,
        }
    }

    fn overlays(&self, _doc: &Document) -> Vec<Overlay> {
        if self.points.is_empty() {
            return vec![];
        }
        let mut pts = self.points.clone();
        if self.poly {
            pts.push(self.hover);
        }
        let mut v = vec![Overlay::Polyline { points: pts, closed: !self.poly, dashed: true }];
        if self.poly {
            v.push(Overlay::Handles(vec![self.points[0]]));
        }
        v
    }

    fn cursor(&self) -> ToolCursor {
        ToolCursor::Crosshair
    }

    fn busy(&self) -> bool {
        !self.points.is_empty()
    }

    fn reset(&mut self) {
        self.points.clear();
        self.dragging = false;
    }
}

/// Colour range (global) or blob (contiguous) selection.
pub struct RegionTool {
    contiguous: bool,
}

impl RegionTool {
    pub fn new(contiguous: bool) -> Self {
        Self { contiguous }
    }
}

impl Tool for RegionTool {
    fn kind(&self) -> ToolKind {
        if self.contiguous { ToolKind::BlobSelect } else { ToolKind::ColorSelect }
    }

    fn input(&mut self, cx: &mut ToolCx, ev: &ToolInput) -> ToolOutcome {
        if ev.phase != Phase::Down {
            return ToolOutcome::None;
        }
        let op = sel_op(&ev.mods, cx.opts.sel_op);
        let seed = [ev.pos[0].floor() as i32, ev.pos[1].floor() as i32];
        let bounds = cx.doc.bounds();
        if !bounds.contains(seed[0], seed[1]) {
            return ToolOutcome::None;
        }
        let doc_snapshot = cx.doc.clone();
        let layer_grid = doc_snapshot.active().and_then(|l| l.raster().cloned());
        let sample = match (&layer_grid, cx.opts.sample_all_layers) {
            (Some(g), false) => Sample::Layer(g),
            _ => Sample::Merged(&doc_snapshot),
        };
        let shape = if self.contiguous {
            flood(&sample, seed, cx.opts.tolerance, bounds, cx.opts.sel_antialias)
        } else {
            color_range(&sample, seed, cx.opts.tolerance, bounds, cx.opts.sel_antialias)
        };
        apply_selection(cx.doc, shape, op, cx.opts.sel_feather);
        ToolOutcome::Commit(if self.contiguous { "Blob Select" } else { "Color Select" }.into())
    }

    fn cursor(&self) -> ToolCursor {
        ToolCursor::Crosshair
    }
}
