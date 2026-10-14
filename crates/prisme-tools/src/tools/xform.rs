use super::*;
use crate::input::Phase;
use crate::transform::{merge_over, quad_bounds, split_by_selection, translate, warp, Resample};
use prisme_core::geom::{Affine2, Homography};
use prisme_core::layer::{FillContent, Layer, LayerKind};
use prisme_core::{IRect, TileGrid};
use std::sync::Arc;

fn translate_layer(l: &mut Layer, dx: i32, dy: i32) {
    if let Some(m) = &mut l.mask {
        m.grid = Arc::new(translate(&m.grid, dx, dy));
    }
    match &mut l.kind {
        LayerKind::Raster(g) => *g = Arc::new(translate(g, dx, dy)),
        LayerKind::Vector(v) => v.transform = Affine2::translate(dx as f32, dy as f32).mul(&v.transform),
        LayerKind::Text(t) => {
            t.position[0] += dx as f32;
            t.position[1] += dy as f32;
        }
        LayerKind::Fill(f) => {
            if let FillContent::Gradient(_, geo) = &mut f.content {
                for p in [&mut geo.start, &mut geo.end] {
                    p[0] += dx as f32;
                    p[1] += dy as f32;
                }
            }
        }
        LayerKind::Group(g) => {
            for c in &mut g.children {
                translate_layer(c, dx, dy);
            }
        }
    }
}

/// Topmost visible raster layer with alpha at `p`.
fn pick_layer(doc: &Document, p: [f32; 2]) -> Option<LayerId> {
    let (x, y) = (p[0].floor() as i32, p[1].floor() as i32);
    doc.all_layers().into_iter().rev().find(|l| l.visible && l.raster().is_some_and(|g| g.get_pixel(x, y)[3] > 0.05)).map(|l| l.id)
}

struct MoveSession {
    layer: LayerId,
    original: Layer,
    start: [f32; 2],
    /// Floating pixels (selection content) and the remainder of the layer.
    floating: Option<(TileGrid, TileGrid)>,
    orig_selection: Option<Arc<TileGrid>>,
    last: [i32; 2],
}

/// Move the active layer (or the selected pixels when there's a selection).
#[derive(Default)]
pub struct MoveTool {
    session: Option<MoveSession>,
}

impl MoveTool {
    fn apply(&self, cx: &mut ToolCx, d: [i32; 2]) {
        let Some(s) = &self.session else { return };
        let Some(l) = cx.doc.layer_mut(s.layer) else { return };
        match &s.floating {
            Some((floating, rest)) => {
                let moved = translate(floating, d[0], d[1]);
                l.kind = LayerKind::Raster(Arc::new(merge_over(rest, &moved)));
                cx.doc.selection = s.orig_selection.as_ref().map(|sel| Arc::new(translate(sel, d[0], d[1])));
            }
            None => {
                let mut nl = s.original.clone();
                translate_layer(&mut nl, d[0], d[1]);
                *l = nl;
            }
        }
    }
}

impl Tool for MoveTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Move
    }

    fn input(&mut self, cx: &mut ToolCx, ev: &ToolInput) -> ToolOutcome {
        match ev.phase {
            Phase::Down => {
                if (cx.opts.move_auto_select || ev.mods.primary)
                    && let Some(id) = pick_layer(cx.doc, ev.pos) {
                        cx.doc.active_layer = Some(id);
                    }
                let Some(layer) = cx.doc.active().cloned() else { return ToolOutcome::Message("No active layer".into()) };
                if layer.lock.position {
                    return ToolOutcome::Message(format!("“{}” is position-locked", layer.name));
                }
                let floating = match (&cx.doc.selection, layer.raster()) {
                    (Some(sel), Some(g)) => Some(split_by_selection(g, sel)),
                    _ => None,
                };
                self.session = Some(MoveSession {
                    layer: layer.id,
                    original: layer,
                    start: ev.pos,
                    floating,
                    orig_selection: cx.doc.selection.clone(),
                    last: [0, 0],
                });
                ToolOutcome::Preview
            }
            Phase::Move => {
                let Some(s) = &self.session else { return ToolOutcome::None };
                let mut d = [(ev.pos[0] - s.start[0]).round() as i32, (ev.pos[1] - s.start[1]).round() as i32];
                if ev.mods.shift {
                    if d[0].abs() > d[1].abs() { d[1] = 0 } else { d[0] = 0 }
                }
                if d == s.last {
                    return ToolOutcome::None;
                }
                self.apply(cx, d);
                self.session.as_mut().unwrap().last = d;
                ToolOutcome::Preview
            }
            Phase::Up => {
                let Some(s) = self.session.take() else { return ToolOutcome::None };
                if s.last == [0, 0] {
                    return ToolOutcome::Cancel;
                }
                ToolOutcome::Commit("Move".into())
            }
            Phase::Hover => ToolOutcome::None,
        }
    }

    fn key(&mut self, cx: &mut ToolCx, key: &ToolKey) -> ToolOutcome {
        let d = match key {
            ToolKey::Left => [-1, 0],
            ToolKey::Right => [1, 0],
            ToolKey::Up => [0, -1],
            ToolKey::Down => [0, 1],
            _ => return ToolOutcome::None,
        };
        let Some(id) = cx.doc.active_layer else { return ToolOutcome::None };
        if let Some(l) = cx.doc.layer_mut(id) {
            if l.lock.position {
                return ToolOutcome::None;
            }
            translate_layer(l, d[0], d[1]);
        }
        ToolOutcome::Commit("Nudge".into())
    }

    fn cursor(&self) -> ToolCursor {
        ToolCursor::Move
    }

    fn busy(&self) -> bool {
        self.session.is_some()
    }

    fn reset(&mut self) {
        self.session = None;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Handle {
    Corner(usize),
    Edge(usize),
    Rotate,
    Inside,
}

struct XformSession {
    layer: LayerId,
    original: Layer,
    source: Arc<TileGrid>,
    rest: Option<TileGrid>,
    orig_selection: Option<Arc<TileGrid>>,
    rect: [f32; 4],
    quad: [[f32; 2]; 4],
    drag: Option<(Handle, [f32; 2], [[f32; 2]; 4])>,
}

/// Free transform: scale (corners/edges, Shift = keep aspect), rotate (outside the box,
/// Shift = 15° steps), distort/perspective (Cmd/Ctrl + corner), move (inside).
/// Enter applies, Escape cancels.
#[derive(Default)]
pub struct TransformTool {
    session: Option<XformSession>,
}

fn rect_of(l: &Layer, doc: &Document) -> Option<[f32; 4]> {
    let b = match &l.kind {
        LayerKind::Raster(g) => match &doc.selection {
            Some(s) => s.content_bounds().intersect(&g.content_bounds()),
            None => g.content_bounds(),
        },
        LayerKind::Vector(v) => {
            let q = [[0.0, 0.0], [v.size[0], 0.0], [v.size[0], v.size[1]], [0.0, v.size[1]]].map(|p| v.transform.apply(p));
            quad_bounds(&q)
        }
        LayerKind::Text(t) => IRect::from_f32(t.position[0], t.position[1], t.position[0] + t.size * t.text.chars().count().max(1) as f32 * 0.6, t.position[1] + t.size * t.line_height),
        _ => return None,
    };
    (!b.is_empty()).then_some([b.x0 as f32, b.y0 as f32, b.x1 as f32, b.y1 as f32])
}

impl TransformTool {
    fn ensure(&mut self, cx: &mut ToolCx) -> bool {
        if self.session.is_some() {
            return true;
        }
        let Some(layer) = cx.doc.active().cloned() else { return false };
        if layer.lock.position {
            return false;
        }
        let Some(rect) = rect_of(&layer, cx.doc) else { return false };
        let (source, rest) = match (layer.raster(), &cx.doc.selection) {
            (Some(g), Some(sel)) => {
                let (f, r) = split_by_selection(g, sel);
                (Arc::new(f), Some(r))
            }
            (Some(g), None) => (g.clone(), None),
            _ => (Arc::new(TileGrid::default()), None),
        };
        let quad = [[rect[0], rect[1]], [rect[2], rect[1]], [rect[2], rect[3]], [rect[0], rect[3]]];
        self.session = Some(XformSession {
            layer: layer.id,
            original: layer,
            source,
            rest,
            orig_selection: cx.doc.selection.clone(),
            rect,
            quad,
            drag: None,
        });
        true
    }

    fn render(&self, cx: &mut ToolCx, mode: Resample) {
        let Some(s) = &self.session else { return };
        let h = Homography::rect_to_quad(s.rect, s.quad);
        let Some(l) = cx.doc.layer_mut(s.layer) else { return };
        match &s.original.kind {
            LayerKind::Raster(_) => {
                let b = quad_bounds(&s.quad).expand(1);
                let warped = warp(&s.source, &h, b, mode);
                let g = match &s.rest {
                    Some(r) => merge_over(r, &warped),
                    None => warped,
                };
                l.kind = LayerKind::Raster(Arc::new(g));
                if let Some(m) = &s.original.mask
                    && s.rest.is_none() {
                        let mut m2 = m.clone();
                        m2.grid = Arc::new(warp(&m.grid, &h, quad_bounds(&s.quad).expand(1), mode));
                        l.mask = Some(m2);
                    }
                if let Some(sel) = &s.orig_selection {
                    cx.doc.selection = Some(Arc::new(warp(sel, &h, quad_bounds(&s.quad).expand(1), Resample::Bilinear)));
                }
            }
            LayerKind::Vector(v) => {
                // Affine fit from three corners.
                let (r, q) = (s.rect, s.quad);
                let (w, hh) = (r[2] - r[0], r[3] - r[1]);
                let a = [(q[1][0] - q[0][0]) / w, (q[3][0] - q[0][0]) / hh];
                let c = [(q[1][1] - q[0][1]) / w, (q[3][1] - q[0][1]) / hh];
                let m = Affine2 { m: [a[0], a[1], q[0][0] - a[0] * r[0] - a[1] * r[1], c[0], c[1], q[0][1] - c[0] * r[0] - c[1] * r[1]] };
                if let LayerKind::Vector(nv) = &mut l.kind {
                    nv.transform = m.mul(&v.transform);
                }
            }
            LayerKind::Text(t) => {
                let (r, q) = (s.rect, s.quad);
                let w0 = r[2] - r[0];
                let top = [q[1][0] - q[0][0], q[1][1] - q[0][1]];
                let w1 = (top[0] * top[0] + top[1] * top[1]).sqrt();
                if let LayerKind::Text(nt) = &mut l.kind {
                    let k = w1 / w0.max(1e-3);
                    nt.size = t.size * k;
                    nt.rotation = t.rotation + top[1].atan2(top[0]);
                    nt.position = [q[0][0] + (t.position[0] - r[0]) * k, q[0][1] + (t.position[1] - r[1]) * k];
                }
            }
            _ => {}
        }
    }

    fn hit(&self, p: [f32; 2], tol: f32) -> Option<Handle> {
        let s = self.session.as_ref()?;
        let q = s.quad;
        let d = |a: [f32; 2]| ((a[0] - p[0]).powi(2) + (a[1] - p[1]).powi(2)).sqrt();
        for i in 0..4 {
            if d(q[i]) < tol {
                return Some(Handle::Corner(i));
            }
        }
        for i in 0..4 {
            let m = [(q[i][0] + q[(i + 1) % 4][0]) * 0.5, (q[i][1] + q[(i + 1) % 4][1]) * 0.5];
            if d(m) < tol {
                return Some(Handle::Edge(i));
            }
        }
        if point_in_quad(p, &q) {
            return Some(Handle::Inside);
        }
        Some(Handle::Rotate)
    }
}

fn point_in_quad(p: [f32; 2], q: &[[f32; 2]; 4]) -> bool {
    let mut inside = false;
    for i in 0..4 {
        let (a, b) = (q[i], q[(i + 1) % 4]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
            inside = !inside;
        }
    }
    inside
}

impl Tool for TransformTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Transform
    }

    fn input(&mut self, cx: &mut ToolCx, ev: &ToolInput) -> ToolOutcome {
        if !self.ensure(cx) {
            return if ev.phase == Phase::Down { ToolOutcome::Message("Nothing to transform".into()) } else { ToolOutcome::None };
        }
        match ev.phase {
            Phase::Down => {
                let h = self.hit(ev.pos, ev.tol(9.0));
                let s = self.session.as_mut().unwrap();
                s.drag = h.map(|h| (h, ev.pos, s.quad));
                ToolOutcome::Preview
            }
            Phase::Move => {
                let s = self.session.as_mut().unwrap();
                let Some((h, start, q0)) = s.drag else { return ToolOutcome::None };
                let d = [ev.pos[0] - start[0], ev.pos[1] - start[1]];
                let mut q = q0;
                match h {
                    Handle::Inside => {
                        for p in &mut q {
                            p[0] += d[0];
                            p[1] += d[1];
                        }
                    }
                    Handle::Corner(i) if ev.mods.primary => {
                        q[i] = [q0[i][0] + d[0], q0[i][1] + d[1]];
                    }
                    Handle::Corner(i) => {
                        // Scale around the opposite corner in the box's local frame.
                        let o = q0[(i + 2) % 4];
                        let c0 = q0[i];
                        let c1 = [c0[0] + d[0], c0[1] + d[1]];
                        let ux = [q0[1][0] - q0[0][0], q0[1][1] - q0[0][1]];
                        let uy = [q0[3][0] - q0[0][0], q0[3][1] - q0[0][1]];
                        let lx = (ux[0] * ux[0] + ux[1] * ux[1]).sqrt().max(1e-6);
                        let ly = (uy[0] * uy[0] + uy[1] * uy[1]).sqrt().max(1e-6);
                        let (ex, ey) = ([ux[0] / lx, ux[1] / lx], [uy[0] / ly, uy[1] / ly]);
                        let proj = |p: [f32; 2]| [(p[0] - o[0]) * ex[0] + (p[1] - o[1]) * ex[1], (p[0] - o[0]) * ey[0] + (p[1] - o[1]) * ey[1]];
                        let a = proj(c0);
                        let b = proj(c1);
                        let mut sx = if a[0].abs() > 1e-6 { b[0] / a[0] } else { 1.0 };
                        let mut sy = if a[1].abs() > 1e-6 { b[1] / a[1] } else { 1.0 };
                        if ev.mods.shift {
                            let s = sx.abs().max(sy.abs());
                            sx = s * sx.signum();
                            sy = s * sy.signum();
                        }
                        for (k, p) in q.iter_mut().enumerate() {
                            let l = proj(q0[k]);
                            let (nx, ny) = (l[0] * sx, l[1] * sy);
                            *p = [o[0] + ex[0] * nx + ey[0] * ny, o[1] + ex[1] * nx + ey[1] * ny];
                        }
                    }
                    Handle::Edge(i) => {
                        // Move the edge along its normal.
                        let a = q0[i];
                        let b = q0[(i + 1) % 4];
                        let e = [b[0] - a[0], b[1] - a[1]];
                        let len = (e[0] * e[0] + e[1] * e[1]).sqrt().max(1e-6);
                        let n = [-e[1] / len, e[0] / len];
                        let t = d[0] * n[0] + d[1] * n[1];
                        for k in [i, (i + 1) % 4] {
                            q[k] = [q0[k][0] + n[0] * t, q0[k][1] + n[1] * t];
                        }
                    }
                    Handle::Rotate => {
                        let c = [(q0[0][0] + q0[2][0]) * 0.5, (q0[0][1] + q0[2][1]) * 0.5];
                        let a0 = (start[1] - c[1]).atan2(start[0] - c[0]);
                        let a1 = (ev.pos[1] - c[1]).atan2(ev.pos[0] - c[0]);
                        let mut a = a1 - a0;
                        if ev.mods.shift {
                            let step = 15f32.to_radians();
                            a = (a / step).round() * step;
                        }
                        let (sn, cs) = a.sin_cos();
                        for (k, p) in q.iter_mut().enumerate() {
                            let v = [q0[k][0] - c[0], q0[k][1] - c[1]];
                            *p = [c[0] + v[0] * cs - v[1] * sn, c[1] + v[0] * sn + v[1] * cs];
                        }
                    }
                }
                s.quad = q;
                self.render(cx, Resample::Nearest);
                ToolOutcome::Preview
            }
            Phase::Up => {
                if let Some(s) = self.session.as_mut() {
                    s.drag = None;
                }
                self.render(cx, Resample::Bilinear);
                ToolOutcome::Preview
            }
            Phase::Hover => ToolOutcome::Preview,
        }
    }

    fn key(&mut self, cx: &mut ToolCx, key: &ToolKey) -> ToolOutcome {
        match key {
            ToolKey::Enter if self.session.is_some() => {
                self.render(cx, Resample::Bicubic);
                self.session = None;
                ToolOutcome::Commit("Free Transform".into())
            }
            ToolKey::Escape if self.session.is_some() => {
                self.session = None;
                ToolOutcome::Cancel
            }
            _ => ToolOutcome::None,
        }
    }

    fn overlays(&self, _doc: &Document) -> Vec<Overlay> {
        let Some(s) = &self.session else { return vec![] };
        let q = s.quad;
        let mut handles = q.to_vec();
        for i in 0..4 {
            handles.push([(q[i][0] + q[(i + 1) % 4][0]) * 0.5, (q[i][1] + q[(i + 1) % 4][1]) * 0.5]);
        }
        vec![Overlay::Quad(q), Overlay::Handles(handles)]
    }

    fn cursor(&self) -> ToolCursor {
        ToolCursor::Move
    }

    fn busy(&self) -> bool {
        self.session.is_some()
    }

    fn reset(&mut self) {
        self.session = None;
    }
}

/// Crop: drag a rectangle, adjust with handles, Enter (or double-click) applies.
#[derive(Default)]
pub struct CropTool {
    rect: Option<[f32; 4]>,
    drag: Option<(Handle, [f32; 2], [f32; 4])>,
    last_click: f64,
}

impl CropTool {
    fn handle(&self, p: [f32; 2], tol: f32) -> Option<Handle> {
        let r = self.rect?;
        let corners = [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]];
        for (i, c) in corners.iter().enumerate() {
            if (c[0] - p[0]).abs() < tol && (c[1] - p[1]).abs() < tol {
                return Some(Handle::Corner(i));
            }
        }
        let inside = p[0] >= r[0] && p[0] <= r[2] && p[1] >= r[1] && p[1] <= r[3];
        if inside {
            if (p[1] - r[1]).abs() < tol {
                return Some(Handle::Edge(0));
            }
            if (p[0] - r[2]).abs() < tol {
                return Some(Handle::Edge(1));
            }
            if (p[1] - r[3]).abs() < tol {
                return Some(Handle::Edge(2));
            }
            if (p[0] - r[0]).abs() < tol {
                return Some(Handle::Edge(3));
            }
            return Some(Handle::Inside);
        }
        None
    }

    fn apply(&mut self, cx: &mut ToolCx) -> ToolOutcome {
        let Some(r) = self.rect.take() else { return ToolOutcome::None };
        let rect = IRect::new(r[0].round() as i32, r[1].round() as i32, r[2].round() as i32, r[3].round() as i32);
        if rect.width() < 1 || rect.height() < 1 {
            return ToolOutcome::Cancel;
        }
        cx.doc.crop(rect, cx.opts.crop_delete_outside);
        ToolOutcome::Commit("Crop".into())
    }
}

impl Tool for CropTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Crop
    }

    fn input(&mut self, cx: &mut ToolCx, ev: &ToolInput) -> ToolOutcome {
        match ev.phase {
            Phase::Down => {
                if ev.time - self.last_click < 0.35 && self.handle(ev.pos, 0.0) == Some(Handle::Inside) {
                    return self.apply(cx);
                }
                self.last_click = ev.time;
                match self.handle(ev.pos, ev.tol(8.0)) {
                    Some(h) => self.drag = Some((h, ev.pos, self.rect.unwrap())),
                    None => {
                        self.rect = Some([ev.pos[0], ev.pos[1], ev.pos[0], ev.pos[1]]);
                        self.drag = Some((Handle::Corner(2), ev.pos, self.rect.unwrap()));
                    }
                }
                ToolOutcome::Preview
            }
            Phase::Move => {
                let Some((h, start, r0)) = self.drag else { return ToolOutcome::None };
                let d = [ev.pos[0] - start[0], ev.pos[1] - start[1]];
                let mut r = r0;
                match h {
                    Handle::Inside => {
                        r = [r0[0] + d[0], r0[1] + d[1], r0[2] + d[0], r0[3] + d[1]];
                    }
                    Handle::Corner(i) => {
                        let (xi, yi) = match i {
                            0 => (0, 1),
                            1 => (2, 1),
                            2 => (2, 3),
                            _ => (0, 3),
                        };
                        r[xi] = r0[xi] + d[0];
                        r[yi] = r0[yi] + d[1];
                        if ev.mods.shift {
                            let w = (r[2] - r[0]).abs();
                            let s = w.max((r[3] - r[1]).abs());
                            r[yi] = if yi == 1 { r[3] - s * (r0[3] - r0[1]).signum().max(1.0) } else { r[1] + s };
                            if xi == 0 {
                                r[0] = r[2] - s;
                            } else {
                                r[2] = r[0] + s;
                            }
                        }
                    }
                    Handle::Edge(i) => match i {
                        0 => r[1] = r0[1] + d[1],
                        1 => r[2] = r0[2] + d[0],
                        2 => r[3] = r0[3] + d[1],
                        _ => r[0] = r0[0] + d[0],
                    },
                    Handle::Rotate => {}
                }
                self.rect = Some([r[0].min(r[2]), r[1].min(r[3]), r[0].max(r[2]), r[1].max(r[3])]);
                ToolOutcome::Preview
            }
            Phase::Up => {
                self.drag = None;
                ToolOutcome::Preview
            }
            Phase::Hover => ToolOutcome::None,
        }
    }

    fn key(&mut self, cx: &mut ToolCx, key: &ToolKey) -> ToolOutcome {
        match key {
            ToolKey::Enter => self.apply(cx),
            ToolKey::Escape => {
                self.rect = None;
                ToolOutcome::Preview
            }
            _ => ToolOutcome::None,
        }
    }

    fn overlays(&self, _doc: &Document) -> Vec<Overlay> {
        let Some(r) = self.rect else { return vec![] };
        let thirds = |a: f32, b: f32, k: f32| a + (b - a) * k;
        let mut v = vec![Overlay::Shade(r), Overlay::Rect { r, dashed: false }];
        for k in [1.0 / 3.0, 2.0 / 3.0] {
            v.push(Overlay::Line { a: [thirds(r[0], r[2], k), r[1]], b: [thirds(r[0], r[2], k), r[3]], label: None });
            v.push(Overlay::Line { a: [r[0], thirds(r[1], r[3], k)], b: [r[2], thirds(r[1], r[3], k)], label: None });
        }
        v.push(Overlay::Handles(vec![[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]]));
        v.push(Overlay::Label { pos: [r[2], r[3]], text: format!("{:.0} × {:.0}", r[2] - r[0], r[3] - r[1]) });
        v
    }

    fn cursor(&self) -> ToolCursor {
        ToolCursor::Crosshair
    }

    fn busy(&self) -> bool {
        self.drag.is_some()
    }

    fn reset(&mut self) {
        self.rect = None;
        self.drag = None;
    }
}
