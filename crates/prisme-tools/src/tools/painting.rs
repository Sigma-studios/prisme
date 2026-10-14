use super::*;
use crate::input::{Phase, PointerKind};
use crate::paint::{draw_gradient, fill_region, RetouchCanvas, RetouchMode};
use crate::region::{flood, Sample};
use prisme_brush::{BlendTarget, BrushSettings, InputSample, Stroke, StrokeCanvas};
use prisme_core::gradient::GradientGeometry;
use prisme_core::layer::{FillContent, FillLayer, Layer, LayerKind, LayerMask};
use prisme_core::TileGrid;
use std::sync::Arc;

/// Where painting goes: the active layer's pixels or its mask.
#[derive(Clone, Copy)]
struct PaintTarget {
    layer: LayerId,
    mask: bool,
    /// Painting on a saved (document-level) mask instead of a layer.
    saved: Option<prisme_core::MaskId>,
}

fn resolve_target(cx: &mut ToolCx) -> Result<PaintTarget, String> {
    if let Some(mid) = cx.opts.edit_saved_mask {
        if cx.doc.saved_mask(mid).is_some() {
            return Ok(PaintTarget { layer: LayerId(0), mask: true, saved: Some(mid) });
        }
    }
    let edit_mask = cx.opts.edit_mask;
    let layer = cx.doc.active().ok_or("No active layer")?;
    if layer.lock.pixels {
        return Err(format!("Layer “{}” is locked", layer.name));
    }
    if edit_mask {
        if layer.mask.is_some() {
            return Ok(PaintTarget { layer: layer.id, mask: true, saved: None });
        }
        return Err("The active layer has no mask".into());
    }
    match layer.kind {
        LayerKind::Raster(_) => Ok(PaintTarget { layer: layer.id, mask: false, saved: None }),
        _ => Err(format!("“{}” is a {} layer — rasterize it to paint", layer.name, layer.kind.type_name().to_lowercase())),
    }
}

fn target_grid(doc: &mut Document, t: PaintTarget) -> Option<&mut TileGrid> {
    if let Some(mid) = t.saved {
        return doc.masks.iter_mut().find(|m| m.id == mid).map(|m| Arc::make_mut(&mut m.mask));
    }
    let l = doc.layer_mut(t.layer)?;
    if t.mask {
        l.mask.as_mut().map(|m| Arc::make_mut(&mut m.grid))
    } else {
        l.raster_mut()
    }
}

fn target_arc(doc: &Document, t: PaintTarget) -> Option<(Arc<TileGrid>, f32, bool)> {
    if let Some(mid) = t.saved {
        return doc.saved_mask(mid).map(|m| (m.mask.clone(), 0.0, false));
    }
    let l = doc.layer(t.layer)?;
    if t.mask {
        l.mask.as_ref().map(|m| (m.grid.clone(), if m.default_visible { 1.0 } else { 0.0 }, false))
    } else {
        l.raster().map(|g| (g.clone(), 0.0, l.lock.alpha))
    }
}

fn sample_of(ev: &ToolInput) -> InputSample {
    InputSample { pos: ev.pos, pressure: ev.pressure, tilt: ev.tilt, tilt_dir: ev.tilt_dir, time: ev.time }
}

/// Paint brush and eraser.
pub struct BrushTool {
    erase: bool,
    stroke: Option<(Stroke, StrokeCanvas, PaintTarget)>,
    last_end: Option<[f32; 2]>,
    hover: Option<[f32; 2]>,
    size: f32,
}

impl BrushTool {
    pub fn new(erase: bool) -> Self {
        Self { erase, stroke: None, last_end: None, hover: None, size: 20.0 }
    }

    fn settings<'a>(&self, cx: &'a ToolCx) -> &'a BrushSettings {
        if self.erase { &cx.opts.eraser } else { &cx.opts.brush }
    }
}

impl Tool for BrushTool {
    fn kind(&self) -> ToolKind {
        if self.erase { ToolKind::Eraser } else { ToolKind::Brush }
    }

    fn input(&mut self, cx: &mut ToolCx, ev: &ToolInput) -> ToolOutcome {
        self.hover = Some(ev.pos);
        self.size = self.settings(cx).size;
        match ev.phase {
            Phase::Down => {
                let target = match resolve_target(cx) {
                    Ok(t) => t,
                    Err(e) => return ToolOutcome::Message(e),
                };
                let Some((base, default, lock_alpha)) = target_arc(cx.doc, target) else { return ToolOutcome::None };
                let erase = self.erase || ev.pointer == PointerKind::Eraser;
                let settings = self.settings(cx).clone();
                let blend = if target.mask {
                    if erase {
                        BlendTarget::Value(0.0)
                    } else {
                        BlendTarget::Value(prisme_color::linear_to_srgb(prisme_color::working_luminance([cx.fg[0], cx.fg[1], cx.fg[2]])).clamp(0.0, 1.0))
                    }
                } else if erase {
                    BlendTarget::Erase
                } else {
                    BlendTarget::Color(cx.fg)
                };
                let canvas = StrokeCanvas::new(base, blend, settings.opacity, cx.doc.selection.clone(), lock_alpha, default);
                let mut stroke = Stroke::new(settings, (ev.time * 1000.0) as u64 + 1);
                let mut dabs = vec![];
                // Shift-click: straight line from the end of the previous stroke.
                if let (true, Some(prev)) = (ev.mods.shift, self.last_end) {
                    dabs.extend(stroke.add(InputSample { pos: prev, ..sample_of(ev) }));
                }
                dabs.extend(stroke.add(sample_of(ev)));
                self.stroke = Some((stroke, canvas, target));
                let (_, canvas, target) = self.stroke.as_mut().unwrap();
                if let Some(g) = target_grid(cx.doc, *target) {
                    canvas.apply(&dabs, g);
                }
                ToolOutcome::Preview
            }
            Phase::Move => {
                let Some((stroke, canvas, target)) = self.stroke.as_mut() else { return ToolOutcome::None };
                let dabs = stroke.add(sample_of(ev));
                if dabs.is_empty() {
                    return ToolOutcome::None;
                }
                if let Some(g) = target_grid(cx.doc, *target) {
                    canvas.apply(&dabs, g);
                }
                ToolOutcome::Preview
            }
            Phase::Up => {
                let Some((mut stroke, mut canvas, target)) = self.stroke.take() else { return ToolOutcome::None };
                let dabs = stroke.add(sample_of(ev));
                if let Some(g) = target_grid(cx.doc, target) {
                    canvas.apply(&dabs, g);
                    canvas.finish(g);
                }
                self.last_end = Some(ev.pos);
                ToolOutcome::Commit(if self.erase { "Erase" } else { "Brush Stroke" }.into())
            }
            Phase::Hover => ToolOutcome::None,
        }
    }

    fn tick(&mut self, cx: &mut ToolCx, time: f64) -> ToolOutcome {
        let Some((stroke, canvas, target)) = self.stroke.as_mut() else { return ToolOutcome::None };
        let dabs = stroke.tick(time);
        if dabs.is_empty() {
            return ToolOutcome::None;
        }
        if let Some(g) = target_grid(cx.doc, *target) {
            canvas.apply(&dabs, g);
        }
        ToolOutcome::Preview
    }

    fn cursor(&self) -> ToolCursor {
        ToolCursor::Brush
    }

    fn busy(&self) -> bool {
        self.stroke.is_some()
    }

    fn reset(&mut self) {
        self.stroke = None;
    }
}

/// Clone stamp and healing brush. Alt-click sets the source.
pub struct CloneTool {
    heal: bool,
    source: Option<[f32; 2]>,
    offset: Option<[f32; 2]>,
    stroke: Option<(Stroke, RetouchCanvas, PaintTarget)>,
    hover: [f32; 2],
}

impl CloneTool {
    pub fn new(heal: bool) -> Self {
        Self { heal, source: None, offset: None, stroke: None, hover: [0.0; 2] }
    }
}

impl Tool for CloneTool {
    fn kind(&self) -> ToolKind {
        if self.heal { ToolKind::Heal } else { ToolKind::Clone }
    }

    fn input(&mut self, cx: &mut ToolCx, ev: &ToolInput) -> ToolOutcome {
        self.hover = ev.pos;
        match ev.phase {
            Phase::Down if ev.mods.alt => {
                self.source = Some(ev.pos);
                self.offset = None;
                ToolOutcome::Message("Clone source set".into())
            }
            Phase::Down => {
                let Some(src) = self.source else {
                    return ToolOutcome::Message(format!("{}-click to set the source first", if cfg!(target_os = "macos") { "⌥" } else { "Alt" }));
                };
                let target = match resolve_target(cx) {
                    Ok(t) if !t.mask => t,
                    Ok(_) => return ToolOutcome::Message("Clone works on layer pixels".into()),
                    Err(e) => return ToolOutcome::Message(e),
                };
                let offset = match (cx.opts.clone_aligned, self.offset) {
                    (true, Some(o)) => o,
                    _ => [src[0] - ev.pos[0], src[1] - ev.pos[1]],
                };
                self.offset = Some(offset);
                let Some((base, _, _)) = target_arc(cx.doc, target) else { return ToolOutcome::None };
                let mode = if self.heal { RetouchMode::Heal { offset } } else { RetouchMode::Clone { offset } };
                let canvas = RetouchCanvas::new(mode, base, cx.doc.selection.clone());
                let mut stroke = Stroke::new(cx.opts.clone_brush.clone(), 7);
                let dabs = stroke.add(sample_of(ev));
                self.stroke = Some((stroke, canvas, target));
                let (_, canvas, target) = self.stroke.as_mut().unwrap();
                if let Some(g) = target_grid(cx.doc, *target) {
                    canvas.apply(&dabs, g);
                }
                ToolOutcome::Preview
            }
            Phase::Move => {
                let Some((stroke, canvas, target)) = self.stroke.as_mut() else { return ToolOutcome::Preview };
                let dabs = stroke.add(sample_of(ev));
                if let Some(g) = target_grid(cx.doc, *target) {
                    canvas.apply(&dabs, g);
                }
                ToolOutcome::Preview
            }
            Phase::Up => {
                if self.stroke.take().is_some() {
                    if !cx.opts.clone_aligned {
                        self.offset = None;
                    }
                    return ToolOutcome::Commit(if self.heal { "Heal" } else { "Clone Stamp" }.into());
                }
                ToolOutcome::None
            }
            Phase::Hover => ToolOutcome::Preview,
        }
    }

    fn overlays(&self, _doc: &Document) -> Vec<Overlay> {
        let src = match self.offset {
            Some(o) => [self.hover[0] + o[0], self.hover[1] + o[1]],
            None => match self.source {
                Some(s) => s,
                None => return vec![],
            },
        };
        vec![Overlay::Crosshair(src)]
    }

    fn cursor(&self) -> ToolCursor {
        ToolCursor::Brush
    }

    fn busy(&self) -> bool {
        self.stroke.is_some()
    }

    fn reset(&mut self) {
        self.stroke = None;
    }
}

/// Smudge and blur brushes.
pub struct RetouchTool {
    blur: bool,
    stroke: Option<(Stroke, RetouchCanvas, PaintTarget)>,
}

impl RetouchTool {
    pub fn new(blur: bool) -> Self {
        Self { blur, stroke: None }
    }
}

impl Tool for RetouchTool {
    fn kind(&self) -> ToolKind {
        if self.blur { ToolKind::Blur } else { ToolKind::Smudge }
    }

    fn input(&mut self, cx: &mut ToolCx, ev: &ToolInput) -> ToolOutcome {
        match ev.phase {
            Phase::Down => {
                let target = match resolve_target(cx) {
                    Ok(t) if !t.mask => t,
                    Ok(_) => return ToolOutcome::Message("Smudge/blur work on layer pixels".into()),
                    Err(e) => return ToolOutcome::Message(e),
                };
                let Some((base, _, _)) = target_arc(cx.doc, target) else { return ToolOutcome::None };
                let strength = (cx.opts.retouch_strength * ev.pressure.max(0.05)).clamp(0.0, 1.0);
                let mode = if self.blur { RetouchMode::Blur { strength } } else { RetouchMode::Smudge { strength } };
                let canvas = RetouchCanvas::new(mode, base, cx.doc.selection.clone());
                let mut stroke = Stroke::new(cx.opts.retouch_brush.clone(), 3);
                let dabs = stroke.add(sample_of(ev));
                self.stroke = Some((stroke, canvas, target));
                let (_, canvas, target) = self.stroke.as_mut().unwrap();
                if let Some(g) = target_grid(cx.doc, *target) {
                    canvas.apply(&dabs, g);
                }
                ToolOutcome::Preview
            }
            Phase::Move => {
                let Some((stroke, canvas, target)) = self.stroke.as_mut() else { return ToolOutcome::None };
                let dabs = stroke.add(sample_of(ev));
                if let Some(g) = target_grid(cx.doc, *target) {
                    canvas.apply(&dabs, g);
                }
                ToolOutcome::Preview
            }
            Phase::Up => {
                if self.stroke.take().is_some() {
                    return ToolOutcome::Commit(if self.blur { "Blur" } else { "Smudge" }.into());
                }
                ToolOutcome::None
            }
            Phase::Hover => ToolOutcome::None,
        }
    }

    fn cursor(&self) -> ToolCursor {
        ToolCursor::Brush
    }

    fn busy(&self) -> bool {
        self.stroke.is_some()
    }

    fn reset(&mut self) {
        self.stroke = None;
    }
}

/// Paint bucket: fills the contiguous similar region with the foreground colour
/// (Alt: background colour).
pub struct FillTool;

impl Tool for FillTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Fill
    }

    fn input(&mut self, cx: &mut ToolCx, ev: &ToolInput) -> ToolOutcome {
        if ev.phase != Phase::Down {
            return ToolOutcome::None;
        }
        let target = match resolve_target(cx) {
            Ok(t) => t,
            Err(e) => return ToolOutcome::Message(e),
        };
        let seed = [ev.pos[0].floor() as i32, ev.pos[1].floor() as i32];
        let bounds = cx.doc.bounds();
        if !bounds.contains(seed[0], seed[1]) {
            return ToolOutcome::None;
        }
        let snapshot = cx.doc.clone();
        let Some((grid, _, _)) = target_arc(&snapshot, target) else { return ToolOutcome::None };
        let sample = if cx.opts.sample_all_layers && !target.mask { Sample::Merged(&snapshot) } else { Sample::Layer(&grid) };
        let region = flood(&sample, seed, cx.opts.tolerance, bounds, cx.opts.fill_antialias);
        let color = if ev.mods.alt { cx.bg } else { cx.fg };
        let color = if target.mask {
            let v = prisme_color::linear_to_srgb(prisme_color::working_luminance([color[0], color[1], color[2]]));
            [v; 4]
        } else {
            color
        };
        let sel = snapshot.selection.clone();
        if let Some(g) = target_grid(cx.doc, target) {
            fill_region(g, &region, color, cx.opts.fill_opacity, sel.as_deref());
        }
        ToolOutcome::Commit("Fill".into())
    }

    fn cursor(&self) -> ToolCursor {
        ToolCursor::Crosshair
    }
}

/// Gradient tool: drag to define the gradient. Previews live as a temporary fill layer.
#[derive(Default)]
pub struct GradientTool {
    drag: Option<([f32; 2], [f32; 2])>,
    temp: Option<LayerId>,
}

impl GradientTool {
    fn geometry(&self, cx: &ToolCx) -> Option<GradientGeometry> {
        let (a, b) = self.drag?;
        Some(GradientGeometry {
            kind: cx.opts.gradient_kind,
            start: a,
            end: b,
            repeat: cx.opts.gradient_repeat,
            reverse: cx.opts.gradient_reverse,
            dither: cx.opts.gradient_dither,
        })
    }

    fn gradient(&self, cx: &ToolCx) -> prisme_core::Gradient {
        if cx.opts.gradient_use_fg_bg {
            let mut g = prisme_core::Gradient::two(cx.fg, cx.bg);
            g.interp = cx.opts.gradient.interp;
            g
        } else {
            cx.opts.gradient.clone()
        }
    }
}

impl Tool for GradientTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Gradient
    }

    fn input(&mut self, cx: &mut ToolCx, ev: &ToolInput) -> ToolOutcome {
        match ev.phase {
            Phase::Down => {
                if !cx.opts.gradient_as_layer
                    && let Err(e) = resolve_target(cx) {
                        return ToolOutcome::Message(e);
                    }
                self.drag = Some((ev.pos, ev.pos));
                ToolOutcome::None
            }
            Phase::Move if self.drag.is_some() => {
                let (a, _) = self.drag.unwrap();
                let mut b = ev.pos;
                if ev.mods.shift {
                    // Snap to 45° increments.
                    let d = [b[0] - a[0], b[1] - a[1]];
                    let ang = (d[1].atan2(d[0]) / std::f32::consts::FRAC_PI_4).round() * std::f32::consts::FRAC_PI_4;
                    let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
                    b = [a[0] + ang.cos() * len, a[1] + ang.sin() * len];
                }
                self.drag = Some((a, b));
                let geo = self.geometry(cx).unwrap();
                let content = FillContent::Gradient(self.gradient(cx), geo);
                match self.temp.and_then(|id| cx.doc.layer_mut(id)) {
                    Some(l) => {
                        if let LayerKind::Fill(f) = &mut l.kind {
                            f.content = content;
                        }
                    }
                    None => {
                        let id = cx.doc.alloc_layer_id();
                        let mut l = Layer::new(id, "Gradient", LayerKind::Fill(FillLayer { content }));
                        l.opacity = cx.opts.gradient_opacity;
                        if let Some(sel) = &cx.doc.selection {
                            l.mask = Some(LayerMask { grid: sel.clone(), enabled: true, default_visible: false });
                        }
                        let above = cx.doc.active_layer;
                        cx.doc.insert_layer_above(l, above);
                        self.temp = Some(id);
                    }
                }
                ToolOutcome::Preview
            }
            Phase::Up if self.drag.is_some() => {
                let geo = self.geometry(cx);
                let temp = self.temp.take();
                self.drag = None;
                let Some(geo) = geo else { return ToolOutcome::Cancel };
                let len = ((geo.end[0] - geo.start[0]).powi(2) + (geo.end[1] - geo.start[1]).powi(2)).sqrt();
                if len < 1.0 {
                    return ToolOutcome::Cancel;
                }
                if cx.opts.gradient_as_layer {
                    if let Some(id) = temp {
                        cx.doc.active_layer = Some(id);
                    }
                    return ToolOutcome::Commit("Gradient Layer".into());
                }
                if let Some(id) = temp {
                    cx.doc.remove_layer(id);
                }
                let target = match resolve_target(cx) {
                    Ok(t) => t,
                    Err(e) => return ToolOutcome::Message(e),
                };
                let g = self.gradient(cx);
                let bounds = cx.doc.bounds();
                let sel = cx.doc.selection.clone();
                let opacity = cx.opts.gradient_opacity;
                if let Some(grid) = target_grid(cx.doc, target) {
                    draw_gradient(grid, bounds, &g, &geo, opacity, sel.as_deref());
                }
                ToolOutcome::Commit("Gradient".into())
            }
            _ => ToolOutcome::None,
        }
    }

    fn overlays(&self, _doc: &Document) -> Vec<Overlay> {
        match self.drag {
            Some((a, b)) => vec![Overlay::Line { a, b, label: None }, Overlay::Handles(vec![a, b])],
            None => vec![],
        }
    }

    fn cursor(&self) -> ToolCursor {
        ToolCursor::Crosshair
    }

    fn busy(&self) -> bool {
        self.drag.is_some()
    }

    fn reset(&mut self) {
        self.drag = None;
        self.temp = None;
    }
}
