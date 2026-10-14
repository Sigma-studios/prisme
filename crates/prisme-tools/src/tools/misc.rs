use super::*;
use crate::input::Phase;
use prisme_core::layer::{Layer, LayerKind, TextFill, TextLayer};

/// Approximate text bounds when no renderer is available (hosts may override via `TextTool::bounds_fn`).
pub fn approx_text_bounds(t: &TextLayer) -> [f32; 4] {
    let lines: Vec<&str> = t.text.split('\n').collect();
    let w = lines.iter().map(|l| l.chars().count()).max().unwrap_or(1).max(1) as f32 * t.size * 0.58;
    let h = lines.len().max(1) as f32 * t.size * t.line_height;
    [t.position[0], t.position[1], t.position[0] + w, t.position[1] + h]
}

/// Text tool: click to create a text layer or edit an existing one; type to edit;
/// drag a text layer's box to move it; Escape (or clicking elsewhere) finishes.
#[derive(Default)]
pub struct TextTool {
    editing: Option<(LayerId, usize)>,
    drag: Option<([f32; 2], [f32; 2])>,
    /// Host-provided precise bounds (document space).
    pub bounds_fn: Option<Box<dyn Fn(&TextLayer) -> [f32; 4] + Send>>,
    dirty: bool,
}


impl TextTool {
    fn bounds(&self, t: &TextLayer) -> [f32; 4] {
        match &self.bounds_fn {
            Some(f) => f(t),
            None => approx_text_bounds(t),
        }
    }

    fn hit(&self, doc: &Document, p: [f32; 2]) -> Option<LayerId> {
        doc.all_layers().into_iter().rev().find_map(|l| match &l.kind {
            LayerKind::Text(t) if l.visible => {
                let b = self.bounds(t);
                (p[0] >= b[0] - 4.0 && p[0] <= b[2] + 4.0 && p[1] >= b[1] - 4.0 && p[1] <= b[3] + 4.0).then_some(l.id)
            }
            _ => None,
        })
    }

    fn text_mut<'a>(&self, doc: &'a mut Document) -> Option<(&'a mut TextLayer, usize)> {
        let (id, caret) = self.editing?;
        match &mut doc.layer_mut(id)?.kind {
            LayerKind::Text(t) => Some((t, caret)),
            _ => None,
        }
    }

    fn finish(&mut self) -> ToolOutcome {
        let was = self.editing.take();
        if was.is_some() && self.dirty {
            self.dirty = false;
            return ToolOutcome::Commit("Edit Text".into());
        }
        ToolOutcome::Preview
    }

    pub fn editing(&self) -> Option<(LayerId, usize)> {
        self.editing
    }

    pub fn set_caret(&mut self, caret: usize) {
        if let Some((_, c)) = &mut self.editing {
            *c = caret;
        }
    }
}

fn byte_index(s: &str, char_idx: usize) -> usize {
    s.char_indices().nth(char_idx).map(|(i, _)| i).unwrap_or(s.len())
}

impl Tool for TextTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Text
    }

    fn input(&mut self, cx: &mut ToolCx, ev: &ToolInput) -> ToolOutcome {
        match ev.phase {
            Phase::Down => {
                if let Some(id) = self.hit(cx.doc, ev.pos) {
                    if self.editing.map(|e| e.0) != Some(id) {
                        let r = self.finish();
                        let len = match &cx.doc.layer(id).map(|l| &l.kind) {
                            Some(LayerKind::Text(t)) => t.text.chars().count(),
                            _ => 0,
                        };
                        cx.doc.active_layer = Some(id);
                        self.editing = Some((id, len));
                        if let ToolOutcome::Commit(_) = r {
                            return r;
                        }
                    }
                    let pos = match &cx.doc.layer(id).map(|l| &l.kind) {
                        Some(LayerKind::Text(t)) => t.position,
                        _ => ev.pos,
                    };
                    self.drag = Some((ev.pos, pos));
                    return ToolOutcome::EditText(id);
                }
                if self.editing.is_some() {
                    return self.finish();
                }
                let id = cx.doc.alloc_layer_id();
                let t = TextLayer { text: String::new(), position: [ev.pos[0], ev.pos[1] - 32.0], fill: TextFill::Solid(cx.fg), ..Default::default() };
                let above = cx.doc.active_layer;
                cx.doc.insert_layer_above(Layer::new(id, "Text", LayerKind::Text(t)), above);
                cx.doc.active_layer = Some(id);
                self.editing = Some((id, 0));
                self.dirty = true;
                ToolOutcome::EditText(id)
            }
            Phase::Move => {
                let Some((start, p0)) = self.drag else { return ToolOutcome::None };
                if let Some((t, _)) = self.text_mut(cx.doc) {
                    t.position = [p0[0] + ev.pos[0] - start[0], p0[1] + ev.pos[1] - start[1]];
                    self.dirty = true;
                    return ToolOutcome::Preview;
                }
                ToolOutcome::None
            }
            Phase::Up => {
                self.drag = None;
                ToolOutcome::None
            }
            Phase::Hover => ToolOutcome::None,
        }
    }

    fn key(&mut self, cx: &mut ToolCx, key: &ToolKey) -> ToolOutcome {
        if self.editing.is_none() {
            return ToolOutcome::None;
        }
        if *key == ToolKey::Escape {
            return self.finish();
        }
        let Some((t, caret)) = self.text_mut(cx.doc) else { return ToolOutcome::None };
        let len = t.text.chars().count();
        let caret = caret.min(len);
        let new_caret = match key {
            ToolKey::Char(c) => {
                let i = byte_index(&t.text, caret);
                t.text.insert(i, *c);
                caret + 1
            }
            ToolKey::Text(s) => {
                let i = byte_index(&t.text, caret);
                t.text.insert_str(i, s);
                caret + s.chars().count()
            }
            ToolKey::Enter => {
                let i = byte_index(&t.text, caret);
                t.text.insert(i, '\n');
                caret + 1
            }
            ToolKey::Backspace if caret > 0 => {
                let i = byte_index(&t.text, caret - 1);
                t.text.remove(i);
                caret - 1
            }
            ToolKey::Delete if caret < len => {
                let i = byte_index(&t.text, caret);
                t.text.remove(i);
                caret
            }
            ToolKey::Left => caret.saturating_sub(1),
            ToolKey::Right => (caret + 1).min(len),
            ToolKey::Home => 0,
            ToolKey::End => len,
            _ => return ToolOutcome::None,
        };
        let changed = !matches!(key, ToolKey::Left | ToolKey::Right | ToolKey::Home | ToolKey::End);
        if let Some((_, c)) = &mut self.editing {
            *c = new_caret;
        }
        if changed {
            self.dirty = true;
        }
        ToolOutcome::Preview
    }

    fn overlays(&self, doc: &Document) -> Vec<Overlay> {
        let Some((id, _)) = self.editing else { return vec![] };
        match doc.layer(id).map(|l| &l.kind) {
            Some(LayerKind::Text(t)) => vec![Overlay::Rect { r: self.bounds(t), dashed: true }],
            _ => vec![],
        }
    }

    fn cursor(&self) -> ToolCursor {
        ToolCursor::Text
    }

    fn busy(&self) -> bool {
        self.drag.is_some()
    }

    fn reset(&mut self) {
        self.editing = None;
        self.drag = None;
    }
}

/// Colour picker: click samples the displayed colour; Shift samples the active layer's raw
/// stored value (no nodes, no colour transforms); Alt sets the background colour.
pub struct PickerTool;

impl Tool for PickerTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Picker
    }

    fn input(&mut self, _cx: &mut ToolCx, ev: &ToolInput) -> ToolOutcome {
        match ev.phase {
            Phase::Down | Phase::Move => ToolOutcome::SampleColor { pos: ev.pos, raw: ev.mods.shift, to_bg: ev.mods.alt },
            _ => ToolOutcome::None,
        }
    }

    fn cursor(&self) -> ToolCursor {
        ToolCursor::Picker
    }
}

/// Ruler: drag to measure distance and angle.
#[derive(Default)]
pub struct RulerTool {
    line: Option<([f32; 2], [f32; 2])>,
    dragging: bool,
}

impl RulerTool {
    pub fn measurement(&self) -> Option<([f32; 2], [f32; 2])> {
        self.line
    }

    /// Angle in radians of the measured line relative to the horizontal.
    pub fn angle(&self) -> Option<f32> {
        self.line.map(|(a, b)| (b[1] - a[1]).atan2(b[0] - a[0]))
    }
}

impl Tool for RulerTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Ruler
    }

    fn input(&mut self, _cx: &mut ToolCx, ev: &ToolInput) -> ToolOutcome {
        match ev.phase {
            Phase::Down => {
                self.line = Some((ev.pos, ev.pos));
                self.dragging = true;
                ToolOutcome::Preview
            }
            Phase::Move if self.dragging => {
                let (a, _) = self.line.unwrap();
                let mut b = ev.pos;
                if ev.mods.shift {
                    let d = [b[0] - a[0], b[1] - a[1]];
                    let ang = (d[1].atan2(d[0]) / std::f32::consts::FRAC_PI_4).round() * std::f32::consts::FRAC_PI_4;
                    let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
                    b = [a[0] + ang.cos() * len, a[1] + ang.sin() * len];
                }
                self.line = Some((a, b));
                ToolOutcome::Preview
            }
            Phase::Up => {
                self.dragging = false;
                ToolOutcome::Preview
            }
            _ => ToolOutcome::None,
        }
    }

    fn overlays(&self, _doc: &Document) -> Vec<Overlay> {
        let Some((a, b)) = self.line else { return vec![] };
        let d = [b[0] - a[0], b[1] - a[1]];
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
        let ang = -d[1].atan2(d[0]).to_degrees();
        vec![
            Overlay::Line { a, b, label: Some(format!("{len:.1} px  {ang:.1}°  (Δx {:.0}, Δy {:.0})", d[0], d[1])) },
            Overlay::Handles(vec![a, b]),
        ]
    }

    fn cursor(&self) -> ToolCursor {
        ToolCursor::Crosshair
    }

    fn reset(&mut self) {
        self.dragging = false;
    }
}

#[derive(Default)]
pub struct HandTool {
    grab: Option<[f32; 2]>,
}

impl Tool for HandTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Hand
    }

    fn input(&mut self, _cx: &mut ToolCx, ev: &ToolInput) -> ToolOutcome {
        match ev.phase {
            Phase::Down => {
                self.grab = Some(ev.pos);
                ToolOutcome::None
            }
            Phase::Move => match self.grab {
                Some(g) => ToolOutcome::Pan([ev.pos[0] - g[0], ev.pos[1] - g[1]]),
                None => ToolOutcome::None,
            },
            Phase::Up => {
                self.grab = None;
                ToolOutcome::None
            }
            _ => ToolOutcome::None,
        }
    }

    fn cursor(&self) -> ToolCursor {
        ToolCursor::Grab
    }
}

pub struct ZoomTool;

impl Tool for ZoomTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Zoom
    }

    fn input(&mut self, _cx: &mut ToolCx, ev: &ToolInput) -> ToolOutcome {
        if ev.phase == Phase::Down {
            return ToolOutcome::Zoom { at: ev.pos, factor: if ev.mods.alt { 0.5 } else { 2.0 } };
        }
        ToolOutcome::None
    }

    fn cursor(&self) -> ToolCursor {
        ToolCursor::ZoomIn
    }
}
