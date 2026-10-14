mod liquify;
mod misc;
mod painting;
mod select;
mod xform;

use crate::input::{ToolInput, ToolKey};
use crate::options::ToolOptions;
use prisme_core::{Document, IRect, LayerId};

pub use liquify::LiquifyTool;
pub use misc::{approx_text_bounds, HandTool, PickerTool, RulerTool, TextTool, ZoomTool};
pub use painting::{BrushTool, CloneTool, FillTool, GradientTool, RetouchTool};
pub use select::{LassoTool, MarqueeTool, RegionTool};
pub use xform::{CropTool, MoveTool, TransformTool};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum ToolKind {
    Move,
    Transform,
    SelectRect,
    SelectEllipse,
    Lasso,
    PolyLasso,
    ColorSelect,
    BlobSelect,
    Crop,
    Brush,
    Eraser,
    Fill,
    Gradient,
    Text,
    Picker,
    Clone,
    Heal,
    Smudge,
    Blur,
    Liquify,
    Ruler,
    Hand,
    Zoom,
}

impl ToolKind {
    pub const ALL: [ToolKind; 23] = [
        ToolKind::Move,
        ToolKind::Transform,
        ToolKind::SelectRect,
        ToolKind::SelectEllipse,
        ToolKind::Lasso,
        ToolKind::PolyLasso,
        ToolKind::ColorSelect,
        ToolKind::BlobSelect,
        ToolKind::Crop,
        ToolKind::Brush,
        ToolKind::Eraser,
        ToolKind::Fill,
        ToolKind::Gradient,
        ToolKind::Text,
        ToolKind::Picker,
        ToolKind::Clone,
        ToolKind::Heal,
        ToolKind::Smudge,
        ToolKind::Blur,
        ToolKind::Liquify,
        ToolKind::Ruler,
        ToolKind::Hand,
        ToolKind::Zoom,
    ];

    pub fn name(self) -> &'static str {
        match self {
            ToolKind::Move => "Move",
            ToolKind::Transform => "Free Transform",
            ToolKind::SelectRect => "Rectangle Select",
            ToolKind::SelectEllipse => "Ellipse Select",
            ToolKind::Lasso => "Lasso",
            ToolKind::PolyLasso => "Polygonal Lasso",
            ToolKind::ColorSelect => "Color Select",
            ToolKind::BlobSelect => "Blob Select",
            ToolKind::Crop => "Crop",
            ToolKind::Brush => "Brush",
            ToolKind::Eraser => "Eraser",
            ToolKind::Fill => "Fill",
            ToolKind::Gradient => "Gradient",
            ToolKind::Text => "Text",
            ToolKind::Picker => "Color Picker",
            ToolKind::Clone => "Clone Stamp",
            ToolKind::Heal => "Healing Brush",
            ToolKind::Smudge => "Smudge",
            ToolKind::Blur => "Blur",
            ToolKind::Liquify => "Liquify",
            ToolKind::Ruler => "Ruler",
            ToolKind::Hand => "Hand",
            ToolKind::Zoom => "Zoom",
        }
    }

    /// Default single-key shortcut (Photoshop-like).
    pub fn shortcut(self) -> Option<char> {
        Some(match self {
            ToolKind::Move => 'v',
            ToolKind::Transform => 't',
            ToolKind::SelectRect => 'm',
            ToolKind::Lasso => 'l',
            ToolKind::BlobSelect => 'w',
            ToolKind::Crop => 'c',
            ToolKind::Brush => 'b',
            ToolKind::Eraser => 'e',
            ToolKind::Fill => 'g',
            ToolKind::Gradient => 'q',
            ToolKind::Text => 'y',
            ToolKind::Picker => 'i',
            ToolKind::Clone => 's',
            ToolKind::Heal => 'j',
            ToolKind::Smudge => 'r',
            ToolKind::Ruler => 'u',
            ToolKind::Hand => 'h',
            ToolKind::Zoom => 'z',
            _ => return None,
        })
    }

    pub fn uses_brush(self) -> bool {
        matches!(self, ToolKind::Brush | ToolKind::Eraser | ToolKind::Clone | ToolKind::Heal | ToolKind::Smudge | ToolKind::Blur)
    }

    pub fn is_selection(self) -> bool {
        matches!(
            self,
            ToolKind::SelectRect | ToolKind::SelectEllipse | ToolKind::Lasso | ToolKind::PolyLasso | ToolKind::ColorSelect | ToolKind::BlobSelect
        )
    }
}

/// Context passed to tools: a mutable *working copy* of the document.
pub struct ToolCx<'a> {
    pub doc: &'a mut Document,
    pub opts: &'a ToolOptions,
    /// Linear working RGBA.
    pub fg: [f32; 4],
    pub bg: [f32; 4],
    /// GPU host: dab tools emit their strokes here instead of writing tiles (see [`crate::stroke`]).
    pub strokes: Option<&'a mut Vec<crate::StrokeOp>>,
    /// Visible document-space rect (bounds area fills on infinite canvases).
    pub view: IRect,
}

impl ToolCx<'_> {
    /// Region an area fill (bucket, gradient, magic wand) may cover: the page on a Fixed
    /// canvas. An Infinite canvas has no finite "everything" to fill (an unbounded fill can't be
    /// stored as tiles), so: the selection's bounds when `in_selection` and there is one, else
    /// the visible viewport, which is what the user is looking at and pointing into.
    pub fn fill_area(&self, in_selection: bool) -> IRect {
        if !self.doc.info.is_infinite() {
            return self.doc.bounds();
        }
        match &self.doc.selection {
            Some(s) if in_selection => s.content_bounds(),
            _ => self.view,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ToolOutcome {
    None,
    /// The working document changed (re-render, not yet in history).
    Preview,
    /// Commit the working document to history with this label.
    Commit(String),
    /// Discard working changes.
    Cancel,
    /// Ask the host to sample a colour at a document position.
    /// `raw`: sample the active layer's stored value (no nodes / colour transforms).
    SampleColor { pos: [f32; 2], raw: bool, to_bg: bool },
    /// Begin/continue editing a text layer on canvas.
    EditText(LayerId),
    /// Pan the view by a document-space delta.
    Pan([f32; 2]),
    /// Zoom around a document position.
    Zoom { at: [f32; 2], factor: f32 },
    /// Status bar message.
    Message(String),
}

/// On-canvas feedback in document coordinates.
#[derive(Clone, Debug, PartialEq)]
pub enum Overlay {
    Rect { r: [f32; 4], dashed: bool },
    Ellipse { r: [f32; 4] },
    Polyline { points: Vec<[f32; 2]>, closed: bool, dashed: bool },
    Line { a: [f32; 2], b: [f32; 2], label: Option<String> },
    Circle { c: [f32; 2], radius: f32 },
    /// Interactive handles (small squares).
    Handles(Vec<[f32; 2]>),
    Quad([[f32; 2]; 4]),
    Crosshair([f32; 2]),
    /// Shade everything outside this rect (crop).
    Shade([f32; 4]),
    Label { pos: [f32; 2], text: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolCursor {
    Default,
    Crosshair,
    Move,
    Grab,
    Text,
    /// Draw the brush footprint outline with this diameter (document px).
    Brush,
    Picker,
    ZoomIn,
}

pub trait Tool: Send + std::any::Any {
    fn kind(&self) -> ToolKind;
    fn input(&mut self, cx: &mut ToolCx, ev: &ToolInput) -> ToolOutcome;
    fn key(&mut self, _cx: &mut ToolCx, _key: &ToolKey) -> ToolOutcome {
        ToolOutcome::None
    }
    /// Periodic update (airbrush). `time` in seconds.
    fn tick(&mut self, _cx: &mut ToolCx, _time: f64) -> ToolOutcome {
        ToolOutcome::None
    }
    fn overlays(&self, _doc: &Document) -> Vec<Overlay> {
        vec![]
    }
    fn cursor(&self) -> ToolCursor {
        ToolCursor::Default
    }
    /// True while an interaction is in progress (host should not switch documents etc.).
    fn busy(&self) -> bool {
        false
    }
    /// Wants periodic [`Tool::tick`]s (pointer held down); defaults to `busy`.
    fn ticking(&self) -> bool {
        self.busy()
    }
    /// Abort any in-progress interaction.
    fn reset(&mut self) {}
}

pub fn make_tool(kind: ToolKind) -> Box<dyn Tool> {
    match kind {
        ToolKind::Move => Box::new(MoveTool::default()),
        ToolKind::Transform => Box::new(TransformTool::default()),
        ToolKind::SelectRect => Box::new(MarqueeTool::new(false)),
        ToolKind::SelectEllipse => Box::new(MarqueeTool::new(true)),
        ToolKind::Lasso => Box::new(LassoTool::new(false)),
        ToolKind::PolyLasso => Box::new(LassoTool::new(true)),
        ToolKind::ColorSelect => Box::new(RegionTool::new(false)),
        ToolKind::BlobSelect => Box::new(RegionTool::new(true)),
        ToolKind::Crop => Box::new(CropTool::default()),
        ToolKind::Brush => Box::new(BrushTool::new(false)),
        ToolKind::Eraser => Box::new(BrushTool::new(true)),
        ToolKind::Fill => Box::new(FillTool),
        ToolKind::Gradient => Box::new(GradientTool::default()),
        ToolKind::Text => Box::new(TextTool::default()),
        ToolKind::Picker => Box::new(PickerTool),
        ToolKind::Clone => Box::new(CloneTool::new(false)),
        ToolKind::Heal => Box::new(CloneTool::new(true)),
        ToolKind::Smudge => Box::new(RetouchTool::new(false)),
        ToolKind::Blur => Box::new(RetouchTool::new(true)),
        ToolKind::Liquify => Box::new(LiquifyTool::default()),
        ToolKind::Ruler => Box::new(RulerTool::default()),
        ToolKind::Hand => Box::new(HandTool::default()),
        ToolKind::Zoom => Box::new(ZoomTool),
    }
}

/// Selection op from modifiers (Shift = add, Alt = subtract, Shift+Alt = intersect).
pub(crate) fn sel_op(mods: &crate::Modifiers, default: prisme_core::SelectionOp) -> prisme_core::SelectionOp {
    use prisme_core::SelectionOp::*;
    match (mods.shift, mods.alt) {
        (true, true) => Intersect,
        (true, false) => Add,
        (false, true) => Subtract,
        _ => default,
    }
}

/// Apply a new selection shape to the document with feathering and the op.
pub(crate) fn apply_selection(doc: &mut Document, shape: prisme_core::TileGrid, op: prisme_core::SelectionOp, feather: f32) {
    let shape = if feather > 0.0 { prisme_core::mask::feather(&shape, feather) } else { shape };
    let combined = prisme_core::mask::combine(doc.selection.as_deref(), &shape, op);
    doc.selection = if combined.tiles.is_empty() { None } else { Some(std::sync::Arc::new(combined)) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{Modifiers, Phase};

    fn run(tool: &mut dyn Tool, doc: &mut Document, events: &[(Phase, [f32; 2])], mods: Modifiers) -> Vec<ToolOutcome> {
        let opts = ToolOptions::default();
        let mut out = vec![];
        for (i, (ph, p)) in events.iter().enumerate() {
            let mut cx = ToolCx { doc, opts: &opts, fg: [1.0, 0.0, 0.0, 1.0], bg: [1.0; 4], strokes: None, view: IRect::EMPTY };
            let mut ev = ToolInput::mouse(*ph, *p);
            ev.mods = mods;
            ev.time = i as f64 * 0.016;
            out.push(tool.input(&mut cx, &ev));
        }
        out
    }

    #[test]
    fn marquee_selects() {
        let mut doc = Document::with_background("t", 200, 200, Some([1.0; 4]));
        let mut t = make_tool(ToolKind::SelectRect);
        let r = run(t.as_mut(), &mut doc, &[(Phase::Down, [10.0, 10.0]), (Phase::Move, [50.0, 60.0]), (Phase::Up, [50.0, 60.0])], Modifiers::default());
        assert_eq!(r.last(), Some(&ToolOutcome::Commit("Rectangle Select".into())));
        let sel = doc.selection.as_ref().unwrap();
        assert_eq!(sel.get_pixel(30, 30)[0], 1.0);
        assert_eq!(sel.get_pixel(70, 30)[0], 0.0);
        // Shift adds
        run(t.as_mut(), &mut doc, &[(Phase::Down, [100.0, 100.0]), (Phase::Move, [150.0, 150.0]), (Phase::Up, [150.0, 150.0])], Modifiers { shift: true, ..Default::default() });
        let sel = doc.selection.as_ref().unwrap();
        assert_eq!(sel.get_pixel(30, 30)[0], 1.0);
        assert_eq!(sel.get_pixel(120, 120)[0], 1.0);
    }

    #[test]
    fn brush_paints_and_commits() {
        let mut doc = Document::with_background("t", 200, 200, Some([1.0; 4]));
        let mut t = make_tool(ToolKind::Brush);
        let r = run(t.as_mut(), &mut doc, &[(Phase::Down, [20.0, 20.0]), (Phase::Move, [120.0, 20.0]), (Phase::Up, [120.0, 20.0])], Modifiers::default());
        assert_eq!(r.last(), Some(&ToolOutcome::Commit("Brush Stroke".into())));
        let g = doc.active().unwrap().raster().unwrap();
        let p = g.get_pixel(70, 20);
        assert!(p[0] > 0.99 && p[1] < 0.01, "{p:?}");
        assert!(g.get_pixel(70, 80)[1] > 0.99);
    }

    #[test]
    fn fill_fills_region() {
        let mut doc = Document::with_background("t", 100, 100, Some([1.0; 4]));
        let mut t = make_tool(ToolKind::Fill);
        run(t.as_mut(), &mut doc, &[(Phase::Down, [50.0, 50.0])], Modifiers::default());
        let g = doc.active().unwrap().raster().unwrap();
        assert!(g.get_pixel(5, 5)[1] < 0.01);
    }

    #[test]
    fn move_translates_layer() {
        let mut doc = Document::with_background("t", 100, 100, None);
        let id = doc.active_layer.unwrap();
        doc.layer_mut(id).unwrap().raster_mut().unwrap().set_pixel(10, 10, [1.0; 4]);
        let mut t = make_tool(ToolKind::Move);
        run(t.as_mut(), &mut doc, &[(Phase::Down, [0.0, 0.0]), (Phase::Move, [5.0, 7.0]), (Phase::Up, [5.0, 7.0])], Modifiers::default());
        assert_eq!(doc.layer(id).unwrap().raster().unwrap().get_pixel(15, 17), [1.0; 4]);
    }

    #[test]
    fn crop_resizes() {
        let mut doc = Document::with_background("t", 100, 100, Some([1.0; 4]));
        let mut t = make_tool(ToolKind::Crop);
        run(t.as_mut(), &mut doc, &[(Phase::Down, [10.0, 10.0]), (Phase::Move, [60.0, 40.0]), (Phase::Up, [60.0, 40.0])], Modifiers::default());
        let opts = ToolOptions::default();
        let mut cx = ToolCx { doc: &mut doc, opts: &opts, fg: [0.0; 4], bg: [1.0; 4], strokes: None, view: IRect::EMPTY };
        assert_eq!(t.key(&mut cx, &ToolKey::Enter), ToolOutcome::Commit("Crop".into()));
        assert_eq!((doc.info.width, doc.info.height), (50, 30));
    }

    #[test]
    fn text_tool_types() {
        let mut doc = Document::with_background("t", 400, 200, Some([1.0; 4]));
        let mut t = make_tool(ToolKind::Text);
        let r = run(t.as_mut(), &mut doc, &[(Phase::Down, [50.0, 100.0])], Modifiers::default());
        let ToolOutcome::EditText(id) = r[0] else { panic!("{r:?}") };
        let opts = ToolOptions::default();
        let mut cx = ToolCx { doc: &mut doc, opts: &opts, fg: [0.0; 4], bg: [1.0; 4], strokes: None, view: IRect::EMPTY };
        t.key(&mut cx, &ToolKey::Text("Hello".into()));
        t.key(&mut cx, &ToolKey::Backspace);
        assert_eq!(t.key(&mut cx, &ToolKey::Escape), ToolOutcome::Commit("Edit Text".into()));
        match &doc.layer(id).unwrap().kind {
            prisme_core::LayerKind::Text(tl) => assert_eq!(tl.text, "Hell"),
            _ => panic!(),
        }
    }
}
