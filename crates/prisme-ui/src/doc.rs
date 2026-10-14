//! Per-document UI state: history, working copy during tool interactions, view.

use prisme_core::geom::Affine2;
use prisme_core::{Document, History, NodeId};
use prisme_gpu::GpuImage;

/// View transform: screen = center + R(rotation) · (doc − pan) · zoom (points).
#[derive(Clone, Copy, Debug)]
pub struct View {
    /// Screen points per document pixel.
    pub zoom: f32,
    /// Document point shown at the canvas centre.
    pub pan: [f32; 2],
    pub rotation: f32,
    pub fitted: bool,
}

impl Default for View {
    fn default() -> Self {
        Self { zoom: 1.0, pan: [0.0, 0.0], rotation: 0.0, fitted: false }
    }
}

impl View {
    pub fn doc_to_screen(&self, center: egui::Pos2, p: [f32; 2]) -> egui::Pos2 {
        let (s, c) = self.rotation.sin_cos();
        let d = [(p[0] - self.pan[0]) * self.zoom, (p[1] - self.pan[1]) * self.zoom];
        egui::pos2(center.x + d[0] * c - d[1] * s, center.y + d[0] * s + d[1] * c)
    }

    pub fn screen_to_doc(&self, center: egui::Pos2, p: egui::Pos2) -> [f32; 2] {
        let (s, c) = (-self.rotation).sin_cos();
        let d = [(p.x - center.x) / self.zoom, (p.y - center.y) / self.zoom];
        [self.pan[0] + d[0] * c - d[1] * s, self.pan[1] + d[0] * s + d[1] * c]
    }

    /// Affine mapping framebuffer pixels -> document pixels.
    pub fn fb_to_doc(&self, center: egui::Pos2, ppp: f32) -> Affine2 {
        // doc = pan + R(-rot) · (fb/ppp − center) / zoom
        Affine2::translate(self.pan[0], self.pan[1])
            .mul(&Affine2::rotate(-self.rotation))
            .mul(&Affine2::scale(1.0 / (self.zoom * ppp), 1.0 / (self.zoom * ppp)))
            .mul(&Affine2::translate(-center.x * ppp, -center.y * ppp))
    }

    /// Fit document rect `r` (the page, or content bounds on an infinite canvas) in `area`.
    pub fn fit(&mut self, r: prisme_core::IRect, area: egui::Vec2) {
        let (w, h) = (r.width().max(1) as f32, r.height().max(1) as f32);
        let z = ((area.x - 40.0) / w).min((area.y - 40.0) / h);
        self.zoom = z.clamp(0.01, 64.0);
        self.pan = [r.x0 as f32 + w * 0.5, r.y0 as f32 + h * 0.5];
        self.rotation = 0.0;
        self.fitted = true;
    }

    /// Zoom by `factor` keeping document point `anchor` fixed on screen.
    pub fn zoom_at(&mut self, factor: f32, anchor: [f32; 2]) {
        let nz = (self.zoom * factor).clamp(0.01, 256.0);
        let k = self.zoom / nz;
        self.pan = [anchor[0] + (self.pan[0] - anchor[0]) * k, anchor[1] + (self.pan[1] - anchor[1]) * k];
        self.zoom = nz;
        self.fitted = false;
    }
}

/// Cached GPU render of the canvas.
#[derive(Default)]
pub struct RenderCache {
    pub key: Option<(u64, prisme_core::IRect, u32, u8)>,
    pub image: Option<GpuImage>,
    pub mip: u32,
}

pub struct DocState {
    pub id: u64,
    pub history: History,
    /// Working copy while a tool interaction is in progress.
    pub working: Option<Document>,
    pub view: View,
    pub path: Option<std::path::PathBuf>,
    pub saved_revision: u64,
    /// Bumped on every visible change (render cache key).
    pub version: u64,
    pub render: RenderCache,
    /// Show the node graph output (true) or the raw layer stack (false).
    pub show_nodes: bool,
    /// Preview a specific node instead of the output.
    pub preview_node: Option<NodeId>,
    pub selected_node: Option<NodeId>,
    /// Low-res placeholder shown while the full image decodes.
    pub placeholder: Option<egui::TextureHandle>,
    /// Selection outline cache: (selection tile-ids hash, polylines in doc space).
    pub outline: Option<(u64, Vec<Vec<[f32; 2]>>)>,
    /// GPU brush stroke being painted or waiting to land (see `crate::stroke`).
    pub stroke: Option<crate::stroke::DocStroke>,
    /// Document-space rect visible on the canvas at the last frame (area fills on infinite
    /// canvases).
    pub view_rect: prisme_core::IRect,
}

static NEXT_DOC_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl DocState {
    pub fn new(doc: Document) -> Self {
        let history = History::new(doc);
        let rev = history.revision();
        Self {
            id: NEXT_DOC_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            history,
            working: None,
            view: View::default(),
            path: None,
            saved_revision: rev,
            version: 1,
            render: RenderCache::default(),
            show_nodes: true,
            preview_node: None,
            selected_node: None,
            placeholder: None,
            outline: None,
            stroke: None,
            view_rect: prisme_core::IRect::EMPTY,
        }
    }

    pub fn from_history(history: History) -> Self {
        let mut s = Self::new(history.current().clone());
        s.saved_revision = history.revision();
        s.history = history;
        s
    }

    pub fn doc(&self) -> &Document {
        self.working.as_ref().unwrap_or_else(|| self.history.current())
    }

    pub fn is_dirty(&self) -> bool {
        self.history.revision() != self.saved_revision || self.stroke_pending()
    }

    pub fn title(&self) -> String {
        let n = &self.doc().info.name;
        if self.is_dirty() { format!("{n} •") } else { n.clone() }
    }

    pub fn touch(&mut self) {
        self.version += 1;
    }

    /// Commit an edit made through a closure (applies to the working copy if one exists).
    pub fn edit(&mut self, label: &str, f: impl FnOnce(&mut Document)) {
        self.finish_stroke();
        match &mut self.working {
            Some(w) => f(w),
            None => {
                let mut d = self.history.current().clone();
                f(&mut d);
                self.history.push(label, d);
            }
        }
        self.touch();
    }

    /// Like `edit` but merges consecutive edits with the same key (slider drags).
    pub fn edit_merge(&mut self, label: &str, key: u64, f: impl FnOnce(&mut Document)) {
        self.finish_stroke();
        match &mut self.working {
            Some(w) => f(w),
            None => {
                let mut d = self.history.current().clone();
                f(&mut d);
                self.history.push_merge(label, d, Some(key));
            }
        }
        self.touch();
    }

    /// Change UI-ish state (active layer) without a history entry.
    pub fn set_active_layer(&mut self, id: prisme_core::LayerId) {
        match &mut self.working {
            Some(w) => w.active_layer = Some(id),
            None => {
                let mut d = self.history.current().clone();
                d.active_layer = Some(id);
                self.history.replace_current(d);
            }
        }
        self.touch();
    }

    pub fn commit_working(&mut self, label: &str) {
        self.finish_stroke();
        if let Some(w) = self.working.take() {
            self.history.push(label, w);
        }
        self.touch();
    }

    pub fn cancel_working(&mut self) {
        if self.working.take().is_some() {
            self.touch();
        }
    }

    pub fn undo(&mut self) {
        self.finish_stroke();
        self.working = None;
        self.history.undo();
        self.touch();
    }

    pub fn redo(&mut self) {
        self.finish_stroke();
        self.working = None;
        self.history.redo();
        self.touch();
    }
}
