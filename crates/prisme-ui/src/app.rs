use crate::actions::Action;
use crate::dialogs::Dialogs;
use crate::doc::DocState;
use crate::jobs::Job;
use crate::rasterizer::Rasterizer;
use crate::settings::{Settings, SETTINGS_KEY};
use crate::theme;
use egui::{Color32, RichText, Ui};
use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};
use prisme_core::NodeId;
use prisme_gpu::{scopes::Scopes, DisplayRenderer, Engine, GpuContext, GpuImage, Readback};
use prisme_text::FontDb;
use prisme_tools::{make_tool, Tool, ToolKind};
use std::collections::HashMap;
use std::sync::Arc;
use egui_phosphor::regular as ph;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Tab {
    Canvas,
    Layers,
    Nodes,
    Properties,
    Color,
    History,
    Masks,
    Scopes,
    Brushes,
    Navigator,
}

impl Tab {
    pub const ALL: [Tab; 10] =
        [Tab::Canvas, Tab::Layers, Tab::Nodes, Tab::Properties, Tab::Color, Tab::History, Tab::Masks, Tab::Scopes, Tab::Brushes, Tab::Navigator];

    pub fn title(self) -> &'static str {
        match self {
            Tab::Canvas => "Canvas",
            Tab::Layers => "Layers",
            Tab::Nodes => "Node Graph",
            Tab::Properties => "Properties",
            Tab::Color => "Color",
            Tab::History => "History",
            Tab::Masks => "Masks",
            Tab::Scopes => "Scopes",
            Tab::Brushes => "Brushes",
            Tab::Navigator => "Navigator",
        }
    }
}

pub struct Gpu {
    pub engine: Engine,
    pub rs: egui_wgpu::RenderState,
    pub scopes: Scopes,
    pub scope_time: f64,
    /// Node thumbnails: node -> (doc version, texture id, image kept alive).
    pub thumbs: HashMap<(u64, NodeId), (u64, egui::TextureId, GpuImage)>,
    pub pick: Option<(Readback, bool)>,
}

/// Everything except the dock layout (so panels can borrow it mutably).
pub struct AppState {
    pub docs: Vec<DocState>,
    pub active: usize,
    pub tools: HashMap<ToolKind, Box<dyn Tool>>,
    pub tool: ToolKind,
    pub settings: Settings,
    pub gpu: Option<Gpu>,
    pub fonts: Arc<FontDb>,
    pub rasterizer: Arc<Rasterizer>,
    pub font_generation: u64,
    pub dialogs: Dialogs,
    pub jobs: Vec<Job>,
    pub status: Option<(String, f64)>,
    /// Internal clipboard: full-precision pixels and their origin.
    pub clipboard: Option<(prisme_core::TileGrid, prisme_core::IRect)>,
    pub node_editors: HashMap<u64, crate::panels::nodes::NodeEditor>,
    pub hover_doc: Option<[f32; 2]>,
    pub display_profile: Option<String>,
    /// HDR output state, refreshed every frame by [`crate::hdr::update`].
    pub hdr: crate::hdr::HdrStatus,
    pub last_autosave: f64,
    pub start_time: f64,
    pub frame_ms: f32,
    pub space_pan: bool,
}

pub struct PrismeApp {
    dock: DockState<Tab>,
    pub s: AppState,
    script: Option<crate::script::Script>,
}

/// Bumped when the default layout changes so the new default applies once.
const DOCK_KEY: &str = "prisme-dock-v2";

/// Open the node editor docked under the canvas (or focus it if already open).
pub fn show_node_editor(dock: &mut DockState<Tab>) {
    if let Some(loc) = dock.find_tab(&Tab::Nodes) {
        let _ = dock.set_active_tab(loc);
        return;
    }
    match dock.find_tab(&Tab::Canvas) {
        Some(path) if path.surface == egui_dock::SurfaceIndex::main() => {
            dock.main_surface_mut().split_below(path.node, 0.62, vec![Tab::Nodes]);
        }
        _ => dock.push_to_focused_leaf(Tab::Nodes),
    }
}

pub fn default_dock_pub() -> DockState<Tab> {
    default_dock()
}

fn default_dock() -> DockState<Tab> {
    let mut dock = DockState::new(vec![Tab::Canvas]);
    let surface = dock.main_surface_mut();
    let [canvas, right] = surface.split_right(NodeIndex::root(), 0.76, vec![Tab::Color, Tab::Brushes, Tab::Navigator]);
    let [_, layers] = surface.split_below(right, 0.38, vec![Tab::Layers, Tab::Masks, Tab::History]);
    let [_, _props] = surface.split_below(layers, 0.5, vec![Tab::Properties, Tab::Scopes]);
    let _ = canvas; // the node editor is opened on demand (Nodes ▸ Show Node Editor)
    dock
}

impl PrismeApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let settings: Settings = cc
            .storage
            .and_then(|s| s.get_string(SETTINGS_KEY))
            .and_then(|j| serde_json::from_str(&j).ok())
            .unwrap_or_default();
        let dock: DockState<Tab> = cc
            .storage
            .and_then(|s| s.get_string(DOCK_KEY))
            .and_then(|j| serde_json::from_str(&j).ok())
            .unwrap_or_else(default_dock);
        theme::apply(&cc.egui_ctx, settings.theme);
        let mut fonts = egui::FontDefinitions::default();
        egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
        cc.egui_ctx.set_fonts(fonts);
        egui_extras_images(&cc.egui_ctx);
        let ctx = cc.egui_ctx.clone();
        prisme_platform::set_wake_callback(move || ctx.request_repaint());

        let fonts = FontDb::new();
        fonts.load_system_fonts_async();
        let rasterizer = Arc::new(Rasterizer::new(fonts.clone()));

        let mut display_profile = None;
        let gpu = cc.wgpu_render_state.clone().map(|rs| {
            let ctx = GpuContext::new(rs.device.clone(), rs.queue.clone());
            let mut engine = Engine::new(ctx);
            engine.rasterizer = Some(rasterizer.clone() as Arc<dyn prisme_gpu::LayerRasterizer>);
            let mut renderer = DisplayRenderer::new(&rs.device, rs.target_format);
            if settings.use_display_profile {
                display_profile = install_display_profile(&mut renderer, &rs);
            }
            rs.renderer.write().callback_resources.insert(renderer);
            let scopes = Scopes::new(&rs.device);
            Gpu { engine, rs, scopes, scope_time: 0.0, thumbs: HashMap::new(), pick: None }
        });
        log::info!("Prisme: wgpu render state available = {}", gpu.is_some());
        if gpu.is_none() {
            log::error!("Prisme requires the wgpu renderer (WebGPU on the web).");
        }

        let tools = ToolKind::ALL.iter().map(|k| (*k, make_tool(*k))).collect();
        let now = prisme_platform::now_seconds();
        let mut s = AppState {
            docs: vec![],
            active: 0,
            tools,
            tool: ToolKind::Brush,
            settings,
            gpu,
            fonts,
            rasterizer,
            font_generation: 0,
            dialogs: Dialogs::default(),
            jobs: vec![],
            status: None,
            clipboard: None,
            node_editors: HashMap::new(),
            hover_doc: None,
            display_profile,
            hdr: Default::default(),
            last_autosave: now,
            start_time: now,
            frame_ms: 0.0,
            space_pan: false,
        };
        s.install_text_bounds();
        crate::jobs::check_recovery(&mut s);
        PrismeApp { dock, s, script: crate::script::Script::from_env() }
    }

    /// Open files passed on the command line (native) or via URL (web).
    pub fn open_bytes(&mut self, name: String, bytes: Vec<u8>, path: Option<std::path::PathBuf>) {
        crate::jobs::decode_file(&mut self.s, name, bytes, None, path);
    }
}

/// Image loaders aren't needed (we upload textures ourselves); placeholder for symmetry.
fn egui_extras_images(_ctx: &egui::Context) {}

fn install_display_profile(renderer: &mut DisplayRenderer, rs: &egui_wgpu::RenderState) -> Option<String> {
    let bytes = prisme_platform::display_icc_profile()?;
    let p = prisme_color::IccProfile::parse(&bytes)?;
    let lut = prisme_color::lut::bake_display_lut(&p, prisme_color::lut::DISPLAY_LUT_SIZE)?;
    renderer.set_lut(&rs.device, &rs.queue, Some((&lut, prisme_color::lut::DISPLAY_LUT_SIZE as u32)));
    Some(p.description())
}

impl AppState {
    /// The active document; None while a non-document tab (Keyboard Shortcuts) is focused.
    pub fn doc(&self) -> Option<&DocState> {
        if self.dialogs.shortcuts.focused {
            return None;
        }
        self.docs.get(self.active)
    }

    pub fn doc_mut(&mut self) -> Option<&mut DocState> {
        if self.dialogs.shortcuts.focused {
            return None;
        }
        self.docs.get_mut(self.active)
    }

    pub fn toast(&mut self, msg: impl Into<String>) {
        self.status = Some((msg.into(), prisme_platform::now_seconds()));
    }

    pub fn add_doc(&mut self, d: DocState) {
        crate::shortcuts::unfocus(self);
        self.docs.push(d);
        self.active = self.docs.len() - 1;
    }

    pub fn set_tool(&mut self, k: ToolKind) {
        if k == self.tool {
            return;
        }
        // Leaving a tool aborts its interaction (uncommitted previews are discarded).
        if let Some(t) = self.tools.get_mut(&self.tool)
            && (t.busy() || k != self.tool) {
                t.reset();
            }
        if let Some(d) = self.doc_mut() {
            d.cancel_working();
        }
        self.tool = k;
    }

    pub fn set_display_profile(&mut self, on: bool) {
        self.settings.use_display_profile = on;
        if let Some(g) = &self.gpu {
            let mut r = g.rs.renderer.write();
            if let Some(dr) = r.callback_resources.get_mut::<DisplayRenderer>() {
                if on {
                    self.display_profile = install_display_profile(dr, &g.rs);
                } else {
                    dr.set_lut(&g.rs.device, &g.rs.queue, None);
                    self.display_profile = None;
                }
            }
        }
    }

    /// Give the text tool precise bounds from the text renderer.
    pub fn install_text_bounds(&mut self) {
        let r = self.rasterizer.clone();
        if let Some(t) = self.tools.get_mut(&ToolKind::Text) {
            let any: &mut dyn std::any::Any = t.as_mut();
            if let Some(tt) = any.downcast_mut::<prisme_tools::tools::TextTool>() {
                tt.bounds_fn = Some(Box::new(move |l| r.text.bounds(l)));
            }
        }
    }

    pub fn text_editing(&mut self) -> Option<(prisme_core::LayerId, usize)> {
        let t = self.tools.get_mut(&ToolKind::Text)?;
        let any: &mut dyn std::any::Any = t.as_mut();
        any.downcast_mut::<prisme_tools::tools::TextTool>()?.editing()
    }
}

struct Viewer<'a> {
    s: &'a mut AppState,
}

impl TabViewer for Viewer<'_> {
    type Tab = Tab;

    fn title(&mut self, tab: &mut Tab) -> egui::WidgetText {
        tab.title().into()
    }

    fn id(&mut self, tab: &mut Tab) -> egui::Id {
        egui::Id::new(("prisme-tab", *tab))
    }

    fn ui(&mut self, ui: &mut Ui, tab: &mut Tab) {
        match tab {
            Tab::Canvas => crate::canvas::canvas_tab(self.s, ui),
            Tab::Layers => crate::panels::layers::ui(self.s, ui),
            Tab::Nodes => crate::panels::nodes::ui(self.s, ui),
            Tab::Properties => crate::panels::properties::ui(self.s, ui),
            Tab::Color => crate::panels::color::ui(self.s, ui),
            Tab::History => crate::panels::history::ui(self.s, ui),
            Tab::Masks => crate::panels::masks::ui(self.s, ui),
            Tab::Scopes => crate::panels::scopes::ui(self.s, ui),
            Tab::Brushes => crate::panels::brushes::ui(self.s, ui),
            Tab::Navigator => crate::panels::navigator::ui(self.s, ui),
        }
    }

    fn is_closeable(&self, tab: &Tab) -> bool {
        *tab != Tab::Canvas
    }

    fn scroll_bars(&self, tab: &Tab) -> [bool; 2] {
        match tab {
            Tab::Canvas | Tab::Nodes => [false, false],
            _ => [false, true],
        }
    }
}

impl eframe::App for PrismeApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let t0 = prisme_platform::now_seconds();
        if self.s.frame_ms == 0.0 {
            log::info!("Prisme: first frame");
        }
        let ctx = ui.ctx().clone();
        let s = &mut self.s;
        crate::jobs::poll(s, &ctx);
        crate::stroke::poll(s, &ctx);
        if let Some(sc) = &mut self.script {
            crate::script::step(s, sc, &ctx);
        }

        // Re-render text when new fonts arrive.
        let fg = s.fonts.generation();
        if fg != s.font_generation {
            s.font_generation = fg;
            if let Some(g) = &mut s.gpu {
                g.engine.clear_caches();
            }
            for d in &mut s.docs {
                d.touch();
            }
        }

        crate::hdr::update(s, &ctx);
        // Window close with unsaved documents: hold it and ask.
        let closing = ctx.input(|i| i.viewport().close_requested());
        if closing {
            // A stroke still landing isn't in the history yet: land it before checking dirtiness.
            s.docs.iter_mut().for_each(|d| d.finish_stroke());
        }
        if closing && !s.dialogs.allow_quit && s.docs.iter().any(|d| d.is_dirty()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            s.dialogs.confirm_quit = true;
        } else if closing {
            // Clean exit (all saved, or changes discarded): nothing to recover next launch.
            crate::jobs::clear_autosaves(s);
        }
        crate::jobs::quit_flow(s, &ctx);
        crate::commands::handle_shortcuts(s, &ctx);
        handle_dropped_files(s, &ctx);

        egui::Panel::top("menu").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| crate::commands::menu_bar(s, ui, &mut self.dock));
        });
        egui::Panel::top("options").show(ui, |ui| crate::panels::options::ui(s, ui));
        egui::Panel::bottom("status").exact_size(22.0).show(ui, |ui| status_bar(s, ui));
        egui::Panel::left("toolbar").exact_size(44.0).resizable(false).show(ui, |ui| toolbar(s, ui));

        if std::mem::take(&mut s.dialogs.shortcuts.reveal)
            && let Some(loc) = self.dock.find_tab(&Tab::Canvas) {
                let _ = self.dock.set_active_tab(loc);
            }
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
            let mut style = Style::from_egui(ui.style().as_ref());
            style.tab_bar.bg_fill = theme::BG0;
            style.main_surface_border_stroke = egui::Stroke::NONE;
            DockArea::new(&mut self.dock)
                .style(style)
                .show_leaf_collapse_buttons(false)
                .show_leaf_close_all_buttons(false)
                .show_inside(ui, &mut Viewer { s });
        });

        crate::dialogs::show(s, &ctx);
        crate::jobs::autosave(s);
        if !s.jobs.is_empty() || s.gpu.as_ref().is_some_and(|g| g.pick.is_some() || g.scopes.busy()) {
            ctx.request_repaint();
        }
        s.frame_ms = ((prisme_platform::now_seconds() - t0) * 1000.0) as f32;
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if let Ok(j) = serde_json::to_string(&self.s.settings) {
            storage.set_string(SETTINGS_KEY, j);
        }
        if let Ok(j) = serde_json::to_string(&self.dock) {
            storage.set_string(DOCK_KEY, j);
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(&mut *self)
    }
}

impl PrismeApp {
    pub fn reset_layout(&mut self) {
        self.dock = default_dock();
    }

    pub fn dock_mut(&mut self) -> &mut DockState<Tab> {
        &mut self.dock
    }
}

fn handle_dropped_files(s: &mut AppState, ctx: &egui::Context) {
    let files = ctx.input(|i| i.raw.dropped_files.clone());
    for f in files {
        let path = f.path().to_path_buf();
        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "dropped".into());
        // Dropping onto an open document places it as a layer (hold Shift to open separately).
        let place = !s.docs.is_empty() && !ctx.input(|i| i.modifiers.shift) && !prisme_io::is_project_file(&name);
        let into = if place { s.doc().map(|d| d.id) } else { None };
        let p = (!path.as_os_str().is_empty()).then_some(path);
        #[cfg(not(target_arch = "wasm32"))]
        if let Ok(bytes) = f.bytes() {
            crate::jobs::decode_file(s, name, bytes, into, p);
        }
        #[cfg(target_arch = "wasm32")]
        {
            let f2 = f.clone();
            let pending = prisme_platform::spawn_async(async move { f2.bytes_async().await });
            s.jobs.push(crate::jobs::Job::Dropped { pending, name, into });
            let _ = p;
        }
    }
}

pub fn tool_icon(k: ToolKind) -> &'static str {
    match k {
        ToolKind::Move => ph::ARROWS_OUT_CARDINAL,
        ToolKind::Transform => ph::BOUNDING_BOX,
        ToolKind::SelectRect => ph::SELECTION,
        ToolKind::SelectEllipse => ph::CIRCLE_DASHED,
        ToolKind::Lasso => ph::LASSO,
        ToolKind::PolyLasso => ph::POLYGON,
        ToolKind::ColorSelect => ph::PALETTE,
        ToolKind::BlobSelect => ph::MAGIC_WAND,
        ToolKind::Crop => ph::CROP,
        ToolKind::Brush => ph::PAINT_BRUSH,
        ToolKind::Eraser => ph::ERASER,
        ToolKind::Fill => ph::PAINT_BUCKET,
        ToolKind::Gradient => ph::GRADIENT,
        ToolKind::Text => ph::TEXT_T,
        ToolKind::Picker => ph::EYEDROPPER,
        ToolKind::Clone => ph::STAMP,
        ToolKind::Heal => ph::BANDAIDS,
        ToolKind::Smudge => ph::HAND_POINTING,
        ToolKind::Blur => ph::DROP,
        ToolKind::Ruler => ph::RULER,
        ToolKind::Hand => ph::HAND,
        ToolKind::Zoom => ph::MAGNIFYING_GLASS,
    }
}

fn toolbar(s: &mut AppState, ui: &mut Ui) {
    ui.add_space(6.0);
    ui.vertical_centered(|ui| {
        let groups: &[&[ToolKind]] = &[
            &[ToolKind::Move, ToolKind::Transform, ToolKind::Crop],
            &[ToolKind::SelectRect, ToolKind::SelectEllipse, ToolKind::Lasso, ToolKind::PolyLasso, ToolKind::ColorSelect, ToolKind::BlobSelect],
            &[ToolKind::Brush, ToolKind::Eraser, ToolKind::Fill, ToolKind::Gradient],
            &[ToolKind::Clone, ToolKind::Heal, ToolKind::Smudge, ToolKind::Blur],
            &[ToolKind::Text, ToolKind::Picker, ToolKind::Ruler],
            &[ToolKind::Hand, ToolKind::Zoom],
        ];
        for (gi, g) in groups.iter().enumerate() {
            if gi > 0 {
                ui.add_space(2.0);
                ui.separator();
            }
            for k in *g {
                let selected = s.tool == *k;
                let txt = RichText::new(tool_icon(*k)).size(16.0).color(if selected { Color32::WHITE } else { theme::TEXT });
                let btn = egui::Button::new(txt).min_size(egui::vec2(32.0, 28.0)).selected(selected);
                let key = s.settings.keymap.label(ui.ctx(), Action::Tool(*k));
                let tip = if key.is_empty() { k.name().to_string() } else { format!("{}  ({key})", k.name()) };
                if ui.add(btn).on_hover_text(tip).clicked() {
                    s.set_tool(*k);
                }
            }
        }
        ui.add_space(10.0);
        // Foreground / background swatches.
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(34.0, 34.0), egui::Sense::click());
        let bg_r = egui::Rect::from_min_size(rect.min + egui::vec2(12.0, 12.0), egui::vec2(20.0, 20.0));
        let fg_r = egui::Rect::from_min_size(rect.min, egui::vec2(20.0, 20.0));
        ui.painter().rect_filled(bg_r, 3.0, theme::to_color32(s.settings.bg));
        ui.painter().rect_stroke(bg_r, 3.0, egui::Stroke::new(1.0, Color32::from_gray(10)), egui::StrokeKind::Outside);
        ui.painter().rect_filled(fg_r, 3.0, theme::to_color32(s.settings.fg));
        ui.painter().rect_stroke(fg_r, 3.0, egui::Stroke::new(1.0, Color32::WHITE), egui::StrokeKind::Outside);
        if resp.clicked() {
            std::mem::swap(&mut s.settings.fg, &mut s.settings.bg);
        }
        resp.on_hover_text("Foreground / background (click or X to swap, D to reset)");
    });
}

fn status_bar(s: &mut AppState, ui: &mut Ui) {
    ui.horizontal_centered(|ui| {
        if let Some(d) = s.doc() {
            let doc = d.doc();
            let size = if doc.info.is_infinite() { "Infinite".to_string() } else { format!("{} × {} px", doc.info.width, doc.info.height) };
            ui.label(RichText::new(size).color(theme::TEXT_DIM));
            ui.separator();
            ui.label(RichText::new(format!("{:.0}%", d.view.zoom * 100.0)).color(theme::TEXT_DIM));
            ui.separator();
            ui.label(RichText::new(if doc.effective_format() == prisme_core::PixelFormat::F32 { "32-bit float" } else { "16-bit float" }).color(theme::TEXT_DIM));
            if let Some(p) = s.hover_doc {
                ui.separator();
                ui.label(RichText::new(format!("x {:.0}  y {:.0}", p[0], p[1])).monospace().color(theme::TEXT_DIM));
            }
            if !d.show_nodes {
                ui.separator();
                ui.label(RichText::new("Raw layer stack (nodes off)").color(theme::WARN));
            }
        }
        ui.separator();
        ui.label(RichText::new(s.tool.name()).color(theme::TEXT_DIM));
        if let Some((m, t)) = &s.status
            && prisme_platform::now_seconds() - t < 6.0 {
                ui.separator();
                ui.label(RichText::new(m).color(theme::TEXT));
            }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if let Some(p) = &s.display_profile {
                ui.label(RichText::new(format!("Display: {p}")).small().color(theme::TEXT_DIM));
            } else {
                ui.label(RichText::new("Display: sRGB").small().color(theme::TEXT_DIM));
            }
            if !s.jobs.is_empty() {
                ui.spinner();
                ui.label(RichText::new(crate::jobs::describe(&s.jobs)).small());
            }
        });
    });
}
