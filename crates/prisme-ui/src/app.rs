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
    Shortcuts,
    Performance,
}

impl Tab {
    pub const ALL: [Tab; 11] =
        [Tab::Canvas, Tab::Layers, Tab::Nodes, Tab::Properties, Tab::Color, Tab::History, Tab::Masks, Tab::Scopes, Tab::Brushes, Tab::Navigator, Tab::Performance];

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
            Tab::Shortcuts => "Keyboard Shortcuts",
            Tab::Performance => "Performance",
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
    /// Frames spent waiting to create the launch document (None once settled).
    pub startup_frames: Option<u32>,
    /// Frame timings (Performance panel, slow-frame warnings).
    pub perf: crate::perf::PerfState,
    /// Show this panel (opened in the focused leaf if it isn't docked).
    pub open_tab: Option<Tab>,
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
    show_below_canvas(dock, Tab::Nodes, 0.62);
}

/// Show `tab`, docking it under the canvas (taking `1 - fraction` of its height) if it isn't
/// open: panels that are used while watching the canvas.
pub fn show_below_canvas(dock: &mut DockState<Tab>, tab: Tab, fraction: f32) {
    if let Some(loc) = dock.find_tab(&tab) {
        let _ = dock.set_active_tab(loc);
        return;
    }
    match dock.find_tab(&Tab::Canvas) {
        Some(path) if path.surface == egui_dock::SurfaceIndex::main() => {
            dock.main_surface_mut().split_below(path.node, fraction, vec![tab]);
        }
        _ => dock.push_to_focused_leaf(tab),
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
        let wake = cc.egui_ctx.clone();
        prisme_platform::install_quit_guard(move || wake.request_repaint());
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
            startup_frames: Some(0),
            perf: Default::default(),
            open_tab: None,
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
    pub fn doc(&self) -> Option<&DocState> {
        self.docs.get(self.active)
    }

    pub fn doc_mut(&mut self) -> Option<&mut DocState> {
        self.docs.get_mut(self.active)
    }

    pub fn toast(&mut self, msg: impl Into<String>) {
        self.status = Some((msg.into(), prisme_platform::now_seconds()));
    }

    /// Release everything held for a document that left `docs` (closed, or a loading
    /// placeholder replaced or abandoned): its node editor, thumbnails, and the engine's GPU
    /// memory for it. Every removal from `docs` goes through here.
    pub fn forget_doc(&mut self, d: DocState) {
        self.node_editors.remove(&d.id);
        if let Some(g) = &mut self.gpu {
            let ids: Vec<_> = g.thumbs.keys().filter(|k| k.0 == d.id).copied().collect();
            let mut r = g.rs.renderer.write();
            for k in ids {
                if let Some((_, t, _)) = g.thumbs.remove(&k) {
                    r.free_texture(&t);
                }
            }
            drop(r);
            g.engine.release_document(d.history.current().uid);
        }
        // Its pixels: a long history holds thousands of tile buffers, ~100 ms to free. Off
        // the UI thread on native (plain data); the web has no thread to spare.
        let crate::doc::DocState { history, working, .. } = d;
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::Builder::new().name("free-document".into()).spawn(move || drop((history, working))).ok();
        #[cfg(target_arch = "wasm32")]
        drop((history, working));
    }

    pub fn add_doc(&mut self, d: DocState) {
        crate::shortcuts::unfocus(self);
        self.startup_frames = None;
        // The blank launch document gives way to the first real one.
        if let Some(i) = self.docs.iter().position(|d| d.blank_startup && !d.is_dirty() && d.path.is_none()) {
            crate::commands::close_doc(self, i);
        }
        self.docs.push(d);
        self.active = self.docs.len() - 1;
    }

    pub fn set_tool(&mut self, k: ToolKind) {
        if k == self.tool {
            return;
        }
        // Leaving Liquify applies its session; other tools abort their interaction
        // (uncommitted previews are discarded).
        crate::stroke::commit_liquify(self);
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
        match tab {
            Tab::Shortcuts => format!("{} {}", egui_phosphor::regular::KEYBOARD, tab.title()).into(),
            _ => tab.title().into(),
        }
    }

    fn id(&mut self, tab: &mut Tab) -> egui::Id {
        egui::Id::new(("prisme-tab", *tab))
    }

    fn ui(&mut self, ui: &mut Ui, tab: &mut Tab) {
        let _span = prisme_core::perf::span(tab.title());
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
            Tab::Shortcuts => crate::shortcuts::ui(self.s, ui),
            Tab::Performance => crate::panels::perf::ui(self.s, ui),
        }
    }

    fn is_closeable(&self, tab: &Tab) -> bool {
        *tab != Tab::Canvas
    }

    fn scroll_bars(&self, tab: &Tab) -> [bool; 2] {
        match tab {
            Tab::Canvas | Tab::Nodes | Tab::Shortcuts => [false, false],
            _ => [false, true],
        }
    }
}

impl eframe::App for PrismeApp {
    fn ui(&mut self, ui: &mut Ui, frame: &mut eframe::Frame) {
        let t0 = prisme_platform::now_seconds();
        if self.s.frame_ms == 0.0 {
            log::info!("Prisme: first frame");
        }
        let ctx = ui.ctx().clone();
        let s = &mut self.s;
        crate::perf::begin(s, frame.info().cpu_usage);
        {
            let _s = prisme_core::perf::span("jobs");
            crate::jobs::poll(s, &ctx);
        }
        {
            let _s = prisme_core::perf::span("land strokes");
            crate::stroke::poll(s, &ctx);
        }
        startup_doc(s, &ctx);
        if let Some(sc) = &mut self.script {
            let _s = prisme_core::perf::span("script");
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
        // App-level quit held by the platform (macOS Cmd+Q / Dock ▸ Quit) counts as a close.
        let quit_req = prisme_platform::take_quit_request();
        let closing = quit_req || ctx.input(|i| i.viewport().close_requested());
        if closing {
            // A stroke still landing isn't in the history yet: land it before checking dirtiness.
            crate::stroke::commit_liquify(s);
            s.docs.iter_mut().for_each(|d| d.finish_stroke());
        }
        if closing && !s.dialogs.allow_quit && s.docs.iter().any(|d| d.is_dirty()) {
            if !quit_req {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            }
            s.dialogs.confirm_quit = true;
        } else if closing {
            // Clean exit (all saved, or changes discarded): nothing to recover next launch.
            crate::jobs::clear_autosaves(s);
            if quit_req {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        crate::jobs::quit_flow(s, &ctx);
        {
            let _s = prisme_core::perf::span("shortcuts");
            crate::commands::handle_shortcuts(s, &ctx);
        }
        handle_dropped_files(s, &ctx);

        let chrome = prisme_core::perf::span("menus & bars");
        egui::Panel::top("menu").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| crate::commands::menu_bar(s, ui, &mut self.dock));
        });
        egui::Panel::top("options").show(ui, |ui| crate::panels::options::ui(s, ui));
        egui::Panel::bottom("status").exact_size(22.0).show(ui, |ui| status_bar(s, ui));
        egui::Panel::left("toolbar").exact_size(44.0).resizable(false).show(ui, |ui| toolbar(s, ui));
        drop(chrome);

        match s.dialogs.shortcuts.request.take() {
            // Show the shortcuts tab, adding it next to Canvas if it isn't open.
            Some(true) => match (self.dock.find_tab(&Tab::Shortcuts), self.dock.find_tab(&Tab::Canvas)) {
                (Some(p), _) => {
                    let _ = self.dock.set_active_tab(p);
                }
                (None, Some(c)) => match self.dock.leaf_mut(egui_dock::NodePath { surface: c.surface, node: c.node }) {
                    Ok(leaf) => leaf.append_tab(Tab::Shortcuts),
                    Err(_) => self.dock.push_to_focused_leaf(Tab::Shortcuts),
                },
                (None, None) => self.dock.push_to_focused_leaf(Tab::Shortcuts),
            },
            Some(false) => {
                if let Some(p) = self.dock.find_tab(&Tab::Shortcuts) {
                    self.dock.remove_tab(p);
                }
            }
            None => {}
        }
        match s.open_tab.take() {
            Some(Tab::Performance) => show_below_canvas(&mut self.dock, Tab::Performance, 0.6),
            Some(t) => match self.dock.find_tab(&t) {
                Some(p) => {
                    let _ = self.dock.set_active_tab(p);
                }
                None => self.dock.push_to_focused_leaf(t),
            },
            None => {}
        }
        if std::mem::take(&mut s.dialogs.shortcuts.reveal)
            && let Some(loc) = self.dock.find_tab(&Tab::Canvas) {
                let _ = self.dock.set_active_tab(loc);
            }
        // Visible = active tab of its leaf; a hidden or closed tab stops any rebind in progress.
        let visible = self.dock.find_tab(&Tab::Shortcuts).is_some_and(|p| {
            self.dock.leaf(egui_dock::NodePath { surface: p.surface, node: p.node }).is_ok_and(|l| l.active == p.tab)
        });
        s.dialogs.shortcuts.focused = visible;
        s.dialogs.shortcuts.dock_focused = matches!(self.dock.find_active_focused(), Some((_, Tab::Shortcuts)));
        if !visible {
            crate::shortcuts::stop_editing(s);
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

        {
            let _s = prisme_core::perf::span("dialogs");
            crate::dialogs::show(s, &ctx);
        }
        {
            let _s = prisme_core::perf::span("autosave");
            crate::jobs::autosave(s);
        }
        if !s.jobs.is_empty() || s.gpu.as_ref().is_some_and(|g| g.pick.is_some() || g.scopes.busy()) {
            ctx.request_repaint();
        }
        s.frame_ms = ((prisme_platform::now_seconds() - t0) * 1000.0) as f32;
        crate::perf::end(s, &ctx, s.frame_ms);
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

/// Opened with nothing to show (no file, nothing to recover): start a blank document the
/// size of the screen, framed by the canvas's fit on first show.
fn startup_doc(s: &mut AppState, ctx: &egui::Context) {
    let Some(frames) = s.startup_frames else { return };
    // Files from the command line decode as jobs; a recovery prompt decides for itself.
    if !s.docs.is_empty() {
        s.startup_frames = None;
        return;
    }
    if !s.jobs.is_empty() || !s.dialogs.recovery.is_empty() {
        return;
    }
    let native = ctx.input(|i| {
        let v = i.viewport();
        v.monitor_size.map(|m| m * v.native_pixels_per_point.unwrap_or(i.pixels_per_point))
    });
    let size = native.map(|m| [m.x.round() as u32, m.y.round() as u32]).or_else(prisme_platform::screen_size_px).filter(|s| s[0] > 0 && s[1] > 0);
    // The monitor can take a frame or two to be known.
    let size = match size {
        Some(sz) => sz,
        None if frames < 10 => {
            s.startup_frames = Some(frames + 1);
            ctx.request_repaint();
            return;
        }
        None => [1920, 1080],
    };
    let nd = crate::dialogs::NewDoc::default();
    let mut d = DocState::new(prisme_core::Document::with_background(nd.name, size[0], size[1], Some([1.0; 4])));
    d.blank_startup = true;
    s.add_doc(d);
}

fn handle_dropped_files(s: &mut AppState, ctx: &egui::Context) {
    let files = ctx.input(|i| i.raw.dropped_files.clone());
    for f in files {
        let path = f.path().to_path_buf();
        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "dropped".into());
        // Files pasted on the web come this way too: Cmd+Alt+V opens them separately.
        let separately = ctx.input(|i| i.modifiers.shift || i.modifiers.alt);
        let into = drop_target(s, &name, separately);
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

/// Where a dropped or pasted file goes: placed as a layer in the open document, unless opened
/// `separately` (Shift), it is a project, or the document is the untouched launch one (which
/// the file replaces).
fn drop_target(s: &AppState, name: &str, separately: bool) -> Option<u64> {
    let blank = s.doc().is_some_and(|d| d.blank_startup && !d.is_dirty());
    let place = !s.docs.is_empty() && !blank && !separately && !prisme_io::is_project_file(name);
    if place { s.doc().map(|d| d.id) } else { None }
}

/// Opens a file from disk the way a drop does (files copied in a file manager and pasted).
pub fn open_dropped_path(s: &mut AppState, path: std::path::PathBuf, separately: bool) {
    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "pasted".into());
    match std::fs::read(&path) {
        Ok(bytes) => {
            let into = drop_target(s, &name, separately);
            crate::jobs::decode_file(s, name, bytes, into, Some(path));
        }
        Err(e) => s.toast(format!("Can't read {name}: {e}")),
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
        ToolKind::Liquify => ph::HURRICANE,
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
            &[ToolKind::Clone, ToolKind::Heal, ToolKind::Smudge, ToolKind::Blur, ToolKind::Liquify],
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
