//! Modal-ish windows: new document, image/canvas size, PNG export, preferences, etc.

use crate::app::AppState;
use crate::doc::DocState;
use crate::theme;
use egui::{ComboBox, DragValue, RichText, Window};
use prisme_core::Document;
use prisme_export::{BitDepth, ColorMode, ColorTag, FilterStrategy, ResizeFilter, TargetPriority, TargetSize};
use prisme_tools::transform::Resample;
use std::sync::Arc;

pub struct NewDoc {
    pub open: bool,
    pub w: u32,
    pub h: u32,
    pub name: String,
    pub fill: u8, // 0 white, 1 black, 2 transparent, 3 background colour
    pub f32: bool,
    /// Canvas mode chosen at the top of the dialog: infinite has no page, so no size fields.
    pub infinite: bool,
}

/// Hidden starting frame of an infinite canvas (initial view; centre for Vignette / Transform).
const INFINITE_FRAME: (u32, u32) = (1920, 1080);

impl Default for NewDoc {
    fn default() -> Self {
        Self { open: false, w: 1920, h: 1080, name: "Untitled".into(), fill: 0, f32: false, infinite: false }
    }
}

#[derive(Default)]
pub struct ExportDialog {
    pub open: bool,
    pub pixels: Option<(Arc<Vec<f32>>, u32, u32)>,
    pub estimate: Option<u64>,
    pub last_result: Option<String>,
    pub notes: Vec<String>,
    pub target_kb: f32,
    pub opts_snapshot: Option<String>,
    /// Exporting the selection (Export Selection…) rather than the whole document.
    pub selection: bool,
}

#[derive(Default)]
pub struct Dialogs {
    pub new_doc: NewDoc,
    pub image_size: Option<(u32, u32, bool, u8)>,
    pub canvas_size: Option<(u32, u32, [f32; 2])>,
    pub export: ExportDialog,
    pub prefs_open: bool,
    pub about_open: bool,
    pub confirm_close: Option<usize>,
    /// Documents to close once their save completes.
    pub close_after_save: Vec<u64>,
    /// "Unsaved changes" prompt shown on window close.
    pub confirm_quit: bool,
    /// Saving dirty documents one by one, then quitting.
    pub quit_after_save: bool,
    /// Let the next close request through.
    pub allow_quit: bool,
    pub modify_sel: Option<(String, f32)>,
    pub recovery: Vec<(u64, String)>,
    pub shortcuts: crate::shortcuts::ShortcutsTab,
}

impl Dialogs {
    pub fn open_image_size(&mut self, d: Option<&DocState>) {
        if let Some(d) = d {
            self.image_size = Some((d.doc().info.width, d.doc().info.height, true, 2));
        }
    }

    pub fn open_canvas_size(&mut self, d: Option<&DocState>) {
        if let Some(d) = d {
            self.canvas_size = Some((d.doc().info.width, d.doc().info.height, [0.5, 0.5]));
        }
    }
}

/// Open the PNG export dialog for the whole document, or for the selection's bounds (alpha
/// applied) when `selection` and there is one. The whole document of an infinite canvas is
/// its content bounds.
pub fn open_export(s: &mut AppState, selection: bool) {
    if s.docs.is_empty() {
        return;
    }
    s.dialogs.export.selection = selection && s.doc().is_some_and(|d| d.doc().selection.is_some());
    s.dialogs.export.open = true;
    s.dialogs.export.pixels = None;
    s.dialogs.export.estimate = None;
    if s.dialogs.export.target_kb <= 0.0 {
        s.dialogs.export.target_kb = 500.0;
    }
    start_export_render(s);
}

/// (Re-)render the export dialog's region.
pub fn start_export_render(s: &mut AppState) {
    let sel = s.dialogs.export.selection;
    crate::jobs::start_render(s, sel, crate::jobs::RenderPurpose::Export { then_encode: false });
}

pub fn show(s: &mut AppState, ctx: &egui::Context) {
    new_doc(s, ctx);
    image_size(s, ctx);
    canvas_size(s, ctx);
    export(s, ctx);
    prefs(s, ctx);
    misc(s, ctx);
}

fn new_doc(s: &mut AppState, ctx: &egui::Context) {
    if !s.dialogs.new_doc.open {
        return;
    }
    let mut open = true;
    let mut create = false;
    Window::new("New Image").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
        let nd = &mut s.dialogs.new_doc;
        // Mode first: it decides which fields exist below.
        egui::Grid::new("new-doc").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
            ui.label("Canvas");
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                ui.selectable_value(&mut nd.infinite, false, format!("{} Fixed size", egui_phosphor::regular::FRAME_CORNERS));
                ui.selectable_value(&mut nd.infinite, true, format!("{} Infinite", egui_phosphor::regular::INFINITY));
            });
            ui.end_row();
            ui.label("Name");
            ui.text_edit_singleline(&mut nd.name);
            ui.end_row();
            if nd.infinite {
                ui.label("");
                ui.label(RichText::new("No page: paint anywhere.\nExport Selection exports a region; Export exports everything you painted.").small().color(theme::TEXT_DIM));
                ui.end_row();
            } else {
                ui.label("Size");
                ui.horizontal(|ui| {
                    ui.add(DragValue::new(&mut nd.w).range(1..=65535).suffix(" px"));
                    ui.label("×");
                    ui.add(DragValue::new(&mut nd.h).range(1..=65535).suffix(" px"));
                });
                ui.end_row();
                ui.label("");
                ui.horizontal_wrapped(|ui| {
                    ui.set_max_width(420.0);
                    for (n, w, h) in [("HD", 1920, 1080), ("4K", 3840, 2160), ("Square", 2048, 2048), ("A4 300dpi", 2480, 3508), ("Instagram", 1080, 1350), ("Icon", 512, 512)] {
                        if ui.selectable_label(nd.w == w && nd.h == h, n).clicked() {
                            nd.w = w;
                            nd.h = h;
                        }
                    }
                });
                ui.end_row();
            }
            ui.label("Background");
            ui.horizontal(|ui| {
                for (i, n) in ["White", "Black", "Transparent", "BG colour"].iter().enumerate() {
                    ui.selectable_value(&mut nd.fill, i as u8, *n);
                }
            });
            ui.end_row();
            ui.label("");
            ui.checkbox(&mut nd.f32, "32-bit float storage (HDR / heavy grading)");
            ui.end_row();
        });
        ui.add_space(6.0);
        // Fixed-height row: a bare right_to_left layout would take all remaining height.
        ui.allocate_ui_with_layout(egui::vec2(ui.available_width(), 24.0), egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button(RichText::new("Create").strong()).clicked() {
                create = true;
            }
        });
    });
    if create {
        let nd = &s.dialogs.new_doc;
        let fill = match nd.fill {
            0 => Some([1.0, 1.0, 1.0, 1.0]),
            1 => Some([0.0, 0.0, 0.0, 1.0]),
            2 => None,
            _ => Some(s.settings.bg),
        };
        let mut d = if nd.infinite { Document::infinite(nd.name.clone(), INFINITE_FRAME.0, INFINITE_FRAME.1, fill) } else { Document::with_background(nd.name.clone(), nd.w, nd.h, fill) };
        if nd.f32 {
            d.info.format = prisme_core::PixelFormat::F32;
            for l in &mut d.layers {
                if let prisme_core::LayerKind::Raster(g) = &mut l.kind {
                    *g = Arc::new(g.convert_format(prisme_core::PixelFormat::F32));
                }
            }
        }
        s.add_doc(DocState::new(d));
        open = false;
    }
    s.dialogs.new_doc.open = open;
}

fn image_size(s: &mut AppState, ctx: &egui::Context) {
    let Some((mut w, mut h, mut keep, mut rs)) = s.dialogs.image_size else { return };
    let Some(d) = s.doc() else { return };
    let (ow, oh) = (d.doc().info.width, d.doc().info.height);
    let mut open = true;
    let mut apply = false;
    Window::new("Image Size").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
        ui.label(RichText::new(format!("Current: {ow} × {oh}")).color(theme::TEXT_DIM));
        ui.horizontal(|ui| {
            ui.label("Width");
            if ui.add(DragValue::new(&mut w).range(1..=65535)).changed() && keep {
                h = ((w as f64 / ow as f64) * oh as f64).round().max(1.0) as u32;
            }
            ui.label("Height");
            if ui.add(DragValue::new(&mut h).range(1..=65535)).changed() && keep {
                w = ((h as f64 / oh as f64) * ow as f64).round().max(1.0) as u32;
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut keep, "Keep proportions");
            for pct in [25, 50, 200] {
                if ui.small_button(format!("{pct}%")).clicked() {
                    w = (ow * pct / 100).max(1);
                    h = (oh * pct / 100).max(1);
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Resample");
            for (i, n) in ["Nearest", "Bilinear", "Bicubic"].iter().enumerate() {
                ui.selectable_value(&mut rs, i as u8, *n);
            }
        });
        if ui.button("Apply").clicked() {
            apply = true;
        }
    });
    if apply {
        let mode = [Resample::Nearest, Resample::Bilinear, Resample::Bicubic][rs as usize];
        crate::commands::edit(s, "Image Size", |d| prisme_tools::transform::resize_document(d, w, h, mode));
        open = false;
    }
    s.dialogs.image_size = open.then_some((w, h, keep, rs));
}

fn canvas_size(s: &mut AppState, ctx: &egui::Context) {
    let Some((mut w, mut h, mut anchor)) = s.dialogs.canvas_size else { return };
    let mut open = true;
    let mut apply = false;
    Window::new("Canvas Size").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label("Width");
            ui.add(DragValue::new(&mut w).range(1..=65535));
            ui.label("Height");
            ui.add(DragValue::new(&mut h).range(1..=65535));
        });
        ui.label("Anchor");
        egui::Grid::new("anchor").show(ui, |ui| {
            for y in 0..3 {
                for x in 0..3 {
                    let a = [x as f32 * 0.5, y as f32 * 0.5];
                    if ui.selectable_label(anchor == a, if anchor == a { "●" } else { "○" }).clicked() {
                        anchor = a;
                    }
                }
                ui.end_row();
            }
        });
        if ui.button("Apply").clicked() {
            apply = true;
        }
    });
    if apply {
        crate::commands::edit(s, "Canvas Size", |d| d.resize_canvas(w, h, anchor[0], anchor[1]));
        open = false;
    }
    s.dialogs.canvas_size = open.then_some((w, h, anchor));
}

fn export(s: &mut AppState, ctx: &egui::Context) {
    if !s.dialogs.export.open {
        return;
    }
    let mut open = true;
    let mut do_export = false;
    let mut opts = s.settings.export.clone();
    let rendering = s.jobs.iter().any(|j| matches!(j, crate::jobs::Job::Render(_)));
    let encoding = s.jobs.iter().any(|j| matches!(j, crate::jobs::Job::Encode { .. }));
    let title = if s.dialogs.export.selection { "Export Selection" } else { "Export PNG" };
    Window::new(title).id(egui::Id::new("export-png")).open(&mut open).collapsible(false).resizable(false).default_width(380.0).show(ctx, |ui| {
        egui::Grid::new("png-opts").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
            ui.label("Colour");
            ComboBox::from_id_salt("cm").selected_text(format!("{:?}", opts.color_mode)).show_ui(ui, |ui| {
                for m in [ColorMode::Auto, ColorMode::Rgba, ColorMode::Rgb, ColorMode::GrayAlpha, ColorMode::Gray, ColorMode::Palette] {
                    ui.selectable_value(&mut opts.color_mode, m, format!("{m:?}"));
                }
            });
            ui.end_row();
            ui.label("Bit depth");
            ComboBox::from_id_salt("bd").selected_text(format!("{:?}", opts.bit_depth)).show_ui(ui, |ui| {
                for b in [BitDepth::Auto, BitDepth::One, BitDepth::Two, BitDepth::Four, BitDepth::Eight, BitDepth::Sixteen] {
                    ui.selectable_value(&mut opts.bit_depth, b, format!("{b:?}"));
                }
            });
            ui.end_row();
            if opts.color_mode == ColorMode::Palette {
                ui.label("Colours");
                ui.add(egui::Slider::new(&mut opts.palette_colors, 2..=256));
                ui.end_row();
                ui.label("Dither");
                ui.add(egui::Slider::new(&mut opts.dither, 0.0..=1.0));
                ui.end_row();
            }
            ui.label("Colour space");
            ComboBox::from_id_salt("cs").selected_text(opts.target_space.name()).show_ui(ui, |ui| {
                for sp in prisme_color::RgbSpace::ALL {
                    ui.selectable_value(&mut opts.target_space, sp, sp.name());
                }
            });
            ui.end_row();
            ui.label("Tag");
            ComboBox::from_id_salt("tag").selected_text(format!("{:?}", opts.color_tag)).show_ui(ui, |ui| {
                for t in [ColorTag::Icc, ColorTag::Cicp, ColorTag::SrgbChunk, ColorTag::None] {
                    ui.selectable_value(&mut opts.color_tag, t, format!("{t:?}"));
                }
            });
            ui.end_row();
            ui.label("Compression");
            ui.add(egui::Slider::new(&mut opts.effort, 0..=6).text("effort"));
            ui.end_row();
            ui.label("");
            ui.horizontal(|ui| {
                ui.checkbox(&mut opts.zopfli, "Zopfli");
                if opts.zopfli {
                    ui.add(DragValue::new(&mut opts.zopfli_iterations).range(1..=100).suffix(" iter"));
                }
            });
            ui.end_row();
            ui.label("Filter");
            ComboBox::from_id_salt("flt").selected_text(format!("{:?}", opts.filter)).show_ui(ui, |ui| {
                for f in [FilterStrategy::Auto, FilterStrategy::None, FilterStrategy::Sub, FilterStrategy::Up, FilterStrategy::Average, FilterStrategy::Paeth, FilterStrategy::MinSum, FilterStrategy::Entropy, FilterStrategy::Brute] {
                    ui.selectable_value(&mut opts.filter, f, format!("{f:?}"));
                }
            });
            ui.end_row();
            ui.label("");
            ui.horizontal(|ui| {
                ui.checkbox(&mut opts.interlace, "Interlace");
                ui.checkbox(&mut opts.strip_metadata, "Strip metadata");
                ui.checkbox(&mut opts.clean_transparent, "Clean transparent");
            });
            ui.end_row();
            ui.label("Scale");
            ui.horizontal(|ui| {
                let mut pct = opts.scale * 100.0;
                if ui.add(DragValue::new(&mut pct).range(1.0..=400.0).suffix("%")).changed() {
                    opts.scale = pct / 100.0;
                }
                ComboBox::from_id_salt("rf").selected_text(format!("{:?}", opts.resize_filter)).show_ui(ui, |ui| {
                    for f in [ResizeFilter::Lanczos3, ResizeFilter::Mitchell, ResizeFilter::Bilinear, ResizeFilter::Nearest] {
                        ui.selectable_value(&mut opts.resize_filter, f, format!("{f:?}"));
                    }
                });
            });
            ui.end_row();
        });
        ui.separator();
        let mut target = opts.target.is_some();
        ui.checkbox(&mut target, RichText::new("Target file size").strong());
        if target && opts.target.is_none() {
            opts.target = Some(TargetSize { max_bytes: (s.dialogs.export.target_kb * 1024.0) as u64, allow_resize: false, min_scale: 0.25, priority: TargetPriority::Balanced, allow_lossy_palette: true });
        } else if !target {
            opts.target = None;
        }
        if let Some(t) = &mut opts.target {
            ui.horizontal(|ui| {
                let mut kb = t.max_bytes as f32 / 1024.0;
                if ui.add(DragValue::new(&mut kb).range(1.0..=1_000_000.0).suffix(" KB")).changed() {
                    t.max_bytes = (kb * 1024.0) as u64;
                    s.dialogs.export.target_kb = kb;
                }
                ui.checkbox(&mut t.allow_lossy_palette, "Allow palette");
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut t.allow_resize, "Allow resolution changes");
                if t.allow_resize {
                    ui.label("min");
                    let mut pct = t.min_scale * 100.0;
                    if ui.add(DragValue::new(&mut pct).range(5.0..=100.0).suffix("%")).changed() {
                        t.min_scale = pct / 100.0;
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("Priority");
                for (p, n) in [(TargetPriority::ColorsFirst, "Reduce colours first"), (TargetPriority::ResolutionFirst, "Reduce size first"), (TargetPriority::Balanced, "Balanced")] {
                    ui.selectable_value(&mut t.priority, p, n);
                }
            });
        }
        ui.separator();
        match (&s.dialogs.export.pixels, rendering) {
            (_, true) => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Rendering full resolution…");
                });
            }
            (Some((_, w, h)), false) => {
                let est = s.dialogs.export.estimate.map(|b| format!("≈ {:.1} KB", b as f64 / 1024.0)).unwrap_or("estimating…".into());
                ui.label(format!("{} × {} px  ·  estimated {est}", (*w as f32 * opts.scale).round(), (*h as f32 * opts.scale).round()));
            }
            _ => {}
        }
        if let Some(r) = &s.dialogs.export.last_result {
            ui.label(RichText::new(format!("Last export: {r}")).color(theme::ACCENT));
            for n in &s.dialogs.export.notes {
                ui.label(RichText::new(format!("• {n}")).small().color(theme::TEXT_DIM));
            }
        }
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let ready = s.dialogs.export.pixels.is_some() && !rendering && !encoding;
            if ui.add_enabled(ready, egui::Button::new(RichText::new("Export…").strong())).clicked() {
                do_export = true;
            }
            if encoding {
                ui.spinner();
                let progress = s.jobs.iter().find_map(|j| match j {
                    crate::jobs::Job::Encode { control, .. } => control.progress(),
                    _ => None,
                });
                match progress {
                    Some((f, stage)) => ui.label(format!("{stage}… {:.0}%", f * 100.0)),
                    None => ui.label("Encoding…"),
                };
                if ui.button("Cancel").clicked() {
                    crate::jobs::cancel_encode(s);
                }
            }
            if ui.button("Re-render").on_hover_text("Re-render the image (after edits)").clicked() {
                start_export_render(s);
            }
        });
    });
    // Re-estimate when options change.
    let snap = serde_json::to_string(&opts).ok();
    if snap != s.dialogs.export.opts_snapshot {
        s.dialogs.export.opts_snapshot = snap;
        if let Some((px, w, h)) = s.dialogs.export.pixels.clone() {
            s.settings.export = opts.clone();
            crate::jobs::start_estimate(s, px, w, h);
        }
    }
    s.settings.export = opts;
    if do_export
        && let Some((px, w, h)) = s.dialogs.export.pixels.clone() {
            crate::jobs::start_encode(s, px, w, h);
        }
    if !open {
        s.dialogs.export.pixels = None;
    }
    s.dialogs.export.open = open;
}

fn prefs(s: &mut AppState, ctx: &egui::Context) {
    if !s.dialogs.prefs_open {
        return;
    }
    let mut open = true;
    let mut shortcuts = false;
    Window::new("Preferences").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
        ui.label(RichText::new("General").strong());
        ui.horizontal(|ui| {
            ui.label("Autosave every");
            ui.add(DragValue::new(&mut s.settings.autosave_minutes).range(0.0..=60.0).suffix(" min"));
            ui.label(RichText::new("(0 = off)").small().color(theme::TEXT_DIM));
        });
        ui.horizontal(|ui| {
            ui.label("History memory budget");
            let mut mb = s.docs.first().map(|d| d.history.budget_bytes as f32 / 1e6).unwrap_or(2048.0);
            if ui.add(DragValue::new(&mut mb).range(64.0..=65536.0).suffix(" MB")).changed() {
                for d in &mut s.docs {
                    d.history.budget_bytes = (mb * 1e6) as usize;
                }
            }
        });
        ui.label(RichText::new(format!("Platform: {} · fonts loaded: {}", prisme_platform::platform_name(), s.fonts.families().len())).small().color(theme::TEXT_DIM));
        ui.separator();
        ui.collapsing("Display", |ui| display_prefs(s, ui));
        ui.separator();
        shortcuts = ui.button(format!("{} Keyboard Shortcuts…", egui_phosphor::regular::KEYBOARD)).clicked();
    });
    if shortcuts {
        crate::shortcuts::open(s);
        open = false;
    }
    s.dialogs.prefs_open = open;
}

fn display_prefs(s: &mut AppState, ui: &mut egui::Ui) {
    use crate::hdr::HdrMode;
    use prisme_gpu::HighlightMode;
    let hdr = s.hdr.clone();
    ui.add_enabled_ui(hdr.capable, |ui| {
        ui.horizontal(|ui| {
            ui.label("HDR output");
            ComboBox::from_id_salt("hdr-mode").selected_text(s.settings.hdr_mode.label()).show_ui(ui, |ui| {
                for m in HdrMode::ALL {
                    ui.selectable_value(&mut s.settings.hdr_mode, m, m.label());
                }
            });
        })
        .response
        .on_hover_text("Auto: extended range when the display reports HDR capability. Values above 1.0 (SDR white) are shown brighter than white; the UI stays at SDR white.");
    });
    let state = if hdr.active {
        let cur = hdr.headroom.map(|h| format!("{h:.2}×")).unwrap_or_else(|| "unknown".into());
        let pot = hdr.potential.map(|p| format!(", up to {p:.1}×")).unwrap_or_default();
        format!("Active — headroom {cur}{pot} SDR white; highlights limited to {:.2}×", hdr.peak)
    } else {
        format!("Inactive — {}", hdr.reason)
    };
    ui.label(RichText::new(state).small().color(if hdr.active || hdr.capable { theme::TEXT_DIM } else { theme::WARN }));
    if let (Some(f), Some(c)) = (hdr.format, hdr.color_space) {
        ui.label(RichText::new(format!("Surface: {f:?} · {c:?}")).small().color(theme::TEXT_DIM));
    }
    ui.add_enabled_ui(hdr.capable, |ui| {
        ui.horizontal(|ui| {
            ui.label("Highlights");
            ui.selectable_value(&mut s.settings.hdr_highlights, HighlightMode::SoftRolloff, "Soft roll-off")
                .on_hover_text("Compress highlights smoothly from 75% of the display peak (keeps hue and detail)");
            ui.selectable_value(&mut s.settings.hdr_highlights, HighlightMode::Clip, "Clip at display peak");
        });
        // Only meaningful where the platform doesn't report the headroom (browsers).
        ui.add_enabled_ui(hdr.headroom.is_none(), |ui| {
            ui.horizontal(|ui| {
                ui.label("Assumed display peak");
                ui.add(DragValue::new(&mut s.settings.hdr_peak_fallback).range(1.0..=16.0).speed(0.05).suffix("× SDR white"));
            })
            .response
            .on_hover_text("Used when the platform doesn't report the display's HDR headroom (web browsers). SDR white itself follows the OS brightness setting.");
        });
    });
    ui.label(
        RichText::new(if hdr.active {
            "In HDR mode the OS/browser colour-manages the canvas (extended sRGB); the display ICC profile and dithering apply to SDR only."
        } else {
            "SDR: the canvas uses the display ICC profile when enabled (View menu)."
        })
        .small()
        .color(theme::TEXT_DIM),
    );
}

fn misc(s: &mut AppState, ctx: &egui::Context) {
    if s.dialogs.about_open {
        let mut open = true;
        Window::new("About Prisme").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
            ui.label(RichText::new("Prisme").size(24.0).color(theme::ACCENT));
            ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
            ui.label("All-Rust, GPU-accelerated, colour-managed image editor.");
            ui.label(RichText::new("Working space: linear Rec.2020 · float pipeline · WebGPU/Metal/Vulkan/DX12").small().color(theme::TEXT_DIM));
        });
        s.dialogs.about_open = open;
    }
    if let Some(i) = s.dialogs.confirm_close {
        let name = s.docs.get(i).map(|d| d.doc().info.name.clone()).unwrap_or_default();
        let pick = confirm(ctx, "Unsaved changes", &["Save", "Don't save", "Cancel"], |ui| {
            ui.label(format!("“{name}” has unsaved changes."));
        });
        match pick {
            Some(0) => {
                s.dialogs.confirm_close = None;
                if let Some(id) = s.docs.get(i).map(|d| d.id) {
                    s.dialogs.close_after_save.push(id);
                    crate::jobs::save_doc(s, i, false);
                }
            }
            Some(1) => crate::commands::close_doc(s, i),
            Some(_) => s.dialogs.confirm_close = None,
            None => {}
        }
    }
    if s.dialogs.confirm_quit {
        let names: Vec<String> = s.docs.iter().filter(|d| d.is_dirty()).map(|d| d.doc().info.name.clone()).collect();
        let save = if names.len() == 1 { "Save" } else { "Save all" };
        let pick = confirm(ctx, "Quit Prisme?", &[save, "Don't save", "Cancel"], |ui| {
            ui.label(if names.len() == 1 { "This document has unsaved changes:".to_string() } else { format!("{} documents have unsaved changes:", names.len()) });
            for n in &names {
                ui.label(RichText::new(format!("• {n}")).color(theme::TEXT_DIM));
            }
        });
        match pick {
            Some(0) => {
                s.dialogs.confirm_quit = false;
                s.dialogs.quit_after_save = true;
            }
            Some(1) => {
                crate::jobs::discard_autosaves(s);
                s.dialogs.confirm_quit = false;
                s.dialogs.allow_quit = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            Some(_) => s.dialogs.confirm_quit = false,
            None => {}
        }
    }
    if let Some((op, mut amount)) = s.dialogs.modify_sel.clone() {
        let mut open = true;
        let mut apply = false;
        Window::new(format!("{} selection", op[..1].to_uppercase() + &op[1..])).open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
            ui.add(DragValue::new(&mut amount).range(0.5..=500.0).suffix(" px"));
            if ui.button("Apply").clicked() {
                apply = true;
            }
        });
        if apply {
            crate::commands::selection_modify(s, &op, amount);
            open = false;
        }
        s.dialogs.modify_sel = open.then_some((op, amount));
    }
    if !s.dialogs.recovery.is_empty() {
        Window::new("Recover documents").collapsible(false).resizable(false).show(ctx, |ui| {
            ui.label("Prisme found autosaved documents from a previous session:");
            let list = s.dialogs.recovery.clone();
            for (id, name) in &list {
                ui.horizontal(|ui| {
                    ui.label(name);
                    if ui.button("Restore").clicked() {
                        crate::jobs::recover(s, *id);
                        s.dialogs.recovery.retain(|x| x.0 != *id);
                    }
                });
            }
            if ui.button("Discard all").clicked() {
                crate::jobs::discard_recovery(s);
            }
        });
    }
}

/// Centred modal confirmation. Keyboard focus stays inside it: the first button starts
/// focused, Tab/arrows move, Enter/Space press, Escape picks the last (cancel) button.
/// Returns the index of the button chosen this frame.
fn confirm(ctx: &egui::Context, title: &str, buttons: &[&str], body: impl FnOnce(&mut egui::Ui)) -> Option<usize> {
    let mut pick = None;
    egui::Modal::new(egui::Id::new(("confirm", title))).show(ctx, |ui| {
        ui.set_min_width(260.0);
        ui.label(RichText::new(title).strong().size(15.0));
        ui.add_space(4.0);
        body(ui);
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            let resps: Vec<egui::Response> = buttons.iter().map(|b| ui.button(*b)).collect();
            if !resps.iter().any(|r| r.has_focus()) {
                resps[0].request_focus();
            }
            pick = resps.iter().position(|r| r.clicked());
        });
    });
    if pick.is_none() && ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
        pick = Some(buttons.len() - 1);
    }
    pick
}
