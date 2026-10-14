//! Background jobs: opening/decoding, saving, export rendering/encoding, clipboard,
//! screen picking, font/LUT loading, autosave and crash recovery.

use crate::app::AppState;
use crate::doc::DocState;
use prisme_core::{Document, IRect, NodeId};
use prisme_gpu::{EvalFormat, EvalRequest, Readback, RenderSource};
use prisme_platform::{FileFilter, Pending, PickedFile};
use std::sync::Arc;

pub use crate::offload::Loaded;
use crate::offload::{DecodeJob, EncodePngJob, SaveJob};
use prisme_platform::JobControl;

pub enum Job {
    OpenDialog { pending: Pending<Vec<PickedFile>>, place: bool },
    Decode { name: String, pending: Pending<Result<Loaded, String>>, into: Option<u64>, placeholder: Option<u64>, path: Option<std::path::PathBuf> },
    Save { pending: Pending<Result<Option<String>, String>>, doc: u64, revision: u64 },
    ClipboardSet(Pending<Result<(), String>>),
    ClipboardGet { pending: Pending<Option<prisme_platform::ClipboardImage>>, new_doc: bool },
    ScreenPick { pending: Pending<Option<[u8; 3]>>, to_bg: bool },
    FontFiles(Pending<Vec<PickedFile>>),
    LutFile { pending: Pending<Vec<PickedFile>>, doc: u64, node: NodeId, key: String },
    Render(RenderJob),
    Encode { pending: Pending<Result<prisme_export::ExportResult, String>>, control: JobControl, name: String },
    Estimate(Pending<u64>),
    Autosave(Pending<Result<(), String>>),
    Recover(Pending<Option<Vec<u8>>>),
    RecoverLoad(Pending<Result<Loaded, String>>),
    Dropped { pending: Pending<Result<Vec<u8>, String>>, name: String, into: Option<u64> },
}

pub fn describe(jobs: &[Job]) -> String {
    jobs.iter()
        .map(|j| match j {
            Job::Decode { name, .. } => format!("Opening {name}"),
            Job::Render(r) => format!("{} {:.0}%", if r.purpose == RenderPurpose::CopyMerged { "Copying" } else { "Rendering" }, r.next as f32 / r.tiles.len().max(1) as f32 * 100.0),
            Job::Encode { control, .. } => match control.progress() {
                Some((f, stage)) => format!("Encoding PNG {:.0}% ({stage})", f * 100.0),
                None => "Encoding PNG".into(),
            },
            Job::Save { .. } => "Saving".into(),
            Job::Estimate(_) => "Estimating size".into(),
            _ => String::new(),
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" · ")
}

fn image_filters() -> Vec<FileFilter> {
    let mut exts: Vec<&str> = prisme_io::supported_extensions().to_vec();
    exts.push("prisme");
    vec![FileFilter::new("Images & projects", &exts), FileFilter::new("Prisme project", &["prisme"])]
}

pub fn open_dialog(s: &mut AppState) {
    s.jobs.push(Job::OpenDialog { pending: prisme_platform::open_files("Open", &image_filters(), true), place: false });
}

pub fn place_dialog(s: &mut AppState) {
    let f = vec![FileFilter::new("Images", prisme_io::supported_extensions())];
    s.jobs.push(Job::OpenDialog { pending: prisme_platform::open_files("Place", &f, false), place: true });
}

/// Decode a file in the background. `into` places it as layer(s) in an open document.
pub fn decode_file(s: &mut AppState, name: String, bytes: Vec<u8>, into: Option<u64>, path: Option<std::path::PathBuf>) {
    let project = prisme_io::is_project_file(&name);
    // Instant feedback: show an embedded/fast preview while the full decode runs.
    let mut placeholder = None;
    if into.is_none() {
        let preview = if project {
            prisme_io::container::read_preview(&bytes).map(|(w, h, rgba)| {
                let px: Vec<f32> = rgba.chunks(4).flat_map(|p| {
                    let c = prisme_color::srgb().to_working([p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0]);
                    [c[0], c[1], c[2], p[3] as f32 / 255.0]
                }).collect();
                (w, h, px)
            })
        } else {
            prisme_io::decode_preview(&bytes, &name, 1024).map(|p| (p.width, p.height, p.pixels))
        };
        if let Some((w, h, px)) = preview {
            let mut d = Document::new(format!("{name} (loading…)"), w, h, prisme_core::PixelFormat::F16);
            let id = d.alloc_layer_id();
            let g = prisme_core::TileGrid::from_dense(prisme_core::PixelFormat::F16, 4, w, h, &px);
            d.layers.push(prisme_core::Layer::new(id, "Preview", prisme_core::LayerKind::Raster(Arc::new(g))));
            let ds = DocState::new(d);
            placeholder = Some(ds.id);
            s.add_doc(ds);
        }
    }
    // Decoding (and building tiles) runs on a thread natively, in a Web Worker on the web.
    let place_into = into.and_then(|id| s.docs.iter().find(|d| d.id == id)).map(|d| (d.doc().info.width, d.doc().info.height));
    let pending = prisme_platform::spawn_worker(DecodeJob { name: name.clone(), bytes, place_into }).pending;
    s.jobs.push(Job::Decode { name, pending, into, placeholder, path });
}

pub fn save(s: &mut AppState, save_as: bool) {
    if s.doc().is_some() {
        save_doc(s, s.active, save_as);
    }
}

/// Save document `i` (encoded in the background). Returns false when nothing was started.
pub fn save_doc(s: &mut AppState, i: usize, save_as: bool) -> bool {
    let Some(d) = s.docs.get_mut(i) else { return false };
    // Land a GPU stroke still in flight so it is part of what gets saved.
    d.finish_stroke();
    let d = &s.docs[i];
    let (doc_id, rev) = (d.id, d.history.revision());
    // Encoding runs in the background (thread / Web Worker); the history clone is cheap.
    let encoded = prisme_platform::spawn_worker(SaveJob { history: d.history.clone(), include_history: true }).pending;
    #[cfg(not(target_arch = "wasm32"))]
    if !save_as
        && let Some(p) = &d.path
        && p.extension().is_some_and(|e| e == "prisme")
    {
        let path = p.clone();
        let pending = prisme_platform::spawn(move || {
            let bytes = encoded.block().unwrap_or_else(|| Err("encoder crashed".into()))?;
            prisme_platform::write_file_path(&path, &bytes)?;
            Ok(Some(path.display().to_string()))
        });
        s.jobs.push(Job::Save { pending, doc: doc_id, revision: rev });
        return true;
    }
    let _ = save_as;
    let name = format!("{}.prisme", d.doc().info.name.trim_end_matches(".prisme"));
    // The dialog opens now (inside the user gesture on the web); the file is written once
    // the bytes are ready.
    let pending = prisme_platform::save_file_deferred("Save project", &name, &[FileFilter::new("Prisme project", &["prisme"])], encoded);
    s.jobs.push(Job::Save { pending, doc: doc_id, revision: rev });
    true
}

/// A document finished saving: close it if that was asked for ("Save" in the close prompt).
fn saved(s: &mut AppState, doc: u64) {
    if s.dialogs.close_after_save.contains(&doc) {
        s.dialogs.close_after_save.retain(|&d| d != doc);
        if let Some(i) = s.docs.iter().position(|d| d.id == doc) {
            crate::commands::close_doc(s, i);
        }
    }
}

/// A save was cancelled or failed: drop any pending close of that document, and abort quitting.
fn save_failed(s: &mut AppState, doc: u64) {
    s.dialogs.close_after_save.retain(|&d| d != doc);
    if std::mem::take(&mut s.dialogs.quit_after_save) {
        s.toast("Quit cancelled: a document wasn't saved");
    }
}

pub fn saving(s: &AppState) -> bool {
    s.jobs.iter().any(|j| matches!(j, Job::Save { doc, .. } if *doc != u64::MAX))
}

/// Drives "Save all and quit": saves dirty documents one at a time (one save dialog open at
/// once), then closes the window.
pub fn quit_flow(s: &mut AppState, ctx: &egui::Context) {
    // A stroke landing or an open Liquify session is unsaved work too.
    prisme_platform::set_unsaved_changes(s.docs.iter().any(|d| d.is_dirty() || d.stroke.is_some()));
    if !s.dialogs.quit_after_save || saving(s) {
        return;
    }
    match s.docs.iter().position(|d| d.is_dirty()) {
        Some(i) => {
            if !save_doc(s, i, false) {
                s.dialogs.quit_after_save = false;
            }
        }
        None => {
            s.dialogs.quit_after_save = false;
            s.dialogs.allow_quit = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}

/// Forget every autosave (clean exit).
pub fn clear_autosaves(s: &AppState) {
    for d in &s.docs {
        prisme_platform::storage_delete(&format!("autosave/{}", d.id));
    }
    prisme_platform::storage_delete(INDEX_KEY);
}

/// Forget autosaves of documents the user chose to discard.
pub fn discard_autosaves(s: &AppState) {
    for d in s.docs.iter().filter(|d| d.is_dirty()) {
        prisme_platform::storage_delete(&format!("autosave/{}", d.id));
    }
    prisme_platform::storage_delete(INDEX_KEY);
}

pub fn pick_screen(s: &mut AppState, to_bg: bool) {
    if !prisme_platform::screen_picker_supported() {
        s.toast("Screen colour picking isn't supported on this platform/browser");
        return;
    }
    s.jobs.push(Job::ScreenPick { pending: prisme_platform::pick_screen_color(), to_bg });
}

pub fn load_font(s: &mut AppState) {
    let f = vec![FileFilter::new("Fonts", &["ttf", "otf", "ttc", "otc"])];
    s.jobs.push(Job::FontFiles(prisme_platform::open_files("Load font", &f, true)));
}

pub fn load_lut(s: &mut AppState, node: NodeId, key: String) {
    let Some(d) = s.doc() else { return };
    let f = vec![FileFilter::new("3D LUT", &["cube"])];
    s.jobs.push(Job::LutFile { pending: prisme_platform::open_files("Load LUT", &f, false), doc: d.id, node, key });
}

// ------------------------------------------------------------------ export rendering

/// What a [`RenderJob`] is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderPurpose {
    /// Pixels for the export dialog (encoded right away when `then_encode`).
    Export { then_encode: bool },
    /// Edit ▸ Copy Merged.
    CopyMerged,
}

/// Renders the graph output over `rect` at full resolution (f32) tile by tile with async
/// readbacks (works the same on the web); `mask` (the selection) multiplies the alpha.
pub struct RenderJob {
    pub doc: Document,
    /// Document-space region; `pixels` are its straight-alpha RGBA rows once done.
    pub rect: IRect,
    pub mask: Option<Arc<prisme_core::TileGrid>>,
    pub tiles: Vec<IRect>,
    pub next: usize,
    pub pending: Option<(IRect, Readback)>,
    pub pixels: Vec<f32>,
    pub purpose: RenderPurpose,
}

/// Largest region rendered in one go (f32 RGBA: 16 bytes/px).
const MAX_RENDER_PIXELS: i64 = 1 << 28;

/// Region rendered for export / Copy Merged: the selection's bounds and mask when `selection`
/// and there is one, else the page (Fixed) or the content bounds (Infinite).
pub fn render_region(s: &AppState, doc: &Document, selection: bool) -> (IRect, Option<Arc<prisme_core::TileGrid>>) {
    match &doc.selection {
        Some(sel) if selection => {
            let b = sel.content_bounds();
            (if doc.info.is_infinite() { b } else { b.intersect(&doc.bounds()) }, Some(sel.clone()))
        }
        _ => (crate::commands::doc_extent(s, doc), None),
    }
}

/// Start rendering the active document's region (see [`render_region`]) for `purpose`.
pub fn start_render(s: &mut AppState, selection: bool, purpose: RenderPurpose) {
    let Some(d) = s.doc() else { return };
    let doc = d.doc().clone();
    let (rect, mask) = render_region(s, &doc, selection);
    if rect.is_empty() {
        s.toast("Nothing to render: the selection is empty");
        return;
    }
    if rect.area() > MAX_RENDER_PIXELS {
        s.toast(format!("{} × {} px is too large to render at once: select a smaller region", rect.width(), rect.height()));
        return;
    }
    let ts = 2048;
    let mut tiles = vec![];
    for y in (rect.y0..rect.y1).step_by(ts as usize) {
        for x in (rect.x0..rect.x1).step_by(ts as usize) {
            tiles.push(IRect::new(x, y, (x + ts).min(rect.x1), (y + ts).min(rect.y1)));
        }
    }
    s.jobs.retain(|j| !matches!(j, Job::Render(r) if r.purpose == purpose || matches!((r.purpose, purpose), (RenderPurpose::Export { .. }, RenderPurpose::Export { .. }))));
    let pixels = vec![0.0; rect.area() as usize * 4];
    s.jobs.push(Job::Render(RenderJob { pixels, doc, rect, mask, tiles, next: 0, pending: None, purpose }));
}

fn step_render(s: &mut AppState, r: &mut RenderJob) -> bool {
    let Some(g) = s.gpu.as_mut() else { return true };
    if let Some((rect, rb)) = &r.pending {
        let Some(data) = rb.try_take(&g.rs.device) else { return false };
        let w = r.rect.width() as usize;
        let rw = rect.width() as usize;
        for y in 0..rect.height() as usize {
            let dst = ((rect.y0 - r.rect.y0) as usize + y) * w * 4 + (rect.x0 - r.rect.x0) as usize * 4;
            if data.len() >= (y + 1) * rw * 4 {
                r.pixels[dst..dst + rw * 4].copy_from_slice(&data[y * rw * 4..(y + 1) * rw * 4]);
            }
        }
        r.pending = None;
    }
    if r.next >= r.tiles.len() {
        // Un-premultiply for the encoder, then apply the selection's alpha.
        for p in r.pixels.chunks_exact_mut(4) {
            if p[3] > 1e-7 {
                p[0] /= p[3];
                p[1] /= p[3];
                p[2] /= p[3];
            }
        }
        if let Some(m) = &r.mask {
            let a = m.read_rect(&r.rect);
            for (p, a) in r.pixels.chunks_exact_mut(4).zip(a) {
                p[3] *= a.clamp(0.0, 1.0);
            }
        }
        return true;
    }
    let rect = r.tiles[r.next];
    r.next += 1;
    let source = if r.doc.graph.is_trivial() { RenderSource::LayerStack } else { RenderSource::Graph };
    if let Some(img) = g.engine.evaluate(&EvalRequest { doc: &r.doc, rect, mip: 0, format: EvalFormat::F32, source }) {
        r.pending = Some((rect, Readback::start(&g.rs.device, &g.rs.queue, &img, rect)));
    }
    false
}

pub fn start_encode(s: &mut AppState, pixels: Arc<Vec<f32>>, w: u32, h: u32) {
    let opts = s.settings.export.clone();
    let name = s.doc().map(|d| d.doc().info.name.clone()).unwrap_or("image".into());
    let task = prisme_platform::spawn_worker(EncodePngJob { pixels, width: w, height: h, opts });
    s.jobs.push(Job::Encode { pending: task.pending, control: task.control, name });
}

/// Cancel a running PNG export encode.
pub fn cancel_encode(s: &mut AppState) {
    for j in &s.jobs {
        if let Job::Encode { control, .. } = j {
            control.cancel();
        }
    }
}

pub fn start_estimate(s: &mut AppState, pixels: Arc<Vec<f32>>, w: u32, h: u32) {
    let opts = s.settings.export.clone();
    s.jobs.retain(|j| !matches!(j, Job::Estimate(_)));
    s.jobs.push(Job::Estimate(prisme_platform::spawn(move || prisme_export::estimate_size(&pixels, w, h, &opts))));
}

// ------------------------------------------------------------------ polling

pub fn poll(s: &mut AppState, ctx: &egui::Context) {
    // Displayed-colour picker readback.
    let pick = s.gpu.as_mut().and_then(|g| {
        let (rb, to_bg) = g.pick.as_ref()?;
        let v = rb.try_take(&g.rs.device)?;
        let to_bg = *to_bg;
        g.pick = None;
        Some((v, to_bg))
    });
    if let Some((v, to_bg)) = pick
        && v.len() >= 4 && v[3] > 0.0 {
            crate::canvas::set_color(s, [v[0] / v[3], v[1] / v[3], v[2] / v[3], 1.0], to_bg);
        }

    let jobs = std::mem::take(&mut s.jobs);
    let mut keep = vec![];
    for job in jobs {
        match job {
            Job::OpenDialog { mut pending, place } => match pending.poll() {
                Some(files) => {
                    for f in files {
                        let into = if place { s.doc().map(|d| d.id) } else { None };
                        decode_file(s, f.name, f.bytes, into, f.path);
                    }
                }
                None if !pending.is_abandoned() => keep.push(Job::OpenDialog { pending, place }),
                None => {}
            },
            Job::Decode { name, mut pending, into, placeholder, path } => match pending.poll() {
                Some(Ok(loaded)) => {
                    let t = prisme_platform::now_seconds() - s.start_time;
                    match loaded {
                        Loaded::Doc(doc) => replace_or_add(s, placeholder, DocState::new(doc), path),
                        Loaded::History(h) => replace_or_add(s, placeholder, DocState::from_history(h), path),
                        Loaded::Place(prepared) => {
                            if let Some(d) = s.docs.iter_mut().find(|d| Some(d.id) == into) {
                                d.edit("Place", |doc| {
                                    let ids = prisme_io::add_prepared_layers(doc, prepared);
                                    if let Some(l) = ids.last() {
                                        doc.active_layer = Some(*l);
                                    }
                                });
                            }
                        }
                    }
                    log::info!("opened {name} ({t:.2}s since start)");
                    s.toast(format!("Opened {name}"));
                }
                Some(Err(e)) => {
                    if let Some(p) = placeholder {
                        s.docs.retain(|d| d.id != p);
                        s.active = s.active.min(s.docs.len().saturating_sub(1));
                    }
                    s.toast(format!("Couldn't open {name}: {e}"));
                }
                None if pending.is_abandoned() => s.toast(format!("Decoder crashed on {name}")),
                None => keep.push(Job::Decode { name, pending, into, placeholder, path }),
            },
            Job::Save { mut pending, doc, revision } => match pending.poll() {
                Some(Ok(Some(n))) => {
                    if let Some(d) = s.docs.iter_mut().find(|d| d.id == doc) {
                        d.saved_revision = revision;
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            d.path = Some(std::path::PathBuf::from(&n));
                        }
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    if doc != u64::MAX {
                        add_recent(s, std::path::Path::new(&n));
                    }
                    s.toast(format!("Saved {n}"));
                    saved(s, doc);
                }
                Some(Ok(None)) => save_failed(s, doc),
                Some(Err(e)) => {
                    s.toast(format!("Save failed: {e}"));
                    save_failed(s, doc);
                }
                None => keep.push(Job::Save { pending, doc, revision }),
            },
            Job::ClipboardSet(mut p) => {
                if p.poll().is_none() && !p.is_done() {
                    keep.push(Job::ClipboardSet(p));
                }
            }
            Job::ClipboardGet { mut pending, new_doc } => match pending.poll() {
                Some(img) => {
                    // Prefer our own full-precision clipboard when the system one matches it.
                    let internal = s.clipboard.clone().filter(|(_, b)| img.as_ref().is_none_or(|i| i.width == b.width() as u32 && i.height == b.height() as u32));
                    if let Some((g, b)) = internal {
                        crate::commands::paste_grid(s, g, b, new_doc);
                    } else if let Some(i) = img {
                        let px: Vec<f32> = i.rgba.chunks(4).flat_map(|p| {
                            let c = prisme_color::srgb().to_working([p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0]);
                            [c[0], c[1], c[2], p[3] as f32 / 255.0]
                        }).collect();
                        let g = prisme_core::TileGrid::from_dense(prisme_core::PixelFormat::F16, 4, i.width, i.height, &px);
                        crate::commands::paste_grid(s, g, IRect::from_size(i.width, i.height), new_doc);
                    } else {
                        s.toast("Clipboard has no image");
                    }
                }
                None if pending.is_abandoned() => {}
                None => keep.push(Job::ClipboardGet { pending, new_doc }),
            },
            Job::ScreenPick { mut pending, to_bg } => match pending.poll() {
                Some(Some(c)) => {
                    let w = prisme_color::srgb().to_working([c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0]);
                    crate::canvas::set_color(s, [w[0], w[1], w[2], 1.0], to_bg);
                }
                Some(None) => {}
                None if pending.is_abandoned() => {}
                None => keep.push(Job::ScreenPick { pending, to_bg }),
            },
            Job::FontFiles(mut p) => match p.poll() {
                Some(files) => {
                    let mut added = vec![];
                    for f in files {
                        added.extend(s.fonts.add_font_data(f.bytes));
                    }
                    s.toast(if added.is_empty() { "No usable fonts in file".to_string() } else { format!("Loaded fonts: {}", added.join(", ")) });
                }
                None if !p.is_abandoned() => keep.push(Job::FontFiles(p)),
                None => {}
            },
            Job::LutFile { mut pending, doc, node, key } => match pending.poll() {
                Some(files) => {
                    if let Some(f) = files.into_iter().next() {
                        let text = String::from_utf8_lossy(&f.bytes).to_string();
                        if prisme_graph::parse_cube(&text).is_none() {
                            s.toast("Not a valid 3D .cube LUT");
                        } else if let Some(d) = s.docs.iter_mut().find(|d| d.id == doc) {
                            d.edit("Load LUT", |doc| {
                                if let Some(n) = doc.graph.nodes.get_mut(&node) {
                                    n.params.insert(key.clone(), prisme_core::ParamValue::Text(text));
                                }
                            });
                        }
                    }
                }
                None if !pending.is_abandoned() => keep.push(Job::LutFile { pending, doc, node, key }),
                None => {}
            },
            Job::Render(mut r) => {
                if step_render(s, &mut r) {
                    let (w, h) = (r.rect.width() as u32, r.rect.height() as u32);
                    match r.purpose {
                        RenderPurpose::CopyMerged => {
                            let mut g = prisme_core::TileGrid::from_dense(prisme_core::PixelFormat::F32, 4, w, h, &r.pixels).translated(r.rect.x0, r.rect.y0);
                            g.remove_clear_tiles();
                            crate::commands::set_clipboard(s, g, r.rect);
                        }
                        RenderPurpose::Export { then_encode } => {
                            let px = Arc::new(std::mem::take(&mut r.pixels));
                            s.dialogs.export.pixels = Some((px.clone(), w, h));
                            start_estimate(s, px.clone(), w, h);
                            if then_encode {
                                start_encode(s, px, w, h);
                            }
                        }
                    }
                } else {
                    keep.push(Job::Render(r));
                }
                ctx.request_repaint();
            }
            Job::Encode { mut pending, control, name } => match pending.poll() {
                Some(Ok(res)) => {
                    s.dialogs.export.last_result = Some(format!(
                        "{} × {} · {} · {:.1} KB{}",
                        res.width,
                        res.height,
                        match res.palette_colors { Some(n) => format!("{n} colours"), None => format!("{}-bit {:?}", res.bit_depth, res.color_mode) },
                        res.bytes.len() as f64 / 1024.0,
                        match res.met_target { Some(false) => " (target not met)", _ => "" }
                    ));
                    s.dialogs.export.notes = res.notes.clone();
                    let file = format!("{}.png", name.trim_end_matches(".png"));
                    let p = prisme_platform::save_file("Export PNG", &file, &[FileFilter::new("PNG", &["png"])], res.bytes);
                    keep.push(Job::Save { pending: p, doc: u64::MAX, revision: 0 });
                }
                Some(Err(_)) if control.is_cancelled() => s.toast("Export cancelled"),
                Some(Err(e)) => s.toast(format!("Export failed: {e}")),
                None if pending.is_abandoned() && control.is_cancelled() => s.toast("Export cancelled"),
                None if pending.is_abandoned() => s.toast("Export crashed"),
                None => keep.push(Job::Encode { pending, control, name }),
            },
            Job::Estimate(mut p) => match p.poll() {
                Some(v) => s.dialogs.export.estimate = Some(v),
                None if !p.is_abandoned() => keep.push(Job::Estimate(p)),
                None => {}
            },
            Job::Autosave(mut p) => {
                if p.poll().is_none() && !p.is_done() {
                    keep.push(Job::Autosave(p));
                }
            }
            Job::Dropped { mut pending, name, into } => match pending.poll() {
                Some(Ok(bytes)) => decode_file(s, name, bytes, into, None),
                Some(Err(e)) => s.toast(format!("Couldn't read {name}: {e}")),
                None if !pending.is_abandoned() => keep.push(Job::Dropped { pending, name, into }),
                None => {}
            },
            Job::Recover(mut p) => match p.poll() {
                Some(Some(bytes)) => {
                    let job = DecodeJob { name: "autosave.prisme".into(), bytes, place_into: None };
                    keep.push(Job::RecoverLoad(prisme_platform::spawn_worker(job).pending));
                }
                Some(None) => {}
                None if !p.is_abandoned() => keep.push(Job::Recover(p)),
                None => {}
            },
            Job::RecoverLoad(mut p) => match p.poll() {
                Some(Ok(Loaded::History(h))) => {
                    let mut d = DocState::from_history(h);
                    d.saved_revision = u64::MAX; // mark dirty
                    s.add_doc(d);
                    s.toast("Recovered document");
                }
                Some(Ok(_)) => s.toast("Recovery failed: not a project"),
                Some(Err(e)) => s.toast(format!("Recovery failed: {e}")),
                None if p.is_abandoned() => s.toast("Recovery failed: loader crashed"),
                None => keep.push(Job::RecoverLoad(p)),
            },
        }
    }
    keep.append(&mut s.jobs);
    s.jobs = keep;
}

const MAX_RECENT: usize = 12;

/// Remember a file in the Open Recent list.
pub fn add_recent(s: &mut AppState, path: &std::path::Path) {
    let p = path.to_string_lossy().to_string();
    let r = &mut s.settings.recent_files;
    r.retain(|x| *x != p);
    r.insert(0, p);
    r.truncate(MAX_RECENT);
}

/// Reopen a file from the recent list (native only; the web has no stable file paths).
pub fn open_recent(s: &mut AppState, path: &str) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let p = std::path::PathBuf::from(path);
        match std::fs::read(&p) {
            Ok(bytes) => {
                let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                decode_file(s, name, bytes, None, Some(p));
            }
            Err(e) => {
                s.settings.recent_files.retain(|x| x != path);
                s.toast(format!("Couldn't open {path}: {e}"));
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = path;
        s.toast("Open Recent isn't available on the web");
    }
}

fn replace_or_add(s: &mut AppState, placeholder: Option<u64>, mut d: DocState, path: Option<std::path::PathBuf>) {
    if let Some(p) = &path {
        add_recent(s, p);
    }
    d.path = path;
    if let Some(i) = placeholder.and_then(|p| s.docs.iter().position(|x| x.id == p)) {
        s.docs[i] = d;
        crate::shortcuts::unfocus(s);
        s.active = i;
    } else {
        s.add_doc(d);
    }
}

// ------------------------------------------------------------------ autosave / recovery

const INDEX_KEY: &str = "autosave/index";

pub fn autosave(s: &mut AppState) {
    let now = prisme_platform::now_seconds();
    let interval = s.settings.autosave_minutes as f64 * 60.0;
    if interval <= 0.0 || now - s.last_autosave < interval || s.jobs.iter().any(|j| matches!(j, Job::Autosave(_))) {
        return;
    }
    s.last_autosave = now;
    let mut index = vec![];
    for d in &s.docs {
        if !d.is_dirty() {
            continue;
        }
        index.push((d.id, d.doc().info.name.clone()));
        let key = format!("autosave/{}", d.id);
        let encoded = prisme_platform::spawn_worker(SaveJob { history: d.history.clone(), include_history: false }).pending;
        let pending = prisme_platform::spawn_async(async move {
            let bytes = encoded.wait().await.unwrap_or_else(|| Err("encoder crashed".into()))?;
            #[cfg(not(target_arch = "wasm32"))]
            {
                prisme_platform::storage_write(&key, &bytes)
            }
            #[cfg(target_arch = "wasm32")]
            {
                prisme_platform::storage_write_large(&key, bytes).wait().await.unwrap_or(Ok(()))
            }
        });
        s.jobs.push(Job::Autosave(pending));
    }
    if let Ok(j) = serde_json::to_vec(&index) {
        let _ = prisme_platform::storage_write(INDEX_KEY, &j);
    }
}

pub fn check_recovery(s: &mut AppState) {
    let Some(j) = prisme_platform::storage_read(INDEX_KEY) else { return };
    let index: Vec<(u64, String)> = serde_json::from_slice(&j).unwrap_or_default();
    if !index.is_empty() {
        s.dialogs.recovery = index;
    }
}

pub fn recover(s: &mut AppState, id: u64) {
    let key = format!("autosave/{id}");
    #[cfg(not(target_arch = "wasm32"))]
    s.jobs.push(Job::Recover(Pending::ready(prisme_platform::storage_read(&key))));
    #[cfg(target_arch = "wasm32")]
    s.jobs.push(Job::Recover(prisme_platform::storage_read_large(&key)));
}

pub fn discard_recovery(s: &mut AppState) {
    for (id, _) in std::mem::take(&mut s.dialogs.recovery) {
        prisme_platform::storage_delete(&format!("autosave/{id}"));
    }
    prisme_platform::storage_delete(INDEX_KEY);
}
