//! Background jobs: opening/decoding, saving, export rendering/encoding, clipboard,
//! screen picking, font/LUT loading, autosave and crash recovery.

use crate::app::AppState;
use crate::doc::DocState;
use prisme_core::{Document, History, IRect, NodeId};
use prisme_gpu::{EvalFormat, EvalRequest, Readback, RenderSource};
use prisme_platform::{FileFilter, Pending, PickedFile};
use std::sync::Arc;

pub enum Loaded {
    Doc(Document),
    History(History),
    Place(prisme_io::Decoded),
}

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
    Encode { pending: Pending<Result<prisme_export::ExportResult, String>>, name: String },
    Estimate(Pending<u64>),
    Autosave(Pending<Result<(), String>>),
    Recover(Pending<Option<Vec<u8>>>),
    Dropped { pending: Pending<Result<Vec<u8>, String>>, name: String, into: Option<u64> },
}

pub fn describe(jobs: &[Job]) -> String {
    jobs.iter()
        .map(|j| match j {
            Job::Decode { name, .. } => format!("Opening {name}"),
            Job::Render(r) => format!("Rendering {:.0}%", r.next as f32 / r.tiles.len().max(1) as f32 * 100.0),
            Job::Encode { .. } => "Encoding PNG".into(),
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
    let n2 = name.clone();
    let pending = prisme_platform::spawn(move || -> Result<Loaded, String> {
        if project {
            return prisme_io::container::load(&bytes).map(Loaded::History).map_err(|e| e.to_string());
        }
        let decoded = prisme_io::decode(&bytes, &n2).map_err(|e| e.to_string())?;
        if into.is_some() {
            Ok(Loaded::Place(decoded))
        } else {
            Ok(Loaded::Doc(prisme_io::document_from_decoded(&n2, decoded)))
        }
    });
    s.jobs.push(Job::Decode { name, pending, into, placeholder, path });
}

pub fn save(s: &mut AppState, save_as: bool) {
    let Some(d) = s.doc() else { return };
    let preview = None;
    let bytes = match prisme_io::container::save_with_preview(&d.history, true, preview) {
        Ok(b) => b,
        Err(e) => {
            s.toast(format!("Save failed: {e}"));
            return;
        }
    };
    let (doc_id, rev) = (d.id, d.history.revision());
    #[cfg(not(target_arch = "wasm32"))]
    if !save_as
        && let Some(p) = &d.path
            && p.extension().is_some_and(|e| e == "prisme") {
                match prisme_platform::write_file_path(p, &bytes) {
                    Ok(()) => {
                        let msg = format!("Saved {}", p.display());
                        let path = p.clone();
                        add_recent(s, &path);
                        if let Some(d) = s.doc_mut() {
                            d.saved_revision = rev;
                        }
                        s.toast(msg);
                    }
                    Err(e) => s.toast(format!("Save failed: {e}")),
                }
                return;
            }
    let _ = save_as;
    let name = format!("{}.prisme", d.doc().info.name.trim_end_matches(".prisme"));
    let pending = prisme_platform::save_file("Save project", &name, &[FileFilter::new("Prisme project", &["prisme"])], bytes);
    s.jobs.push(Job::Save { pending, doc: doc_id, revision: rev });
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

/// Renders the graph output at full resolution (f32) tile by tile with async readbacks.
pub struct RenderJob {
    pub doc: Document,
    pub tiles: Vec<IRect>,
    pub next: usize,
    pub pending: Option<(IRect, Readback)>,
    pub pixels: Vec<f32>,
    pub then_encode: bool,
}

pub fn start_render(s: &mut AppState, then_encode: bool) {
    let Some(d) = s.doc() else { return };
    let doc = d.doc().clone();
    let (w, h) = (doc.info.width as i32, doc.info.height as i32);
    let ts = 2048;
    let mut tiles = vec![];
    for y in (0..h).step_by(ts as usize) {
        for x in (0..w).step_by(ts as usize) {
            tiles.push(IRect::new(x, y, (x + ts).min(w), (y + ts).min(h)));
        }
    }
    s.jobs.retain(|j| !matches!(j, Job::Render(_)));
    s.jobs.push(Job::Render(RenderJob { pixels: vec![0.0; (w * h * 4) as usize], doc, tiles, next: 0, pending: None, then_encode }));
}

fn step_render(s: &mut AppState, r: &mut RenderJob) -> bool {
    let Some(g) = s.gpu.as_mut() else { return true };
    if let Some((rect, rb)) = &r.pending {
        let Some(data) = rb.try_take(&g.rs.device) else { return false };
        let w = r.doc.info.width as usize;
        let rw = rect.width() as usize;
        for y in 0..rect.height() as usize {
            let dst = ((rect.y0 as usize + y) * w + rect.x0 as usize) * 4;
            if data.len() >= (y + 1) * rw * 4 {
                r.pixels[dst..dst + rw * 4].copy_from_slice(&data[y * rw * 4..(y + 1) * rw * 4]);
            }
        }
        r.pending = None;
    }
    if r.next >= r.tiles.len() {
        // Un-premultiply for the encoder.
        for p in r.pixels.chunks_exact_mut(4) {
            if p[3] > 1e-7 {
                p[0] /= p[3];
                p[1] /= p[3];
                p[2] /= p[3];
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
    let pending = prisme_platform::spawn(move || {
        prisme_export::encode_png(&pixels, w, h, &opts, &mut |_, _| {}, &|| false).map_err(|e| e.to_string())
    });
    s.jobs.push(Job::Encode { pending, name });
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
                        Loaded::Place(decoded) => {
                            if let Some(d) = s.docs.iter_mut().find(|d| Some(d.id) == into) {
                                d.edit("Place", |doc| {
                                    let ids = prisme_io::add_decoded_as_layers(doc, &name, decoded);
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
                }
                Some(Ok(None)) => {}
                Some(Err(e)) => s.toast(format!("Save failed: {e}")),
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
                    let (w, h) = (r.doc.info.width, r.doc.info.height);
                    let px = Arc::new(std::mem::take(&mut r.pixels));
                    s.dialogs.export.pixels = Some((px.clone(), w, h));
                    start_estimate(s, px.clone(), w, h);
                    if r.then_encode {
                        start_encode(s, px, w, h);
                    }
                } else {
                    keep.push(Job::Render(r));
                }
                ctx.request_repaint();
            }
            Job::Encode { mut pending, name } => match pending.poll() {
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
                Some(Err(e)) => s.toast(format!("Export failed: {e}")),
                None if pending.is_abandoned() => s.toast("Export crashed"),
                None => keep.push(Job::Encode { pending, name }),
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
                Some(Some(bytes)) => match prisme_io::container::load(&bytes) {
                    Ok(h) => {
                        let mut d = DocState::from_history(h);
                        d.saved_revision = u64::MAX; // mark dirty
                        s.add_doc(d);
                        s.toast("Recovered document");
                    }
                    Err(e) => s.toast(format!("Recovery failed: {e}")),
                },
                Some(None) => {}
                None if !p.is_abandoned() => keep.push(Job::Recover(p)),
                None => {}
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
        let h = d.history.clone();
        let key = format!("autosave/{}", d.id);
        let pending = prisme_platform::spawn(move || {
            let bytes = prisme_io::container::save(&h, false).map_err(|e| e.to_string())?;
            #[cfg(not(target_arch = "wasm32"))]
            {
                prisme_platform::storage_write(&key, &bytes)
            }
            #[cfg(target_arch = "wasm32")]
            {
                drop(prisme_platform::storage_write_large(&key, bytes));
                Ok(())
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
