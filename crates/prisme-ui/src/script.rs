//! Scripted automation for end-to-end tests and screenshots (native).
//!
//! `PRISME_SCRIPT="cmd;cmd;..."`, one command per frame:
//! `new:W,H` · `new-infinite:W,H[,0]` (infinite canvas, W×H initial frame, white background; 0: transparent) · `open:PATH` · `tool:<ToolKind debug name>` · `drag:x0,y0,x1,y1` (document px) ·
//! `stroke:x0,y0,x1,y1,…` (polyline drag, in one frame) · `paint:FRAMES,x0,y0,x1,y1,…` (polyline drag over FRAMES frames) · `brush:<preset name>` · `size:N` (brush px) ·
//! `click:x,y` · `altclick:x,y` (clone source) · `key:Enter|Escape` · `type:text` · `node:<kind>` · `param:key=value` (selected node) ·
//! `fg:#rrggbb` · `wait:N` (frames) · `shot:PATH` (window screenshot PNG) · `export:PATH` (PNG at full res) ·
//! `export-sel:PATH` (Export Selection) · `action:<Action debug name>` (e.g. `action:CopyMerged`) ·
//! `clip:PATH` (internal clipboard as PNG, export settings) ·
//! `save:PATH` (.prisme) · `undo` · `redo` · `toggle-nodes` · `hdr:auto|on|off` (persisted setting) ·
//! `hdr-status` (log surface format / colour space / headroom) · `liquify-mode:forward|bloat|pucker|twirl-cw|…`
//! (`LiquifyMode::id`) · `hold:x,y,SECONDS` (press and hold still, ticking the tool at 30 Hz: stationary
//! liquify modes, airbrush) · `sleep:SECONDS` (block the UI thread: wall-clock pauses, e.g. to sample
//! memory while background work finishes) · `panel:<title>` (show a panel, e.g. `panel:Performance`) · `perf-limit:MS` (slow-frame limit) · `perf-report` (log the Performance panel's report) · `quit`.

use crate::app::AppState;
use prisme_tools::{Phase, ToolInput, ToolKey, ToolKind};

#[derive(Default)]
pub struct Script {
    pub cmds: std::collections::VecDeque<String>,
    pub wait: u32,
    pub pending_shot: Option<String>,
    pub export_path: Option<String>,
    /// `paint`: polyline, frames done, frames in all.
    pub painting: Option<(Vec<[f32; 2]>, usize, usize)>,
}

impl Script {
    pub fn from_env() -> Option<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let s = std::env::var("PRISME_SCRIPT").ok()?;
            Some(Script { cmds: s.split(';').map(|c| c.trim().to_string()).filter(|c| !c.is_empty()).collect(), wait: 20, ..Default::default() })
        }
        #[cfg(target_arch = "wasm32")]
        None
    }
}


fn nums(s: &str) -> Vec<f32> {
    s.split(',').filter_map(|v| v.trim().parse().ok()).collect()
}

fn ev(phase: Phase, pos: [f32; 2], t: f64) -> ToolInput {
    let mut e = ToolInput::mouse(phase, pos);
    e.time = t;
    e
}

/// One tool tick (what the canvas does every frame while the pointer is held).
fn tick(s: &mut AppState, time: f64) {
    let (fg, bg, opts, tool) = (s.settings.fg, s.settings.bg, s.settings.tools.clone(), s.tool);
    let (Some(t), Some(d), Some(g)) = (s.tools.get_mut(&tool), s.docs.get_mut(s.active), s.gpu.as_mut()) else { return };
    let Some(w) = d.working.as_mut() else { return };
    let mut ops = vec![];
    let mut cx = prisme_tools::ToolCx { doc: w, opts: &opts, fg, bg, strokes: Some(&mut ops), view: d.view_rect };
    t.tick(&mut cx, time);
    crate::stroke::feed(d, &mut g.engine, ops);
    d.touch();
}

pub fn step(s: &mut AppState, sc: &mut Script, ctx: &egui::Context) {
    // Screenshot results.
    let shots: Vec<std::sync::Arc<egui::ColorImage>> = ctx.input(|i| {
        i.events.iter().filter_map(|e| if let egui::Event::Screenshot { image, .. } = e { Some(image.clone()) } else { None }).collect()
    });
    for img in shots {
        if let Some(path) = sc.pending_shot.take() {
            let rgba: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
            let png = prisme_platform::encode_png(&prisme_platform::ClipboardImage { width: img.size[0] as u32, height: img.size[1] as u32, rgba });
            if let Ok(b) = png {
                let _ = prisme_platform::write_file_path(std::path::Path::new(&path), &b);
                log::info!("script: screenshot -> {path}");
            }
        }
    }
    // Export completion: write the file directly.
    if let Some(path) = &sc.export_path {
        if let Some((px, w, h)) = s.dialogs.export.pixels.clone() {
            let opts = s.settings.export.clone();
            match prisme_export::encode_png(&px, w, h, &opts, &mut |_, _| {}, &|| false) {
                Ok(r) => {
                    let _ = prisme_platform::write_file_path(std::path::Path::new(path), &r.bytes);
                    log::info!("script: exported {} bytes -> {path} ({}x{})", r.bytes.len(), r.width, r.height);
                }
                Err(e) => log::error!("script: export failed: {e}"),
            }
            sc.export_path = None;
            s.dialogs.export.pixels = None;
        } else {
            ctx.request_repaint();
            return;
        }
    }
    // A stroke drawn over several frames: one pointer move per frame.
    if let Some((pts, k, n)) = &mut sc.painting {
        *k += 1;
        let f = *k as f32 / *n as f32 * (pts.len() - 1) as f32;
        let (i, u) = ((f.floor() as usize).min(pts.len() - 2), f - f.floor().min((pts.len() - 2) as f32));
        let p = [pts[i][0] + (pts[i + 1][0] - pts[i][0]) * u, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * u];
        let t = prisme_platform::now_seconds();
        let done = *k >= *n;
        crate::canvas::run_tool(s, ev(if done { Phase::Up } else { Phase::Move }, p, t));
        if done {
            sc.painting = None;
        }
        ctx.request_repaint();
        return;
    }
    // Strokes land asynchronously (GPU readback): let them finish before the next command.
    if sc.pending_shot.is_some() || !s.jobs.is_empty() || s.docs.iter().any(|d| d.stroke_pending()) {
        ctx.request_repaint();
        return;
    }
    if sc.wait > 0 {
        sc.wait -= 1;
        ctx.request_repaint();
        return;
    }
    let Some(cmd) = sc.cmds.pop_front() else { return };
    ctx.request_repaint();
    log::info!("script: {cmd}");
    let (name, arg) = cmd.split_once(':').unwrap_or((cmd.as_str(), ""));
    let t = prisme_platform::now_seconds();
    match name {
        "new" | "new-infinite" => {
            let n = nums(arg);
            let (w, h) = (n[0] as u32, n[1] as u32);
            let d = if name == "new" { prisme_core::Document::with_background("Script", w, h, Some([1.0; 4])) } else { prisme_core::Document::infinite("Script", w, h, (n.get(2) != Some(&0.0)).then_some([1.0; 4])) };
            s.add_doc(crate::doc::DocState::new(d));
        }
        "action" => match crate::actions::Action::all().into_iter().find(|a| format!("{a:?}").eq_ignore_ascii_case(arg)) {
            Some(a) => crate::commands::run_action(s, a, ctx),
            None => log::warn!("script: unknown action {arg}"),
        },
        "clip" => match s.clipboard.clone() {
            Some((g, b)) => {
                let px = g.read_rect(&b);
                match prisme_export::encode_png(&px, b.width() as u32, b.height() as u32, &s.settings.export, &mut |_, _| {}, &|| false) {
                    Ok(r) => {
                        let _ = prisme_platform::write_file_path(std::path::Path::new(arg), &r.bytes);
                        log::info!("script: clipboard {}x{} at {},{} -> {arg}", b.width(), b.height(), b.x0, b.y0);
                    }
                    Err(e) => log::error!("script: clipboard encode failed: {e}"),
                }
            }
            None => log::warn!("script: clipboard is empty"),
        },
        "open" => {
            if let Ok(b) = std::fs::read(arg) {
                crate::jobs::decode_file(s, std::path::Path::new(arg).file_name().unwrap().to_string_lossy().into(), b, None, None);
            }
        }
        "tool" => {
            if let Some(k) = ToolKind::ALL.iter().find(|k| format!("{k:?}").eq_ignore_ascii_case(arg)) {
                s.set_tool(*k);
            }
        }
        "drag" => {
            let n = nums(arg);
            let steps = 24;
            crate::canvas::run_tool(s, ev(Phase::Down, [n[0], n[1]], t));
            for i in 1..=steps {
                let f = i as f32 / steps as f32;
                crate::canvas::run_tool(s, ev(Phase::Move, [n[0] + (n[2] - n[0]) * f, n[1] + (n[3] - n[1]) * f], t + i as f64 * 0.01));
            }
            crate::canvas::run_tool(s, ev(Phase::Up, [n[2], n[3]], t + 0.3));
        }
        "stroke" => {
            let n = nums(arg);
            let pts: Vec<[f32; 2]> = n.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
            if let Some(first) = pts.first() {
                let mut i = 0;
                crate::canvas::run_tool(s, ev(Phase::Down, *first, t));
                for w in pts.windows(2) {
                    for k in 1..=12 {
                        let f = k as f32 / 12.0;
                        i += 1;
                        crate::canvas::run_tool(s, ev(Phase::Move, [w[0][0] + (w[1][0] - w[0][0]) * f, w[0][1] + (w[1][1] - w[0][1]) * f], t + i as f64 * 0.01));
                    }
                }
                crate::canvas::run_tool(s, ev(Phase::Up, *pts.last().unwrap(), t + i as f64 * 0.01 + 0.01));
            }
        }
        "paint" => {
            let n = nums(arg);
            let pts: Vec<[f32; 2]> = n[1..].chunks_exact(2).map(|p| [p[0], p[1]]).collect();
            if pts.len() >= 2 {
                crate::canvas::run_tool(s, ev(Phase::Down, pts[0], t));
                sc.painting = Some((pts, 0, (n[0] as usize).max(1)));
            }
        }
        "brush" => {
            if let Some(p) = prisme_brush::presets().into_iter().find(|p| p.name.eq_ignore_ascii_case(arg)) {
                s.settings.tools.brush = p.settings;
            }
        }
        "size" => {
            if let Ok(v) = arg.parse::<f32>() {
                s.settings.tools.brush.size = v;
                s.settings.tools.eraser.size = v;
                s.settings.tools.clone_brush.size = v;
                s.settings.tools.retouch_brush.size = v;
                s.settings.tools.liquify.size = v;
            }
        }
        "liquify-mode" => match prisme_tools::LiquifyMode::ALL.iter().find(|m| m.id() == arg) {
            Some(m) => s.settings.tools.liquify.mode = *m,
            None => log::warn!("script: unknown liquify mode {arg}"),
        },
        "hold" => {
            let n = nums(arg);
            crate::canvas::run_tool(s, ev(Phase::Down, [n[0], n[1]], t));
            let ticks = (n.get(2).copied().unwrap_or(1.0) * 30.0) as usize;
            for i in 1..=ticks {
                tick(s, t + i as f64 / 30.0);
            }
            crate::canvas::run_tool(s, ev(Phase::Up, [n[0], n[1]], t + ticks as f64 / 30.0));
        }
        "altclick" => {
            let n = nums(arg);
            let mut e = ev(Phase::Down, [n[0], n[1]], t);
            e.mods.alt = true;
            crate::canvas::run_tool(s, e);
            e.phase = Phase::Up;
            crate::canvas::run_tool(s, e);
        }
        "click" => {
            let n = nums(arg);
            crate::canvas::run_tool(s, ev(Phase::Down, [n[0], n[1]], t));
            crate::canvas::run_tool(s, ev(Phase::Up, [n[0], n[1]], t + 0.05));
        }
        "key" => {
            let k = match arg {
                "Enter" => ToolKey::Enter,
                _ => ToolKey::Escape,
            };
            crate::commands::script_key(s, &k);
        }
        "type" => crate::commands::script_key(s, &ToolKey::Text(arg.replace("\\n", "\n"))),
        "node" => {
            if let Some(d) = s.doc_mut() {
                crate::panels::nodes::insert_before_output(d, arg);
            }
        }
        "param" => {
            if let (Some((k, v)), Some(d)) = (arg.split_once('='), s.doc_mut())
                && let Some(n) = d.selected_node {
                    let val = if let Ok(f) = v.parse::<f32>() { prisme_core::ParamValue::Float(f) } else { prisme_core::ParamValue::Bool(v == "true") };
                    let k = k.to_string();
                    d.edit("Script Param", |doc| {
                        if let Some(node) = doc.graph.nodes.get_mut(&n) {
                            node.params.insert(k, val);
                        }
                    });
                }
        }
        "text3d" | "textfx" => {
            let three = name == "text3d";
            if let Some(d) = s.doc_mut() {
                d.edit("Script Text FX", |doc| {
                    if let Some(prisme_core::LayerKind::Text(t)) = doc.active_mut().map(|l| &mut l.kind) {
                        t.size = 140.0;
                        t.fill = prisme_core::layer::TextFill::Gradient {
                            gradient: prisme_core::Gradient::two([1.0, 0.6, 0.05, 1.0], [0.9, 0.05, 0.3, 1.0]),
                            angle: 0.0,
                        };
                        t.outline = Some(prisme_core::layer::TextOutline { width: 4.0, color: [1.0, 1.0, 1.0, 1.0] });
                        t.glow = Some(prisme_core::layer::TextGlow { radius: 24.0, color: [1.0, 0.8, 0.2, 1.0], strength: 1.2, inner: false });
                        if three {
                            t.three_d.enabled = true;
                            t.three_d.depth = 40.0;
                            t.three_d.radius = 700.0;
                            t.three_d.rotation = [0.3, 0.25, 0.0];
                        }
                    }
                });
            }
        }
        "fg" => {
            if let Some(c) = prisme_color::hex_to_working(arg) {
                s.settings.fg = c;
            }
        }
        "wait" => sc.wait = arg.parse().unwrap_or(10),
        "sleep" => {
            #[cfg(not(target_arch = "wasm32"))]
            std::thread::sleep(std::time::Duration::from_secs_f64(arg.parse().unwrap_or(1.0)));
        }
        "shot" => {
            sc.pending_shot = Some(arg.to_string());
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
        }
        "export" | "export-sel" => {
            s.dialogs.export.pixels = None;
            s.dialogs.export.selection = name == "export-sel";
            crate::dialogs::start_export_render(s);
            sc.export_path = Some(arg.to_string());
        }
        "save" => {
            if let Some(d) = s.doc()
                && let Ok(b) = prisme_io::container::save(&d.history, true) {
                    let _ = prisme_platform::write_file_path(std::path::Path::new(arg), &b);
                }
        }
        "undo" => crate::commands::run_action(s, crate::actions::Action::Undo, ctx),
        "redo" => crate::commands::run_action(s, crate::actions::Action::Redo, ctx),
        "toggle-nodes" => crate::commands::run_action(s, crate::actions::Action::ToggleNodeOutput, ctx),
        "hdr" => {
            s.settings.hdr_mode = match arg.to_ascii_lowercase().as_str() {
                "on" => crate::hdr::HdrMode::On,
                "off" => crate::hdr::HdrMode::Off,
                _ => crate::hdr::HdrMode::Auto,
            };
        }
        "panel" => s.open_tab = crate::app::Tab::ALL.into_iter().find(|t| t.title().eq_ignore_ascii_case(arg)),
        "perf-limit" => s.settings.perf_slow_ms = arg.parse().unwrap_or(s.settings.perf_slow_ms),
        "perf-report" => log::info!("script: {}", crate::perf::report(s)),
        "hdr-status" => log::info!("script: hdr-status {:?} (mode {:?}, highlights {:?})", s.hdr, s.settings.hdr_mode, s.settings.hdr_highlights),
        "shortcuts" => crate::shortcuts::open(s),
        "new-dialog" => {
            s.dialogs.new_doc.open = true;
            s.dialogs.new_doc.infinite = arg == "infinite";
        }
        "terminate" => prisme_platform::request_app_terminate(),
        "quit" => {
            s.dialogs.allow_quit = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        other => log::warn!("script: unknown command {other}"),
    }
    sc.wait = 3;
}
