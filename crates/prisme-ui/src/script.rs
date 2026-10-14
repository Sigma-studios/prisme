//! Scripted automation for end-to-end tests and screenshots (native).
//!
//! `PRISME_SCRIPT="cmd;cmd;..."`, one command per frame:
//! `new:W,H` · `open:PATH` · `tool:<ToolKind debug name>` · `drag:x0,y0,x1,y1` (document px) ·
//! `click:x,y` · `key:Enter|Escape` · `type:text` · `node:<kind>` · `param:key=value` (selected node) ·
//! `fg:#rrggbb` · `wait:N` (frames) · `shot:PATH` (window screenshot PNG) · `export:PATH` (PNG at full res) ·
//! `save:PATH` (.prisme) · `undo` · `redo` · `toggle-nodes` · `quit`.

use crate::app::AppState;
use prisme_tools::{Phase, ToolInput, ToolKey, ToolKind};

#[derive(Default)]
pub struct Script {
    pub cmds: std::collections::VecDeque<String>,
    pub wait: u32,
    pub pending_shot: Option<String>,
    pub export_path: Option<String>,
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
    if sc.pending_shot.is_some() || !s.jobs.is_empty() {
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
        "new" => {
            let n = nums(arg);
            let d = prisme_core::Document::with_background("Script", n[0] as u32, n[1] as u32, Some([1.0; 4]));
            s.add_doc(crate::doc::DocState::new(d));
        }
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
        "shot" => {
            sc.pending_shot = Some(arg.to_string());
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
        }
        "export" => {
            s.dialogs.export.pixels = None;
            crate::jobs::start_render(s, false);
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
        "quit" => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
        other => log::warn!("script: unknown command {other}"),
    }
    sc.wait = 3;
}
