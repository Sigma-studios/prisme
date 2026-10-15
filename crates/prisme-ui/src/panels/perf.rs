//! Performance panel: frame-time graph, slow frames, and what each frame spent its time on.

use crate::app::AppState;
use crate::perf::{sorted_spans, Frame};
use crate::theme;
use egui::{Color32, Pos2, Rect, RichText, Sense, Stroke, Ui, Vec2};

const UI_COL: Color32 = Color32::from_rgb(110, 140, 255);
const PAINT_COL: Color32 = Color32::from_rgb(170, 120, 230);
const WAIT_COL: Color32 = Color32::from_rgb(90, 92, 100);
const GPU_COL: Color32 = Color32::from_rgb(240, 160, 70);
const SLOW_COL: Color32 = Color32::from_rgb(235, 90, 80);

pub fn ui(s: &mut AppState, ui: &mut Ui) {
    let limit = s.settings.perf_slow_ms;
    ui.horizontal_wrapped(|ui| {
        let p = &mut s.perf;
        if ui.selectable_label(p.paused, format!("{} Pause", egui_phosphor::regular::PAUSE)).clicked() {
            p.paused = !p.paused;
        }
        if ui.button("Clear").clicked() {
            p.clear();
        }
        if ui.button(format!("{} Copy Report", egui_phosphor::regular::COPY)).on_hover_text("Copy the slow frames and the selected frame's breakdown as text").clicked() {
            let r = crate::perf::report(s);
            ui.ctx().copy_text(r);
            s.toast("Performance report copied");
        }
        ui.checkbox(&mut s.settings.perf_warn, "Warn on slow frames").on_hover_text("Show a status-bar message when a frame takes longer than the limit");
        ui.add(egui::DragValue::new(&mut s.settings.perf_slow_ms).range(8.0..=500.0).speed(1.0).suffix(" ms").prefix("limit "));
    });
    summary(s, ui);
    ui.add_space(4.0);
    graph(s, ui, limit);
    legend(ui);
    ui.separator();
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        slow_list(s, ui);
        let pick = s.perf.selected.or_else(|| s.perf.frames.back().map(|f| f.rec.index));
        if let Some(f) = pick.and_then(|i| s.perf.get(i)) {
            super::section(ui, &format!("Frame #{}", f.rec.index));
            details(ui, f);
        }
        memory(s, ui);
    });
}

fn summary(s: &AppState, ui: &mut Ui) {
    let now = prisme_core::perf::now();
    let recent: Vec<&Frame> = s.perf.frames.iter().rev().take_while(|f| now - f.rec.start < 3.0).filter(|f| f.active).collect();
    let txt = if recent.is_empty() {
        "Idle — draw or pan to measure".to_string()
    } else {
        let mut d: Vec<f32> = recent.iter().map(|f| f.duration_ms).collect();
        d.sort_by(f32::total_cmp);
        let q = |x: f32| d[((d.len() - 1) as f32 * x) as usize];
        format!("Last 3 s: {:.0} fps · median {:.1} ms · 95th {:.1} ms · worst {:.1} ms", 1000.0 / q(0.5).max(0.1), q(0.5), q(0.95), q(1.0))
    };
    ui.label(RichText::new(txt).monospace());
    if !s.perf.gpu_timing {
        ui.label(RichText::new("GPU timing unavailable on this device: GPU work only shows as wait").small().color(theme::TEXT_DIM));
    }
}

fn graph(s: &mut AppState, ui: &mut Ui, limit: f32) {
    let h = 110.0;
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), h), Sense::click());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, theme::BG0);
    let bar = 3.0;
    let n = ((rect.width() / bar) as usize).min(s.perf.frames.len());
    let frames: Vec<&Frame> = s.perf.frames.iter().skip(s.perf.frames.len() - n).collect();
    let top = frames.iter().map(|f| f.duration_ms.min(500.0)).fold(limit * 1.25, f32::max);
    let y = |ms: f32| rect.bottom() - (ms / top).min(1.0) * (h - 4.0);
    for ms in [1000.0 / 60.0, 1000.0 / 30.0, limit] {
        if ms < top {
            let col = if ms == limit { SLOW_COL.gamma_multiply(0.6) } else { theme::BG3 };
            painter.hline(rect.x_range(), y(ms), Stroke::new(1.0, col));
            painter.text(Pos2::new(rect.left() + 3.0, y(ms) - 1.0), egui::Align2::LEFT_BOTTOM, format!("{ms:.0}"), egui::FontId::monospace(9.0), theme::TEXT_DIM);
        }
    }
    let x0 = rect.right() - n as f32 * bar;
    let mut hovered = None;
    for (i, f) in frames.iter().enumerate() {
        let x = x0 + i as f32 * bar;
        let col = |c: Color32| if f.active { c } else { c.gamma_multiply(0.35) };
        let mut base = 0.0;
        for (ms, c) in [(f.ui_ms, UI_COL), (f.paint_ms(), PAINT_COL), (f.wait_ms(), WAIT_COL)] {
            let (a, b) = (y(base), y(base + ms));
            if a - b > 0.2 {
                painter.rect_filled(Rect::from_min_max(Pos2::new(x, b), Pos2::new(x + bar - 1.0, a)), 0.0, col(c));
            }
            base += ms;
        }
        if let Some(g) = f.gpu_ms() {
            painter.rect_filled(Rect::from_center_size(Pos2::new(x + bar / 2.0 - 0.5, y(g)), Vec2::new(bar, 2.0)), 0.0, GPU_COL);
        }
        if f.slow(limit) {
            painter.rect_filled(Rect::from_min_size(Pos2::new(x, rect.top()), Vec2::new(bar - 1.0, 3.0)), 0.0, SLOW_COL);
        }
        if s.perf.selected == Some(f.rec.index) {
            painter.vline(x + bar / 2.0, rect.y_range(), Stroke::new(1.0, theme::TEXT));
        }
        if resp.hover_pos().is_some_and(|p| p.x >= x && p.x < x + bar) {
            hovered = Some(f.rec.index);
        }
    }
    if let Some(i) = hovered {
        if resp.clicked() {
            s.perf.selected = Some(i);
        }
        if let Some(f) = s.perf.get(i) {
            resp.on_hover_ui_at_pointer(|ui| {
                ui.label(RichText::new(format!("#{} {:.1} ms — {}", f.rec.index, f.duration_ms, f.diagnosis())).strong());
                let mut t = String::new();
                crate::perf::frame_text(&mut t, f);
                ui.label(RichText::new(t).monospace().small());
            });
        }
    }
}

fn legend(ui: &mut Ui) {
    ui.horizontal_wrapped(|ui| {
        for (c, l, tip) in [
            (UI_COL, "UI", "Prisme's own work this frame (tools, evaluation setup, panels)"),
            (PAINT_COL, "painting", "egui tessellation and drawing of the window"),
            (WAIT_COL, "wait", "Waiting for the GPU to finish or for the display's refresh (vsync)"),
            (GPU_COL, "GPU", "GPU time of the engine's work issued this frame (evaluation, brush dabs)"),
            (SLOW_COL, "slow", "Over the limit"),
        ] {
            let (r, _) = ui.allocate_exact_size(Vec2::splat(9.0), Sense::hover());
            ui.painter().rect_filled(r, 2.0, c);
            ui.label(RichText::new(l).small()).on_hover_text(tip);
        }
        ui.label(RichText::new("· dim bars: idle frames · click a bar for details").small().color(theme::TEXT_DIM));
    });
}

fn slow_list(s: &mut AppState, ui: &mut Ui) {
    super::section(ui, &format!("Slow frames ({})", s.perf.slow.len()));
    if s.perf.slow.is_empty() {
        ui.label(RichText::new(format!("None over {:.0} ms", s.settings.perf_slow_ms)).color(theme::TEXT_DIM));
        return;
    }
    let now = prisme_core::perf::now();
    let mut pick = None;
    for i in s.perf.slow.iter().rev().take(12) {
        let Some(f) = s.perf.get(*i) else { continue };
        let txt = format!("{:>5.0} ms  {:>4.0} s ago  {}  {}", f.duration_ms, now - f.rec.start, f.tool, f.diagnosis());
        if ui.selectable_label(s.perf.selected == Some(*i), RichText::new(txt).monospace().small()).clicked() {
            pick = Some(*i);
        }
    }
    if pick.is_some() {
        s.perf.selected = pick;
    }
}

fn details(ui: &mut Ui, f: &Frame) {
    ui.label(RichText::new(format!(
        "{:.1} ms = UI {:.1} + painting {:.1} + wait {:.1}{}",
        f.duration_ms,
        f.ui_ms,
        f.paint_ms(),
        f.wait_ms(),
        if f.active { "" } else { "  (idle)" }
    )).monospace());
    for (label, ms) in &f.gpu {
        ui.label(RichText::new(format!("GPU {label}: {ms:.2} ms")).monospace().color(GPU_COL));
    }
    let total = f.ui_ms.max(0.1);
    egui::Grid::new(("perf-spans", f.rec.index)).num_columns(2).spacing([12.0, 1.0]).show(ui, |ui| {
        for sp in sorted_spans(f) {
            if sp.ms < 0.05 {
                continue;
            }
            let heavy = sp.ms > total * 0.3 && sp.ms > 2.0;
            let col = if heavy { theme::WARN } else { theme::TEXT };
            ui.label(RichText::new(format!("{}{}", "   ".repeat(sp.depth as usize), sp.name)).monospace().small().color(col));
            ui.label(RichText::new(format!("{:.2} ms", sp.ms)).monospace().small().color(col));
            ui.end_row();
        }
        for (k, v) in &f.rec.counters {
            ui.label(RichText::new(*k).small().color(theme::TEXT_DIM));
            ui.label(RichText::new(v.to_string()).monospace().small().color(theme::TEXT_DIM));
            ui.end_row();
        }
    });
}

fn memory(s: &AppState, ui: &mut Ui) {
    let Some(g) = &s.gpu else { return };
    super::section(ui, "GPU memory");
    let (uploads, builds, vram) = g.engine.tile_stats();
    ui.label(RichText::new(format!("Resident tiles {} MB · texture pool {} MB", vram >> 20, g.engine.pool_bytes() >> 20)).small());
    ui.label(RichText::new(format!("Tiles uploaded {uploads} · built on the GPU {builds} (since launch)")).small().color(theme::TEXT_DIM));
}
