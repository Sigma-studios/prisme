//! Brush presets with live stroke previews and detailed engine settings.

use crate::app::AppState;
use crate::widgets::slider_row;
use egui::{ColorImage, RichText, TextureHandle, Ui};
use prisme_brush::{presets, BlendTarget, BrushSettings, InputSample, InputSource, Stroke, StrokeCanvas};
use prisme_core::{IRect, PixelFormat, TileGrid};
use prisme_tools::ToolKind;
use std::sync::Arc;

/// Render a short S-curve stroke with the brush into an image (white on dark).
fn preview(settings: &BrushSettings, w: usize, h: usize) -> ColorImage {
    let mut st = settings.clone();
    st.size = st.size.clamp(2.0, h as f32 * 0.6);
    if st.pixel_perfect {
        st.size = st.size.max(3.0);
    }
    let base = Arc::new(TileGrid::rgba(PixelFormat::F32));
    let mut out = (*base).clone();
    let mut canvas = StrokeCanvas::new(base, BlendTarget::Color([1.0, 1.0, 1.0, 1.0]), st.opacity, None, false, 0.0);
    let mut stroke = Stroke::new(st, 42);
    let n = 40;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let x = 10.0 + t * (w as f32 - 20.0);
        let y = h as f32 * 0.5 + (t * std::f32::consts::TAU).sin() * h as f32 * 0.22;
        let pressure = (t * std::f32::consts::PI).sin().max(0.15);
        let dabs = stroke.add(InputSample { pos: [x, y], pressure, tilt: 0.0, tilt_dir: 0.0, time: t as f64 * 0.5 });
        canvas.apply(&dabs, &mut out);
    }
    let px = out.read_rect(&IRect::new(0, 0, w as i32, h as i32));
    let mut img = ColorImage::new([w, h], vec![egui::Color32::TRANSPARENT; w * h]);
    for i in 0..w * h {
        let a = px[i * 4 + 3].clamp(0.0, 1.0);
        let v = (a * 230.0) as u8;
        img.pixels[i] = egui::Color32::from_rgb(28 + v * 3 / 4, 29 + v * 3 / 4, 33 + v * 3 / 4);
    }
    img
}

pub fn ui(s: &mut AppState, ui: &mut Ui) {
    let target_tool = if matches!(s.tool, ToolKind::Eraser) { ToolKind::Eraser } else { ToolKind::Brush };
    ui.label(RichText::new(format!("Presets for {}", target_tool.name())).small().color(crate::theme::TEXT_DIM));
    let cache_id = ui.id().with("previews");
    let mut cache: Vec<(String, TextureHandle)> = ui.data(|d| d.get_temp(cache_id)).unwrap_or_default();
    if cache.is_empty() {
        for p in presets() {
            let img = preview(&p.settings, 180, 44);
            cache.push((p.name.to_string(), ui.ctx().load_texture(format!("brush-{}", p.name), img, egui::TextureOptions::LINEAR)));
        }
    }
    let current = s.settings.tools.brush_preset.clone();
    egui::ScrollArea::vertical().id_salt("presets").max_height(260.0).show(ui, |ui| {
        for (p, (_, tex)) in presets().into_iter().zip(cache.iter()) {
            let selected = current == p.name;
            let frame = egui::Frame::new()
                .fill(if selected { crate::theme::ACCENT_DIM.gamma_multiply(0.5) } else { crate::theme::BG2 })
                .corner_radius(5)
                .inner_margin(4);
            let r = frame
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.add(egui::Image::new(tex).fit_to_exact_size(egui::vec2(140.0, 34.0)));
                        ui.label(p.name);
                    });
                })
                .response;
            if ui.interact(r.rect, ui.id().with(p.name), egui::Sense::click()).clicked() {
                s.settings.tools.brush_preset = p.name.into();
                if target_tool == ToolKind::Eraser {
                    s.settings.tools.eraser = p.settings.clone();
                } else {
                    s.settings.tools.brush = p.settings.clone();
                }
            }
        }
    });
    ui.data_mut(|d| d.insert_temp(cache_id, cache));

    ui.separator();
    super::section(ui, "Brush settings");
    let b = if target_tool == ToolKind::Eraser { &mut s.settings.tools.eraser } else { &mut s.settings.tools.brush };
    slider_row(ui, "Size", &mut b.size, 1.0..=2000.0, 20.0, true);
    slider_row(ui, "Hardness", &mut b.hardness, 0.0..=1.0, 0.8, false);
    slider_row(ui, "Spacing", &mut b.spacing, 0.01..=2.0, 0.1, true);
    slider_row(ui, "Flow", &mut b.flow, 0.01..=1.0, 1.0, false);
    slider_row(ui, "Opacity", &mut b.opacity, 0.0..=1.0, 1.0, false);
    let mut deg = b.angle.to_degrees();
    if slider_row(ui, "Angle", &mut deg, -180.0..=180.0, 0.0, false).changed() {
        b.angle = deg.to_radians();
    }
    slider_row(ui, "Roundness", &mut b.roundness, 0.05..=1.0, 1.0, false);
    slider_row(ui, "Grain", &mut b.grain, 0.0..=1.0, 0.0, false);
    slider_row(ui, "Smoothing", &mut b.smoothing, 0.0..=0.95, 0.0, false);
    ui.checkbox(&mut b.airbrush, "Airbrush (build up while held)");
    ui.checkbox(&mut b.pixel_perfect, "Pixel-perfect (aliased)");
    super::section(ui, "Jitter");
    slider_row(ui, "Size jitter", &mut b.jitter_size, 0.0..=1.0, 0.0, false);
    slider_row(ui, "Angle jitter", &mut b.jitter_angle, 0.0..=1.0, 0.0, false);
    slider_row(ui, "Opacity jitter", &mut b.jitter_opacity, 0.0..=1.0, 0.0, false);
    slider_row(ui, "Scatter", &mut b.scatter, 0.0..=2.0, 0.0, false);
    super::section(ui, "Dynamics (tablet-ready)");
    for (label, dynm) in [("Size", &mut b.size_dynamic), ("Opacity", &mut b.opacity_dynamic), ("Flow", &mut b.flow_dynamic)] {
        ui.horizontal(|ui| {
            ui.add_sized([60.0, 18.0], egui::Label::new(label));
            egui::ComboBox::from_id_salt(("dyn", label)).selected_text(dynm.source.name()).width(80.0).show_ui(ui, |ui| {
                for src in [InputSource::None, InputSource::Pressure, InputSource::Velocity, InputSource::Tilt, InputSource::Random] {
                    ui.selectable_value(&mut dynm.source, src, src.name());
                }
            });
            ui.label("min");
            ui.add(egui::DragValue::new(&mut dynm.min).range(0.0..=1.0).speed(0.01));
        });
    }
    ui.horizontal(|ui| {
        ui.add_sized([60.0, 18.0], egui::Label::new("Angle"));
        egui::ComboBox::from_id_salt("angle-src").selected_text(b.angle_source.name()).width(80.0).show_ui(ui, |ui| {
            for src in [InputSource::None, InputSource::Direction, InputSource::Tilt, InputSource::Random] {
                ui.selectable_value(&mut b.angle_source, src, src.name());
            }
        });
    });
}
