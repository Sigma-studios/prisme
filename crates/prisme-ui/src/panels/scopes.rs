//! Histogram, RGB waveform and vectorscope (computed on the GPU from the displayed image).
//! Histogram and waveform cover an extended linear range (× SDR white, sRGB-encoded axis)
//! so HDR values above 1.0 are visible; a marker shows SDR white and a readout counts
//! samples above it.

use crate::app::AppState;
use egui::{Color32, ColorImage, Pos2, Shape, Stroke, TextureHandle, Ui, Vec2};

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum View {
    #[default]
    Histogram,
    Waveform,
    Vectorscope,
}

pub fn ui(s: &mut AppState, ui: &mut Ui) {
    let vid = ui.id().with("scope-view");
    let mut view: View = ui.data(|d| d.get_temp(vid)).unwrap_or_default();
    ui.horizontal(|ui| {
        ui.selectable_value(&mut view, View::Histogram, "Histogram");
        ui.selectable_value(&mut view, View::Waveform, "Waveform");
        ui.selectable_value(&mut view, View::Vectorscope, "Vectorscope");
    });
    ui.data_mut(|d| d.insert_temp(vid, view));
    if view != View::Vectorscope {
        ui.horizontal(|ui| {
            ui.label("Range");
            let r = s.settings.scope_range;
            let label = |r: f32| if r < 1.0 { "Auto".to_string() } else if r == 1.0 { "1× (SDR)".to_string() } else { format!("{r:.0}× SDR white") };
            egui::ComboBox::from_id_salt("scope-range").selected_text(label(r)).show_ui(ui, |ui| {
                for v in [0.0, 1.0, 2.0, 4.0, 8.0, 16.0] {
                    ui.selectable_value(&mut s.settings.scope_range, v, label(v));
                }
            })
            .response
            .on_hover_text("Linear range shown by the histogram and waveform. Auto: SDR, or the display's HDR peak when HDR output is active.");
        });
    }
    let Some(data) = s.gpu.as_ref().and_then(|g| g.scopes.latest.clone()) else {
        ui.label("Waiting for image…");
        return;
    };
    // Position of SDR white (linear 1.0) on the sRGB-encoded [0, range] axis.
    let white = 1.0 / prisme_color::hdr::encode_extended_srgb(data.range);
    let w = ui.available_width().min(360.0);
    match view {
        View::Histogram => {
            let (rect, _) = ui.allocate_exact_size(Vec2::new(w, 140.0), egui::Sense::hover());
            let p = ui.painter_at(rect);
            p.rect_filled(rect, 4.0, crate::theme::BG0);
            let colors = [Color32::from_rgb(255, 70, 70), Color32::from_rgb(70, 220, 90), Color32::from_rgb(80, 130, 255), Color32::from_gray(220)];
            let max = data.hist_ext.iter().take(3).flat_map(|h| h[1..255].iter()).copied().max().unwrap_or(1).max(1) as f32;
            for (c, h) in data.hist_ext.iter().enumerate().take(3) {
                let pts: Vec<Pos2> = h
                    .iter()
                    .enumerate()
                    .map(|(i, v)| Pos2::new(rect.left() + rect.width() * i as f32 / 255.0, rect.bottom() - rect.height() * (*v as f32 / max).min(1.0) * 0.95))
                    .collect();
                p.add(egui::Shape::line(pts, Stroke::new(1.2, colors[c].gamma_multiply(0.9))));
            }
            let lmax = data.hist_ext[3][1..255].iter().copied().max().unwrap_or(1).max(1) as f32;
            for (i, v) in data.hist_ext[3].iter().enumerate() {
                let x = rect.left() + rect.width() * i as f32 / 255.0;
                p.line_segment([Pos2::new(x, rect.bottom()), Pos2::new(x, rect.bottom() - rect.height() * (*v as f32 / lmax).min(1.0) * 0.95)], Stroke::new(rect.width() / 256.0 + 0.4, Color32::from_white_alpha(28)));
            }
            if data.range > 1.0 {
                let x = rect.left() + rect.width() * white;
                p.add(Shape::dashed_line(&[Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())], Stroke::new(1.0, crate::theme::WARN.gamma_multiply(0.7)), 3.0, 3.0));
                p.text(Pos2::new(x + 3.0, rect.top() + 2.0), egui::Align2::LEFT_TOP, "1.0", egui::FontId::monospace(9.0), crate::theme::WARN);
            }
        }
        View::Waveform | View::Vectorscope => {
            let tex_id = ui.id().with("scope-tex");
            let mut img = ColorImage::new([256, 256], vec![Color32::BLACK; 256 * 256]);
            if view == View::Waveform {
                let max = data.waveform.iter().copied().max().unwrap_or(1).max(1) as f32;
                for i in 0..65536 {
                    let r = (data.waveform[i] as f32 / max * 40.0).min(1.0).sqrt();
                    let g = (data.waveform[65536 + i] as f32 / max * 40.0).min(1.0).sqrt();
                    let b = (data.waveform[131072 + i] as f32 / max * 40.0).min(1.0).sqrt();
                    img.pixels[i] = Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8);
                }
            } else {
                let max = data.vectorscope.iter().copied().max().unwrap_or(1).max(1) as f32;
                for i in 0..65536 {
                    let v = (data.vectorscope[i] as f32 / max * 60.0).min(1.0).sqrt();
                    let (x, y) = ((i % 256) as f32 - 127.5, (i / 256) as f32 - 127.5);
                    let a = (-y).atan2(x);
                    let lab = prisme_color::oklab_to_working([0.7, a.cos() * 0.12, a.sin() * 0.12]);
                    let tint = crate::theme::to_color32([lab[0], lab[1], lab[2], 1.0]);
                    img.pixels[i] = Color32::from_rgb((tint.r() as f32 * v) as u8, (tint.g() as f32 * v) as u8, (tint.b() as f32 * v) as u8);
                    if ((x * x + y * y).sqrt() - 127.0).abs() < 0.8 {
                        img.pixels[i] = Color32::from_gray(60);
                    }
                }
            }
            let tex: TextureHandle = ui.ctx().load_texture(format!("{tex_id:?}"), img, egui::TextureOptions::LINEAR);
            let side = w.min(300.0);
            let r = ui.add(egui::Image::new(&tex).fit_to_exact_size(Vec2::new(if view == View::Waveform { w } else { side }, side * 0.75_f32.max(if view == View::Vectorscope { 1.0 } else { 0.6 }))));
            if view == View::Waveform && data.range > 1.0 {
                let rect = r.rect;
                let y = rect.bottom() - rect.height() * white;
                let p = ui.painter_at(rect);
                p.add(Shape::dashed_line(&[Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)], Stroke::new(1.0, crate::theme::WARN.gamma_multiply(0.7)), 3.0, 3.0));
                p.text(Pos2::new(rect.left() + 2.0, y - 2.0), egui::Align2::LEFT_BOTTOM, "1.0", egui::FontId::monospace(9.0), crate::theme::WARN);
            }
        }
    }
    // Overflow indicator: samples brighter than SDR white (linear sRGB > 1.0).
    if view != View::Vectorscope && data.total > 0 {
        let pct = |n: u32| n as f64 * 100.0 / data.total as f64;
        let any = data.over.iter().any(|&n| n > 0);
        let text = if any {
            format!(
                "Above SDR white: R {:.1}%  G {:.1}%  B {:.1}%  luma {:.1}%",
                pct(data.over[0]),
                pct(data.over[1]),
                pct(data.over[2]),
                pct(data.over[3])
            )
        } else {
            "Nothing above SDR white".to_string()
        };
        ui.label(egui::RichText::new(text).small().color(if any { crate::theme::WARN } else { crate::theme::TEXT_DIM }))
            .on_hover_text("Values > 1.0 are clipped on SDR displays. Use a larger Range to see them.");
    }
}
