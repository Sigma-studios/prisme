use crate::app::AppState;
use crate::theme::{self, to_color32};
use crate::widgets::color_slider;
use crate::widgets::color_wheel::{color_wheel, WheelMode};
use egui::{Color32, RichText, Sense, Ui, Vec2};
use prisme_color::{hsv_to_rgb, oklab_to_oklch, oklab_to_working, oklch_to_oklab, rgb_to_hsv, srgb, working_to_oklab};

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum SliderMode {
    #[default]
    Rgb,
    Hsv,
    Oklch,
}

fn c32(rgb: [f32; 3]) -> Color32 {
    to_color32([rgb[0], rgb[1], rgb[2], 1.0])
}

pub fn ui(s: &mut AppState, ui: &mut Ui) {
    let editing_bg_id = ui.id().with("edit-bg");
    let mut edit_bg: bool = ui.data(|d| d.get_temp(editing_bg_id)).unwrap_or(false);
    ui.horizontal(|ui| {
        ui.selectable_value(&mut edit_bg, false, "Foreground");
        ui.selectable_value(&mut edit_bg, true, "Background");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.selectable_value(&mut s.settings.wheel_mode, WheelMode::Oklch, "OKLCh").on_hover_text("Perceptually uniform wheel");
            ui.selectable_value(&mut s.settings.wheel_mode, WheelMode::Hsv, "HSV");
        });
    });
    ui.data_mut(|d| d.insert_temp(editing_bg_id, edit_bg));
    let mode = s.settings.wheel_mode;
    let mut color = if edit_bg { s.settings.bg } else { s.settings.fg };
    let size = ui.available_width().clamp(140.0, 260.0);
    let mut changed = false;
    ui.vertical_centered(|ui| {
        changed |= color_wheel(ui, ui.id().with(("wheel", edit_bg)), &mut color, mode, size);
    });

    // Swatch, hex, eyedropper.
    ui.horizontal(|ui| {
        let (r, _) = ui.allocate_exact_size(Vec2::new(36.0, 22.0), Sense::hover());
        theme::checker(ui.painter(), r, 5.0);
        ui.painter().rect_filled(r, 3.0, to_color32(color));
        let hex_id = ui.id().with("hex");
        let mut hex: String = ui.data(|d| d.get_temp(hex_id)).unwrap_or_else(|| prisme_color::working_to_hex(color));
        let resp = ui.add(egui::TextEdit::singleline(&mut hex).desired_width(80.0).font(egui::TextStyle::Monospace));
        if resp.changed()
            && let Some(c) = prisme_color::hex_to_working(&hex) {
                color = c;
                changed = true;
            }
        if !resp.has_focus() {
            hex = prisme_color::working_to_hex(color);
        }
        ui.data_mut(|d| d.insert_temp(hex_id, hex));
        if prisme_platform::screen_picker_supported() && ui.button(egui_phosphor::regular::EYEDROPPER).on_hover_text("Pick a colour anywhere on screen (⇧I)").clicked() {
            crate::jobs::pick_screen(s, edit_bg);
        }
    });

    // Sliders.
    let sm_id = ui.id().with("slider-mode");
    let mut sm: SliderMode = ui.data(|d| d.get_temp(sm_id)).unwrap_or_default();
    ui.horizontal(|ui| {
        ui.selectable_value(&mut sm, SliderMode::Rgb, "RGB");
        ui.selectable_value(&mut sm, SliderMode::Hsv, "HSV");
        ui.selectable_value(&mut sm, SliderMode::Oklch, "OKLCh");
    });
    ui.data_mut(|d| d.insert_temp(sm_id, sm));
    let enc = srgb().from_working([color[0], color[1], color[2]]).map(|v| v.clamp(0.0, 1.0));
    match sm {
        SliderMode::Rgb => {
            let mut v = enc.map(|x| x * 255.0);
            let grad = |i: usize| -> Vec<Color32> {
                (0..8).map(|k| {
                    let mut c = enc;
                    c[i] = k as f32 / 7.0;
                    Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8)
                }).collect()
            };
            let mut ch = false;
            ch |= color_slider(ui, "R", &mut v[0], (0.0, 255.0), &grad(0));
            ch |= color_slider(ui, "G", &mut v[1], (0.0, 255.0), &grad(1));
            ch |= color_slider(ui, "B", &mut v[2], (0.0, 255.0), &grad(2));
            if ch {
                let w = srgb().to_working(v.map(|x| x / 255.0));
                color = [w[0], w[1], w[2], color[3]];
                changed = true;
            }
        }
        SliderMode::Hsv => {
            let mut hsv = rgb_to_hsv(enc);
            hsv[0] *= 360.0;
            hsv[1] *= 100.0;
            hsv[2] *= 100.0;
            let hues: Vec<Color32> = (0..13).map(|k| c32(srgb().to_working(hsv_to_rgb([k as f32 / 12.0, 1.0, 1.0])))).collect();
            let sats: Vec<Color32> = (0..8).map(|k| c32(srgb().to_working(hsv_to_rgb([hsv[0] / 360.0, k as f32 / 7.0, hsv[2] / 100.0])))).collect();
            let vals: Vec<Color32> = (0..8).map(|k| c32(srgb().to_working(hsv_to_rgb([hsv[0] / 360.0, hsv[1] / 100.0, k as f32 / 7.0])))).collect();
            let mut ch = false;
            ch |= color_slider(ui, "H", &mut hsv[0], (0.0, 360.0), &hues);
            ch |= color_slider(ui, "S", &mut hsv[1], (0.0, 100.0), &sats);
            ch |= color_slider(ui, "V", &mut hsv[2], (0.0, 100.0), &vals);
            if ch {
                let w = srgb().to_working(hsv_to_rgb([hsv[0] / 360.0, hsv[1] / 100.0, hsv[2] / 100.0]));
                color = [w[0], w[1], w[2], color[3]];
                changed = true;
            }
        }
        SliderMode::Oklch => {
            let mut lch = oklab_to_oklch(working_to_oklab([color[0], color[1], color[2]]));
            lch[2] = lch[2].to_degrees().rem_euclid(360.0);
            let col = |l: [f32; 3]| c32(oklab_to_working(oklch_to_oklab([l[0], l[1], l[2].to_radians()])));
            let ls: Vec<Color32> = (0..8).map(|k| col([k as f32 / 7.0, lch[1], lch[2]])).collect();
            let cs: Vec<Color32> = (0..8).map(|k| col([lch[0], k as f32 / 7.0 * 0.37, lch[2]])).collect();
            let hs: Vec<Color32> = (0..13).map(|k| col([lch[0], lch[1].max(0.08), k as f32 * 30.0])).collect();
            let mut ch = false;
            ch |= color_slider(ui, "L", &mut lch[0], (0.0, 1.0), &ls);
            ch |= color_slider(ui, "C", &mut lch[1], (0.0, 0.37), &cs);
            ch |= color_slider(ui, "H", &mut lch[2], (0.0, 360.0), &hs);
            if ch {
                let w = oklab_to_working(oklch_to_oklab([lch[0], lch[1], lch[2].to_radians()]));
                color = [w[0], w[1], w[2], color[3]];
                changed = true;
            }
        }
    }
    // Unclamped readout: HDR colours (e.g. picked from the canvas) can exceed SDR white.
    let lin = srgb().matrix_from_working().apply([color[0], color[1], color[2]]);
    ui.label(
        RichText::new(format!("Linear sRGB  {:.3}  {:.3}  {:.3}", lin[0], lin[1], lin[2]))
            .small()
            .monospace()
            .color(theme::TEXT_DIM),
    )
    .on_hover_text(format!(
        "Working (linear Rec.2020): {:.4}, {:.4}, {:.4}\n1.0 = SDR reference white; values above it are HDR highlights.",
        color[0], color[1], color[2]
    ));
    let peak = lin[0].max(lin[1]).max(lin[2]);
    if peak > 1.002 {
        ui.label(RichText::new(format!("☀ {peak:.2}× SDR white (sliders show it clipped)")).small().color(theme::WARN));
    }
    if lin.iter().any(|v| *v < -0.002) {
        ui.label(RichText::new("⚠ Outside sRGB gamut (shown clipped)").small().color(theme::WARN));
    }

    if changed {
        if edit_bg {
            s.settings.bg = color;
        } else {
            s.settings.fg = color;
        }
    }

    // Recent colours.
    if !s.settings.recent_colors.is_empty() {
        super::section(ui, "Recent");
        ui.horizontal_wrapped(|ui| {
            for c in s.settings.recent_colors.clone() {
                let (r, resp) = ui.allocate_exact_size(Vec2::splat(18.0), Sense::click());
                ui.painter().rect_filled(r, 3.0, to_color32(c));
                if resp.clicked() {
                    if edit_bg {
                        s.settings.bg = c;
                    } else {
                        s.settings.fg = c;
                    }
                }
            }
        });
    }
    if ui.input(|i| i.pointer.any_released()) && changed {
        crate::canvas::set_color(s, s.settings.fg, false);
    }
}
