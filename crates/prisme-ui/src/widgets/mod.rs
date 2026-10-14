pub mod color_wheel;
pub mod curves;
pub mod gradient;
pub mod levels;
pub mod wheel3;

use egui::{Color32, Mesh, Pos2, Rect, Response, Sense, Ui, Vec2};

/// A labelled slider row with drag-to-scrub value and double-click reset.
pub fn slider_row(ui: &mut Ui, label: &str, v: &mut f32, range: std::ops::RangeInclusive<f32>, default: f32, log: bool) -> Response {
    ui.horizontal(|ui| {
        ui.add_sized([110.0, 18.0], egui::Label::new(egui::RichText::new(label).color(crate::theme::TEXT_DIM)).truncate());
        let mut r = ui.add(egui::Slider::new(v, range).logarithmic(log).clamping(egui::SliderClamping::Never).max_decimals(3));
        if r.double_clicked() {
            *v = default;
            r.mark_changed();
        }
        r
    })
    .inner
}

/// Horizontal gradient strip of colours (for sliders / previews).
pub fn gradient_strip(ui: &Ui, rect: Rect, colors: &[Color32]) {
    let mut mesh = Mesh::default();
    let n = colors.len().max(2);
    for (i, c) in colors.iter().enumerate() {
        let x = rect.left() + rect.width() * i as f32 / (n - 1) as f32;
        mesh.colored_vertex(Pos2::new(x, rect.top()), *c);
        mesh.colored_vertex(Pos2::new(x, rect.bottom()), *c);
        if i > 0 {
            let b = (i as u32 - 1) * 2;
            mesh.add_triangle(b, b + 1, b + 2);
            mesh.add_triangle(b + 1, b + 2, b + 3);
        }
    }
    ui.painter().add(mesh);
}

/// A channel slider drawn over a colour gradient (used by the colour panel).
pub fn color_slider(ui: &mut Ui, label: &str, v: &mut f32, range: (f32, f32), colors: &[Color32]) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.add_sized([18.0, 16.0], egui::Label::new(egui::RichText::new(label).monospace().color(crate::theme::TEXT_DIM)));
        let w = (ui.available_width() - 58.0).max(60.0);
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, 16.0), Sense::click_and_drag());
        let bar = rect.shrink2(Vec2::new(0.0, 3.0));
        crate::theme::checker(ui.painter(), bar, 5.0);
        gradient_strip(ui, bar, colors);
        ui.painter().rect_stroke(bar, 3.0, egui::Stroke::new(1.0, crate::theme::BG3), egui::StrokeKind::Inside);
        if let Some(p) = resp.interact_pointer_pos()
            && (resp.dragged() || resp.clicked()) {
                let t = ((p.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                *v = range.0 + t * (range.1 - range.0);
                changed = true;
            }
        let t = ((*v - range.0) / (range.1 - range.0)).clamp(0.0, 1.0);
        let x = rect.left() + t * rect.width();
        let knob = Rect::from_center_size(Pos2::new(x, rect.center().y), Vec2::new(6.0, 16.0));
        ui.painter().rect_filled(knob, 2.0, Color32::WHITE);
        ui.painter().rect_stroke(knob, 2.0, egui::Stroke::new(1.0, Color32::BLACK), egui::StrokeKind::Outside);
        let r = ui.add(egui::DragValue::new(v).range(range.0..=range.1).speed((range.1 - range.0) / 300.0).max_decimals(2));
        changed |= r.changed();
    });
    changed
}

/// Colour swatch button opening a popup with the colour wheel and hex entry.
/// `color` is linear working RGBA. Returns true when changed.
pub fn color_button(ui: &mut Ui, id: egui::Id, color: &mut [f32; 4], alpha: bool) -> bool {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(36.0, 18.0), Sense::click());
    crate::theme::checker(ui.painter(), rect, 4.0);
    ui.painter().rect_filled(rect, 3.0, crate::theme::to_color32(*color));
    ui.painter().rect_stroke(rect, 3.0, egui::Stroke::new(1.0, crate::theme::BG3), egui::StrokeKind::Outside);
    let mut changed = false;
    egui::Popup::from_toggle_button_response(&resp)
        .id(id.with("popup"))
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.set_min_width(220.0);
            changed |= color_wheel::color_wheel(ui, id.with("wheel"), color, color_wheel::WheelMode::Hsv, 200.0);
            ui.horizontal(|ui| {
                let mut hex = prisme_color::working_to_hex(*color);
                if ui.add(egui::TextEdit::singleline(&mut hex).desired_width(90.0)).changed()
                    && let Some(c) = prisme_color::hex_to_working(&hex) {
                        *color = [c[0], c[1], c[2], if alpha { c[3] } else { color[3] }];
                        changed = true;
                    }
                if alpha {
                    ui.label("α");
                    changed |= ui.add(egui::Slider::new(&mut color[3], 0.0..=1.0).show_value(true)).changed();
                }
            });
        });
    changed
}
