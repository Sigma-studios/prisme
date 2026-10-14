//! Levels editor: histogram with input black/gamma/white handles and an output range bar.

use egui::{Color32, Pos2, Rect, Sense, Shape, Stroke, Ui, Vec2};

pub struct LevelsValue<'a> {
    pub in_black: &'a mut f32,
    pub in_white: &'a mut f32,
    pub gamma: &'a mut f32,
    pub out_black: &'a mut f32,
    pub out_white: &'a mut f32,
}

fn tri(p: &egui::Painter, x: f32, y: f32, fill: Color32) {
    p.add(Shape::convex_polygon(vec![Pos2::new(x, y), Pos2::new(x - 6.0, y + 10.0), Pos2::new(x + 6.0, y + 10.0)], fill, Stroke::new(1.0, Color32::from_gray(10))));
}

pub fn levels_editor(ui: &mut Ui, id: egui::Id, v: LevelsValue, histogram: Option<&[u32; 256]>, tint: Color32) -> bool {
    let w = ui.available_width().min(320.0);
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, 150.0), Sense::click_and_drag());
    let p = ui.painter_at(rect.expand(8.0));
    let hist = Rect::from_min_size(rect.min, Vec2::new(w, 96.0));
    p.rect_filled(hist, 4.0, crate::theme::BG0);
    if let Some(h) = histogram {
        let max = h.iter().copied().max().unwrap_or(1).max(1) as f32;
        for (i, c) in h.iter().enumerate() {
            let x = hist.left() + hist.width() * (i as f32 + 0.5) / 256.0;
            let hh = hist.height() * (*c as f32 / max).sqrt() * 0.95;
            p.line_segment([Pos2::new(x, hist.bottom()), Pos2::new(x, hist.bottom() - hh)], Stroke::new(hist.width() / 256.0 + 0.3, tint.gamma_multiply(0.7)));
        }
    }
    let x_of = |t: f32| hist.left() + t.clamp(0.0, 1.0) * hist.width();
    let t_of = |x: f32| ((x - hist.left()) / hist.width()).clamp(0.0, 1.0);
    // gamma handle position: midpoint where output = 0.5
    let gx = *v.in_black + (*v.in_white - *v.in_black) * 0.5f32.powf(*v.gamma);
    let in_y = hist.bottom() + 2.0;
    let out_bar = Rect::from_min_size(Pos2::new(hist.left(), hist.bottom() + 20.0), Vec2::new(w, 10.0));
    super::gradient_strip(ui, out_bar, &[Color32::BLACK, Color32::WHITE]);
    let out_y = out_bar.bottom() + 2.0;

    let mut drag: u8 = ui.data(|d| d.get_temp(id)).unwrap_or(0);
    let mut changed = false;
    if resp.drag_started()
        && let Some(pos) = resp.interact_pointer_pos() {
            let cands = [
                (1u8, x_of(*v.in_black), in_y),
                (2, x_of(gx), in_y),
                (3, x_of(*v.in_white), in_y),
                (4, x_of(*v.out_black), out_y),
                (5, x_of(*v.out_white), out_y),
            ];
            drag = cands
                .iter()
                .filter(|(_, x, y)| (pos.x - x).abs() < 9.0 && (pos.y - (y + 5.0)).abs() < 14.0)
                .min_by(|a, b| (pos.x - a.1).abs().total_cmp(&(pos.x - b.1).abs()))
                .map(|c| c.0)
                .unwrap_or(0);
        }
    if resp.dragged()
        && let Some(pos) = resp.interact_pointer_pos() {
            let t = t_of(pos.x);
            match drag {
                1 => *v.in_black = t.min(*v.in_white - 0.01),
                3 => *v.in_white = t.max(*v.in_black + 0.01),
                2 => {
                    let rel = ((t - *v.in_black) / (*v.in_white - *v.in_black)).clamp(0.01, 0.99);
                    *v.gamma = (0.5f32.ln() / rel.ln()).recip().recip();
                    *v.gamma = rel.ln() / 0.5f32.ln();
                    *v.gamma = 1.0 / *v.gamma;
                    *v.gamma = v.gamma.clamp(0.1, 9.99);
                }
                4 => *v.out_black = t,
                5 => *v.out_white = t,
                _ => {}
            }
            changed = drag != 0;
        }
    if resp.drag_stopped() {
        drag = 0;
    }
    if resp.double_clicked() {
        *v.in_black = 0.0;
        *v.in_white = 1.0;
        *v.gamma = 1.0;
        *v.out_black = 0.0;
        *v.out_white = 1.0;
        changed = true;
    }
    ui.data_mut(|d| d.insert_temp(id, drag));
    let gx = *v.in_black + (*v.in_white - *v.in_black) * 0.5f32.powf(*v.gamma);
    tri(&p, x_of(*v.in_black), in_y, Color32::BLACK);
    tri(&p, x_of(gx), in_y, Color32::GRAY);
    tri(&p, x_of(*v.in_white), in_y, Color32::WHITE);
    tri(&p, x_of(*v.out_black), out_y, Color32::BLACK);
    tri(&p, x_of(*v.out_white), out_y, Color32::WHITE);
    ui.horizontal(|ui| {
        let f = |ui: &mut Ui, v: &mut f32, r: std::ops::RangeInclusive<f32>, s: f64| {
            let mut x = *v * 255.0;
            if ui.add(egui::DragValue::new(&mut x).range(r).speed(s).max_decimals(0)).changed() {
                *v = x / 255.0;
                true
            } else {
                false
            }
        };
        changed |= f(ui, v.in_black, 0.0..=255.0, 1.0);
        changed |= ui.add(egui::DragValue::new(v.gamma).range(0.1..=9.99).speed(0.01).max_decimals(2)).changed();
        changed |= f(ui, v.in_white, 0.0..=255.0, 1.0);
        ui.separator();
        changed |= f(ui, v.out_black, 0.0..=255.0, 1.0);
        changed |= f(ui, v.out_white, 0.0..=255.0, 1.0);
    });
    changed
}
