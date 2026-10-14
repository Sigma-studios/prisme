//! Lift/gamma/gain style colour-offset wheel: a puck on a hue/chroma disk plus a
//! luminance slider. Value is [a, b, L] where (a, b) in the unit disk.

use egui::{Color32, Mesh, Pos2, Sense, Stroke, Ui, Vec2};
use prisme_color::{oklab_to_working, srgb};

pub fn offset_wheel(ui: &mut Ui, label: &str, v: &mut [f32; 3], size: f32) -> bool {
    let mut changed = false;
    ui.vertical(|ui| {
        ui.set_width(size + 8.0);
        ui.label(egui::RichText::new(label).small().color(crate::theme::TEXT_DIM));
        let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size), Sense::click_and_drag());
        let c = rect.center();
        let r = size * 0.5 - 3.0;
        let painter = ui.painter_at(rect.expand(2.0));
        let mut mesh = Mesh::default();
        mesh.colored_vertex(c, Color32::from_gray(80));
        let n = 72;
        for i in 0..=n {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            let w = oklab_to_working([0.62, a.cos() * 0.12, a.sin() * 0.12]);
            let s = srgb().from_working(w);
            let q = |x: f32| (x.clamp(0.0, 1.0) * 255.0) as u8;
            mesh.colored_vertex(c + Vec2::angled(a) * r, Color32::from_rgb(q(s[0]), q(s[1]), q(s[2])));
            if i > 0 {
                mesh.add_triangle(0, i as u32, i as u32 + 1);
            }
        }
        painter.add(mesh);
        painter.circle_stroke(c, r, Stroke::new(1.0, Color32::from_gray(20)));
        painter.circle_stroke(c, r * 0.5, Stroke::new(0.5, Color32::from_black_alpha(80)));
        painter.line_segment([c - Vec2::X * r, c + Vec2::X * r], Stroke::new(0.5, Color32::from_black_alpha(80)));
        painter.line_segment([c - Vec2::Y * r, c + Vec2::Y * r], Stroke::new(0.5, Color32::from_black_alpha(80)));
        let fine = ui.input(|i| i.modifiers.shift);
        if resp.double_clicked() {
            v[0] = 0.0;
            v[1] = 0.0;
            changed = true;
        } else if resp.dragged() {
            let k = if fine { 0.15 } else { 1.0 };
            let d = resp.drag_delta() / r * k;
            v[0] += d.x;
            v[1] -= d.y;
            let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
            if len > 1.0 {
                v[0] /= len;
                v[1] /= len;
            }
            changed = true;
        }
        let p = Pos2::new(c.x + v[0] * r, c.y - v[1] * r);
        painter.circle_filled(p, 5.0, Color32::WHITE);
        painter.circle_stroke(p, 6.0, Stroke::new(1.0, Color32::BLACK));
        let mut l = v[2];
        let s = ui.add(egui::Slider::new(&mut l, -0.5..=0.5).show_value(false));
        if s.double_clicked() {
            l = 0.0;
        }
        if l != v[2] {
            v[2] = l;
            changed = true;
        }
    });
    changed
}
