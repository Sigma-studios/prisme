//! Curve editor: monotone spline through draggable points, histogram backdrop.
//! Click to add a point, drag to move, double-click or drag far outside to remove.

use egui::{Color32, Pos2, Rect, Sense, Shape, Stroke, Ui, Vec2};
use prisme_graph::eval_curve;

pub fn curve_editor(ui: &mut Ui, id: egui::Id, points: &mut Vec<[f32; 2]>, color: Color32, histogram: Option<&[u32; 256]>) -> bool {
    let w = ui.available_width().min(320.0);
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(w), Sense::click_and_drag());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 4.0, crate::theme::BG0);
    if let Some(h) = histogram {
        let max = h.iter().copied().max().unwrap_or(1).max(1) as f32;
        let mut pts = vec![rect.left_bottom()];
        for (i, v) in h.iter().enumerate() {
            let x = rect.left() + rect.width() * i as f32 / 255.0;
            pts.push(Pos2::new(x, rect.bottom() - rect.height() * ((*v as f32 / max).sqrt()) * 0.9));
        }
        pts.push(rect.right_bottom());
        p.add(Shape::convex_polygon(vec![], Color32::TRANSPARENT, Stroke::NONE));
        for wnd in pts.windows(2) {
            p.add(Shape::convex_polygon(
                vec![wnd[0], wnd[1], Pos2::new(wnd[1].x, rect.bottom()), Pos2::new(wnd[0].x, rect.bottom())],
                Color32::from_white_alpha(18),
                Stroke::NONE,
            ));
        }
    }
    for i in 1..4 {
        let t = i as f32 / 4.0;
        let s = Stroke::new(0.5, Color32::from_white_alpha(30));
        p.line_segment([Pos2::new(rect.left() + rect.width() * t, rect.top()), Pos2::new(rect.left() + rect.width() * t, rect.bottom())], s);
        p.line_segment([Pos2::new(rect.left(), rect.top() + rect.height() * t), Pos2::new(rect.right(), rect.top() + rect.height() * t)], s);
    }
    p.line_segment([rect.left_bottom(), rect.right_top()], Stroke::new(0.5, Color32::from_white_alpha(40)));
    let to_screen = |q: [f32; 2]| Pos2::new(rect.left() + q[0] * rect.width(), rect.bottom() - q[1] * rect.height());
    let to_curve = |s: Pos2| [((s.x - rect.left()) / rect.width()).clamp(0.0, 1.0), ((rect.bottom() - s.y) / rect.height()).clamp(0.0, 1.0)];

    let mut dragging: Option<usize> = ui.data(|d| d.get_temp::<Option<usize>>(id)).flatten();
    let mut changed = false;
    // Use the press origin: by the time drag_started fires the pointer has moved.
    let origin = ui.input(|i| i.pointer.press_origin()).or(resp.interact_pointer_pos());
    if (resp.drag_started() || resp.clicked() || resp.double_clicked())
        && let Some(pos) = origin {
            let hit = points.iter().position(|q| to_screen(*q).distance(pos) < 9.0);
            if resp.double_clicked() {
                if let Some(i) = hit
                    && points.len() > 2 {
                        points.remove(i);
                        changed = true;
                    }
                dragging = None;
            } else {
                dragging = match hit {
                    Some(i) => Some(i),
                    None => {
                        let c = to_curve(pos);
                        let y = eval_curve(points, c[0]);
                        points.push([c[0], if (to_screen([c[0], y]).y - pos.y).abs() < 14.0 { y } else { c[1] }]);
                        points.sort_by(|a, b| a[0].total_cmp(&b[0]));
                        changed = true;
                        points.iter().position(|q| q[0] == c[0])
                    }
                };
            }
        }
    if let (Some(i), Some(pos)) = (dragging, resp.interact_pointer_pos())
        && resp.dragged() && i < points.len() {
            let outside = !rect.expand(30.0).contains(pos);
            if outside && points.len() > 2 && i != 0 && i != points.len() - 1 {
                points.remove(i);
                dragging = None;
            } else {
                let mut c = to_curve(pos);
                // keep x ordering between neighbours
                let lo = if i > 0 { points[i - 1][0] + 0.005 } else { 0.0 };
                let hi = if i + 1 < points.len() { points[i + 1][0] - 0.005 } else { 1.0 };
                c[0] = c[0].clamp(lo, hi.max(lo));
                points[i] = c;
            }
            changed = true;
        }
    if resp.drag_stopped() {
        dragging = None;
    }
    ui.data_mut(|d| d.insert_temp(id, dragging));

    let line: Vec<Pos2> = (0..=128).map(|i| {
        let x = i as f32 / 128.0;
        to_screen([x, eval_curve(points, x).clamp(0.0, 1.0)])
    }).collect();
    p.add(Shape::line(line, Stroke::new(2.0, color)));
    for (i, q) in points.iter().enumerate() {
        let s = to_screen(*q);
        let active = dragging == Some(i);
        p.circle_filled(s, if active { 5.0 } else { 4.0 }, if active { color } else { crate::theme::BG1 });
        p.circle_stroke(s, 4.5, Stroke::new(1.5, color));
    }
    if let Some(h) = resp.hover_pos() {
        let c = to_curve(h);
        p.text(rect.left_top() + Vec2::new(6.0, 4.0), egui::Align2::LEFT_TOP, format!("{:.0} → {:.0}", c[0] * 255.0, eval_curve(points, c[0]) * 255.0), egui::FontId::monospace(10.0), crate::theme::TEXT_DIM);
    }
    let _ = Rect::NOTHING;
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drag_moves_point_without_adding() {
        let ctx = egui::Context::default();
        let mut pts = vec![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]];
        let rect = std::cell::Cell::new(egui::Rect::NOTHING);
        let run = |pts: &mut Vec<[f32; 2]>, events: Vec<egui::Event>| {
            let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(600.0, 600.0))), events, ..Default::default() };
            let mut out = ctx.run_ui(raw, |ui| {
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    rect.set(ui.available_rect_before_wrap());
                    curve_editor(ui, egui::Id::new("c"), pts, Color32::WHITE, None);
                });
            });
            out.textures_delta.clear();
        };
        run(&mut pts, vec![]);
        let rect = rect.get();
        let w = rect.width().min(320.0);
        let to_s = |q: [f32; 2]| egui::pos2(rect.left() + q[0] * w, rect.top() + w - q[1] * w);
        let a = to_s([0.5, 0.5]);
        let b = to_s([0.5, 0.8]);
        let btn = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
        run(&mut pts, vec![egui::Event::PointerMoved(a), btn(a, true)]);
        for k in 1..=6 {
            run(&mut pts, vec![egui::Event::PointerMoved(a + (b - a) * (k as f32 / 6.0))]);
        }
        run(&mut pts, vec![btn(b, false)]);
        assert_eq!(pts.len(), 3, "{pts:?}");
        assert!((pts[1][1] - 0.8).abs() < 0.03, "{pts:?}");
    }
}
