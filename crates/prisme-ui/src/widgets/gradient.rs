//! Gradient editor: preview bar with draggable stops. Click under the bar to add a stop,
//! drag a stop down/away to remove it, click a stop to edit its colour.

use crate::theme::{checker, to_color32};
use egui::{Color32, Pos2, Rect, Sense, Shape, Stroke, Ui, Vec2};
use prisme_core::gradient::{GradientInterp, GradientStop};
use prisme_core::Gradient;

pub fn gradient_editor(ui: &mut Ui, id: egui::Id, g: &mut Gradient) -> bool {
    let mut changed = false;
    let w = ui.available_width().min(320.0);
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, 46.0), Sense::click_and_drag());
    let bar = Rect::from_min_size(rect.min, Vec2::new(w, 24.0));
    checker(ui.painter(), bar, 6.0);
    let lut = g.to_lut(64);
    super::gradient_strip(ui, bar, &lut.iter().map(|c| to_color32(*c)).collect::<Vec<_>>());
    ui.painter().rect_stroke(bar, 3.0, Stroke::new(1.0, crate::theme::BG3), egui::StrokeKind::Inside);
    let stop_y = bar.bottom() + 4.0;
    let x_of = |t: f32| bar.left() + t * bar.width();
    let mut sel: Option<usize> = ui.data(|d| d.get_temp::<Option<usize>>(id.with("sel"))).flatten();
    let mut drag: Option<usize> = ui.data(|d| d.get_temp::<Option<usize>>(id.with("drag"))).flatten();
    // Hit-test from where the press began (drag_started fires after the pointer moved).
    let origin = ui.input(|i| i.pointer.press_origin()).or(resp.interact_pointer_pos());
    let hit_stop = |p: egui::Pos2| g.stops.iter().position(|s| (x_of(s.pos) - p.x).abs() < 8.0 && p.y > bar.bottom() - 6.0);
    if resp.drag_started() {
        if let Some(p) = origin {
            drag = hit_stop(p);
            if drag.is_some() {
                sel = drag;
            }
        }
    } else if resp.clicked() {
        if let Some(p) = origin {
            match hit_stop(p) {
                Some(i) => sel = Some(i),
                // A click in the strip under the bar adds a stop with the gradient's colour there.
                None if p.y > bar.bottom() => {
                    let t = ((p.x - bar.left()) / bar.width()).clamp(0.0, 1.0);
                    let c = g.eval(t);
                    g.stops.push(GradientStop { pos: t, color: c });
                    sel = Some(g.stops.len() - 1);
                    changed = true;
                }
                None => {}
            }
        }
    }
    if let (Some(i), Some(p)) = (drag, resp.interact_pointer_pos())
        && resp.dragged() && i < g.stops.len() {
            if p.y > rect.bottom() + 30.0 && g.stops.len() > 2 {
                g.stops.remove(i);
                drag = None;
                sel = None;
            } else {
                g.stops[i].pos = ((p.x - bar.left()) / bar.width()).clamp(0.0, 1.0);
            }
            changed = true;
        }
    if resp.drag_stopped() {
        drag = None;
    }
    for (i, s) in g.stops.iter().enumerate() {
        let x = x_of(s.pos);
        let pts = vec![Pos2::new(x, stop_y - 4.0), Pos2::new(x - 6.0, stop_y + 4.0), Pos2::new(x - 6.0, stop_y + 16.0), Pos2::new(x + 6.0, stop_y + 16.0), Pos2::new(x + 6.0, stop_y + 4.0)];
        let stroke = if sel == Some(i) { Stroke::new(2.0, crate::theme::ACCENT) } else { Stroke::new(1.0, Color32::from_gray(15)) };
        ui.painter().add(Shape::convex_polygon(pts, to_color32([s.color[0], s.color[1], s.color[2], 1.0]), stroke));
    }
    if let Some(i) = sel.filter(|&i| i < g.stops.len()) {
        ui.horizontal(|ui| {
            let mut c = to_color32(g.stops[i].color);
            if ui.color_edit_button_srgba(&mut c).changed() {
                g.stops[i].color = crate::theme::from_color32(c);
                changed = true;
            }
            ui.label("Location");
            let mut pct = g.stops[i].pos * 100.0;
            if ui.add(egui::DragValue::new(&mut pct).range(0.0..=100.0).suffix("%").max_decimals(1)).changed() {
                g.stops[i].pos = pct / 100.0;
                changed = true;
            }
            if g.stops.len() > 2 && ui.small_button(egui_phosphor::regular::TRASH).on_hover_text("Remove stop").clicked() {
                g.stops.remove(i);
                sel = None;
                changed = true;
            }
        });
    }
    ui.horizontal(|ui| {
        ui.label("Interpolation");
        egui::ComboBox::from_id_salt(id.with("interp")).selected_text(g.interp.name()).show_ui(ui, |ui| {
            for m in GradientInterp::ALL {
                changed |= ui.selectable_value(&mut g.interp, m, m.name()).changed();
            }
        });
        if ui.small_button(egui_phosphor::regular::ARROWS_LEFT_RIGHT).on_hover_text("Reverse").clicked() {
            for s in &mut g.stops {
                s.pos = 1.0 - s.pos;
            }
            changed = true;
        }
    });
    ui.data_mut(|d| {
        d.insert_temp(id.with("sel"), sel);
        d.insert_temp(id.with("drag"), drag);
    });
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(ctx: &egui::Context, g: &mut Gradient, events: Vec<egui::Event>) -> egui::Rect {
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(600.0, 400.0))),
            events,
            ..Default::default()
        };
        let mut rect = egui::Rect::NOTHING;
        let mut out = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default().show_inside(ui, |ui| {
                rect = ui.available_rect_before_wrap();
                gradient_editor(ui, egui::Id::new("g"), g);
            });
        });
        out.textures_delta.clear();
        rect
    }

    fn btn(pos: egui::Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() }
    }

    #[test]
    fn click_selects_and_drag_moves_stop() {
        let ctx = egui::Context::default();
        let mut g = Gradient::default();
        let r = frame(&ctx, &mut g, vec![]);
        frame(&ctx, &mut g, vec![]);
        let w = r.width().min(320.0);
        // stop 1 (pos 1.0) handle centre
        let stop = egui::pos2(r.left() + w - 0.5, r.top() + 24.0 + 12.0);
        frame(&ctx, &mut g, vec![egui::Event::PointerMoved(stop), btn(stop, true)]);
        frame(&ctx, &mut g, vec![btn(stop, false)]);
        frame(&ctx, &mut g, vec![]);
        assert_eq!(g.stops.len(), 2, "clicking a stop must not add stops: {:?}", g.stops);
        // drag stop 1 to the middle
        let mid = egui::pos2(r.left() + w * 0.5, stop.y);
        frame(&ctx, &mut g, vec![egui::Event::PointerMoved(stop), btn(stop, true)]);
        for k in 1..=6 {
            frame(&ctx, &mut g, vec![egui::Event::PointerMoved(stop + (mid - stop) * (k as f32 / 6.0))]);
        }
        frame(&ctx, &mut g, vec![btn(mid, false)]);
        assert_eq!(g.stops.len(), 2, "dragging must not add stops: {:?}", g.stops);
        assert!((g.stops[1].pos - 0.5).abs() < 0.05, "stop should move to the middle: {:?}", g.stops);
    }
}
