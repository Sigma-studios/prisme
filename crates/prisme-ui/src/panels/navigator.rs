//! Navigator: whole-document thumbnail with the visible area; click/drag to pan. Infinite
//! canvases show their (coarse) content bounds, see `Document::extent`.

use crate::app::AppState;
use egui::{Color32, Sense, Stroke, Ui, Vec2};

pub fn ui(s: &mut AppState, ui: &mut Ui) {
    let Some(d) = s.doc() else { return super::no_doc(ui) };
    let doc = d.doc().clone();
    let ext = doc.extent();
    let (ox, oy) = (ext.x0 as f32, ext.y0 as f32);
    let (w, h) = (ext.width().max(1) as f32, ext.height().max(1) as f32);
    let avail = ui.available_width().min(320.0);
    let scale = (avail / w).min(200.0 / h);
    let size = Vec2::new(w * scale, h * scale);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click_and_drag());
    crate::theme::checker(ui.painter(), rect, 6.0);
    let version = d.version;
    let doc_id = d.id;
    let output = doc.graph.output;
    if let Some(g) = s.gpu.as_mut() {
        let key = (doc_id, output);
        let stale = g.thumbs.get(&key).is_none_or(|t| t.0 != version);
        let throttled = ui.input(|i| i.pointer.any_down());
        if stale && !throttled
            && let Some(img) = g.engine.thumbnail(&doc, output, 320) {
                let mut r = g.rs.renderer.write();
                if let Some((_, old, _)) = g.thumbs.remove(&key) {
                    r.free_texture(&old);
                }
                let id = r.register_native_texture(&g.rs.device, img.view(), egui_wgpu::wgpu::FilterMode::Linear);
                g.thumbs.insert(key, (version, id, img));
            }
        if let Some((_, id, img)) = g.thumbs.get(&key) {
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(img.rect.width() as f32 / img.tex.w as f32, img.rect.height() as f32 / img.tex.h as f32));
            ui.painter().image(*id, rect, uv, Color32::WHITE);
        }
    }
    let d = &mut s.docs[s.active];
    // visible area (approximate: canvas size unknown here, use last zoom & a nominal viewport)
    let vp = ui.ctx().content_rect().size() * 0.6;
    let half = Vec2::new(vp.x / d.view.zoom * 0.5, vp.y / d.view.zoom * 0.5);
    let c = egui::pos2(rect.left() + (d.view.pan[0] - ox) * scale, rect.top() + (d.view.pan[1] - oy) * scale);
    let vr = egui::Rect::from_center_size(c, half * 2.0 * scale);
    ui.painter().rect_stroke(vr.intersect(rect.expand(2.0)), 0.0, Stroke::new(1.5, crate::theme::ACCENT), egui::StrokeKind::Middle);
    if let Some(p) = resp.interact_pointer_pos().filter(|_| resp.dragged() || resp.clicked()) {
        d.view.pan = [ox + ((p.x - rect.left()) / scale).clamp(0.0, w), oy + ((p.y - rect.top()) / scale).clamp(0.0, h)];
        d.view.fitted = false;
    }
    ui.horizontal(|ui| {
        let mut z = d.view.zoom * 100.0;
        if ui.add(egui::Slider::new(&mut z, 1.0..=6400.0).logarithmic(true).suffix("%")).changed() {
            let pan = d.view.pan;
            d.view.zoom_at(z / 100.0 / d.view.zoom, pan);
        }
        if ui.button("Fit").clicked() {
            d.view.fitted = false;
            d.view.zoom = 1.0;
            d.view.pan = [0.0, 0.0];
        }
    });
    ui.horizontal(|ui| {
        let mut deg = d.view.rotation.to_degrees();
        ui.label("Rotate view");
        if ui.add(egui::DragValue::new(&mut deg).range(-180.0..=180.0).suffix("°")).changed() {
            d.view.rotation = deg.to_radians();
        }
        if ui.small_button(egui_phosphor::regular::ARROW_COUNTER_CLOCKWISE).clicked() {
            d.view.rotation = 0.0;
        }
    });
}
