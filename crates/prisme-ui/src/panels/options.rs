//! Context-sensitive tool options bar.

use crate::app::AppState;
use egui::{ComboBox, DragValue, RichText, Ui};
use prisme_brush::{presets, BrushSettings, TipShape};
use prisme_core::gradient::{GradientKind, GradientRepeat};
use prisme_core::SelectionOp;
use prisme_tools::{LiquifyMode, ToolKind};

fn brush_controls(ui: &mut Ui, b: &mut BrushSettings, id: &str) {
    ui.label("Size");
    ui.add(DragValue::new(&mut b.size).range(1.0..=2000.0).speed(0.5).suffix(" px"));
    ui.label("Hardness");
    ui.add(egui::Slider::new(&mut b.hardness, 0.0..=1.0).show_value(false));
    ui.label("Opacity");
    let mut op = b.opacity * 100.0;
    if ui.add(DragValue::new(&mut op).range(0.0..=100.0).suffix("%")).changed() {
        b.opacity = op / 100.0;
    }
    ui.label("Flow");
    let mut fl = b.flow * 100.0;
    if ui.add(DragValue::new(&mut fl).range(1.0..=100.0).suffix("%")).changed() {
        b.flow = fl / 100.0;
    }
    ComboBox::from_id_salt(("tip", id)).selected_text(b.tip.name()).width(80.0).show_ui(ui, |ui| {
        for t in TipShape::ALL {
            ui.selectable_value(&mut b.tip, t, t.name());
        }
    });
    ui.checkbox(&mut b.pixel_perfect, "Pixel").on_hover_text("Aliased, pixel-grid snapped dabs");
    ui.label("Smooth");
    ui.add(egui::Slider::new(&mut b.smoothing, 0.0..=0.95).show_value(false));
}

/// Liquify session state: (open, can undo, can redo).
fn liquify_state(s: &mut AppState) -> (bool, bool, bool) {
    let Some(t) = s.tools.get_mut(&ToolKind::Liquify) else { return (false, false, false) };
    let any: &mut dyn std::any::Any = t.as_mut();
    let Some(l) = any.downcast_mut::<prisme_tools::tools::LiquifyTool>() else { return (false, false, false) };
    let (u, r) = l.can_undo();
    (l.active(), u, r)
}

pub fn ui(s: &mut AppState, ui: &mut Ui) {
    let lq = if s.tool == ToolKind::Liquify { liquify_state(s) } else { (false, false, false) };
    let mut lq_key = None;
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{} {}", crate::app::tool_icon(s.tool), s.tool.name())).strong());
        ui.separator();
        let o = &mut s.settings.tools;
        match s.tool {
            k if k.is_selection() => {
                for op in [SelectionOp::Replace, SelectionOp::Add, SelectionOp::Subtract, SelectionOp::Intersect] {
                    ui.selectable_value(&mut o.sel_op, op, op.name());
                }
                ui.separator();
                ui.label("Feather");
                ui.add(DragValue::new(&mut o.sel_feather).range(0.0..=500.0).suffix(" px"));
                ui.checkbox(&mut o.sel_antialias, "Anti-alias");
                if matches!(k, ToolKind::SelectRect | ToolKind::SelectEllipse) {
                    ui.label("Ratio");
                    ui.add(DragValue::new(&mut o.sel_fixed_ratio).range(0.0..=20.0).speed(0.01)).on_hover_text("Fixed aspect ratio (0 = free)");
                }
                if matches!(k, ToolKind::ColorSelect | ToolKind::BlobSelect) {
                    ui.separator();
                    ui.label("Tolerance");
                    ui.add(egui::Slider::new(&mut o.tolerance, 0.0..=1.0));
                    ui.checkbox(&mut o.sample_all_layers, "Sample all layers");
                }
                ui.label(RichText::new("⇧ add · ⌥ subtract · ⇧⌥ intersect").small().color(crate::theme::TEXT_DIM));
            }
            ToolKind::Brush => brush_controls(ui, &mut o.brush, "brush"),
            ToolKind::Eraser => brush_controls(ui, &mut o.eraser, "eraser"),
            ToolKind::Clone | ToolKind::Heal => {
                brush_controls(ui, &mut o.clone_brush, "clone");
                ui.checkbox(&mut o.clone_aligned, "Aligned");
                ui.label(RichText::new("⌥-click sets source").small().color(crate::theme::TEXT_DIM));
            }
            ToolKind::Smudge | ToolKind::Blur => {
                ui.label("Size");
                ui.add(DragValue::new(&mut o.retouch_brush.size).range(1.0..=1000.0).suffix(" px"));
                ui.label("Strength");
                ui.add(egui::Slider::new(&mut o.retouch_strength, 0.0..=1.0));
                ui.label("Hardness");
                ui.add(egui::Slider::new(&mut o.retouch_brush.hardness, 0.0..=1.0).show_value(false));
            }
            ToolKind::Fill => {
                ui.label("Tolerance");
                ui.add(egui::Slider::new(&mut o.tolerance, 0.0..=1.0));
                ui.label("Opacity");
                ui.add(egui::Slider::new(&mut o.fill_opacity, 0.0..=1.0));
                ui.checkbox(&mut o.fill_antialias, "Anti-alias");
                ui.checkbox(&mut o.sample_all_layers, "Sample all layers");
            }
            ToolKind::Gradient => {
                ComboBox::from_id_salt("gkind").selected_text(o.gradient_kind.name()).show_ui(ui, |ui| {
                    for k in GradientKind::ALL {
                        ui.selectable_value(&mut o.gradient_kind, k, k.name());
                    }
                });
                ComboBox::from_id_salt("grep").selected_text(format!("{:?}", o.gradient_repeat)).show_ui(ui, |ui| {
                    for r in [GradientRepeat::Pad, GradientRepeat::Repeat, GradientRepeat::Mirror] {
                        ui.selectable_value(&mut o.gradient_repeat, r, format!("{r:?}"));
                    }
                });
                ui.checkbox(&mut o.gradient_use_fg_bg, "Use FG/BG");
                ui.checkbox(&mut o.gradient_reverse, "Reverse");
                ui.checkbox(&mut o.gradient_dither, "Dither");
                ui.checkbox(&mut o.gradient_as_layer, "As fill layer").on_hover_text("Create a non-destructive gradient fill layer");
                ui.label("Opacity");
                ui.add(egui::Slider::new(&mut o.gradient_opacity, 0.0..=1.0).show_value(false));
            }
            ToolKind::Crop => {
                ui.checkbox(&mut o.crop_delete_outside, "Delete cropped pixels");
                ui.label(RichText::new("Enter / double-click to apply · Esc to cancel").small().color(crate::theme::TEXT_DIM));
            }
            ToolKind::Transform => {
                ui.label(RichText::new("Drag corners to scale (⇧ keep aspect) · outside to rotate · ⌘/Ctrl+corner to distort · Enter to apply").small().color(crate::theme::TEXT_DIM));
            }
            ToolKind::Move => {
                ui.checkbox(&mut o.move_auto_select, "Auto-select layer");
                ui.label(RichText::new("Arrow keys nudge").small().color(crate::theme::TEXT_DIM));
            }
            ToolKind::Picker => {
                ui.label(RichText::new("Click: displayed colour · ⇧ click: raw layer value · ⌥ click: background").small().color(crate::theme::TEXT_DIM));
                if prisme_platform::screen_picker_supported() && ui.button("Pick from screen…").clicked() {
                    crate::jobs::pick_screen(s, false);
                }
            }
            ToolKind::Ruler => {
                let angle = s.tools.get_mut(&ToolKind::Ruler).and_then(|t| {
                    let any: &mut dyn std::any::Any = t.as_mut();
                    any.downcast_mut::<prisme_tools::tools::RulerTool>().and_then(|r| r.angle())
                });
                if let Some(a) = angle
                    && ui.button("Straighten").on_hover_text("Rotate the image so the measured line is level").clicked() {
                        crate::commands::straighten(s, a);
                    }
            }
            ToolKind::Liquify => {
                let l = &mut o.liquify;
                ComboBox::from_id_salt("liquify-mode").selected_text(l.mode.name()).width(150.0).show_ui(ui, |ui| {
                    for m in LiquifyMode::ALL {
                        ui.selectable_value(&mut l.mode, m, m.name());
                    }
                });
                ui.label("Size");
                ui.add(DragValue::new(&mut l.size).range(1.0..=2000.0).speed(0.5).suffix(" px"));
                ui.label("Pressure");
                ui.add(egui::Slider::new(&mut l.pressure, 0.01..=1.0).show_value(false));
                ui.label("Density");
                ui.add(egui::Slider::new(&mut l.density, 0.0..=1.0).show_value(false)).on_hover_text("Edge falloff: low = soft, high = firm");
                ui.add_enabled_ui(l.mode.stationary(), |ui| {
                    ui.label("Rate");
                    ui.add(egui::Slider::new(&mut l.rate, 0.01..=1.0).show_value(false)).on_hover_text("Speed of bloat, pucker, twirl, smooth and reconstruct (they keep acting while the pointer is held still)");
                });
                ui.checkbox(&mut l.show_mask, "Show freeze");
                ui.separator();
                let (open, undo, redo) = lq;
                if ui.add_enabled(undo, egui::Button::new(egui_phosphor::regular::ARROW_COUNTER_CLOCKWISE)).on_hover_text("Undo last liquify stroke").clicked() {
                    lq_key = Some(prisme_tools::ToolKey::Undo);
                }
                if ui.add_enabled(redo, egui::Button::new(egui_phosphor::regular::ARROW_CLOCKWISE)).on_hover_text("Redo liquify stroke").clicked() {
                    lq_key = Some(prisme_tools::ToolKey::Redo);
                }
                if ui.add_enabled(open, egui::Button::new("Apply")).on_hover_text("Bake the deformation (Enter)").clicked() {
                    lq_key = Some(prisme_tools::ToolKey::Enter);
                }
                if ui.add_enabled(open, egui::Button::new("Cancel")).on_hover_text("Discard the deformation (Esc)").clicked() {
                    lq_key = Some(prisme_tools::ToolKey::Escape);
                }
                ui.label(RichText::new("Raster layer pixels only").small().color(crate::theme::TEXT_DIM));
            }
            ToolKind::Text => {
                ui.label(RichText::new("Click to add text, click text to edit, drag to move · Esc to finish · style in Properties").small().color(crate::theme::TEXT_DIM));
            }
            _ => {}
        }
        if s.tool.uses_brush() || s.tool == ToolKind::Fill {
            ui.separator();
            let has_mask = s.doc().and_then(|d| d.doc().active()).is_some_and(|l| l.mask.is_some());
            if has_mask {
                ui.toggle_value(&mut s.settings.tools.edit_mask, "Paint mask").on_hover_text("Paint on the active layer's mask (white reveals, black hides)");
            }
        }
        if matches!(s.tool, ToolKind::Brush | ToolKind::Eraser) {
            ComboBox::from_id_salt("preset").selected_text(s.settings.tools.brush_preset.clone()).show_ui(ui, |ui| {
                for p in presets() {
                    if ui.selectable_label(s.settings.tools.brush_preset == p.name, p.name).clicked() {
                        s.settings.tools.brush_preset = p.name.into();
                        let target = if s.tool == ToolKind::Eraser { &mut s.settings.tools.eraser } else { &mut s.settings.tools.brush };
                        *target = p.settings.clone();
                    }
                }
            });
        }
    });
    if let Some(k) = lq_key {
        crate::commands::tool_key(s, &k);
    }
}
