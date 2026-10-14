use crate::app::AppState;
use crate::commands as cmd;
use crate::theme;
use egui::{ComboBox, RichText, Sense, Ui};
use egui_phosphor::regular as ph;
use prisme_core::layer::{BlendMode, Layer, LayerKind};
use prisme_core::LayerId;

enum RowAction {
    Select(LayerId),
    ToggleVisible(LayerId),
    ToggleExpand(LayerId),
    Rename(LayerId, String),
    Drop { dragged: LayerId, target: LayerId, into: bool },
    EditMask(LayerId, bool),
    ToggleMask(LayerId),
}

fn kind_icon(l: &Layer) -> &'static str {
    match &l.kind {
        LayerKind::Raster(_) => ph::IMAGE,
        LayerKind::Vector(_) => ph::PEN_NIB,
        LayerKind::Text(_) => ph::TEXT_T,
        LayerKind::Fill(_) => ph::GRADIENT,
        LayerKind::Group(g) => {
            if g.expanded { ph::FOLDER_OPEN } else { ph::FOLDER }
        }
    }
}

fn rows(ui: &mut Ui, layers: &[Layer], depth: usize, active: Option<LayerId>, editing_mask: bool, renaming: &mut Option<(LayerId, String)>, out: &mut Vec<RowAction>) {
    for l in layers.iter().rev() {
        let selected = active == Some(l.id);
        let frame = egui::Frame::new()
            .fill(if selected { theme::ACCENT_DIM.gamma_multiply(0.6) } else { egui::Color32::TRANSPARENT })
            .corner_radius(4)
            .inner_margin(egui::Margin::symmetric(4, 2));
        let resp = frame
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.add_space(depth as f32 * 14.0);
                    let eye = if l.visible { ph::EYE } else { ph::EYE_SLASH };
                    if ui.add(egui::Button::new(RichText::new(eye).size(12.0)).frame(false).min_size(egui::vec2(18.0, 18.0))).on_hover_text("Toggle visibility").clicked() {
                        out.push(RowAction::ToggleVisible(l.id));
                    }
                    if l.is_group() {
                        if ui.add(egui::Button::new(kind_icon(l)).frame(false)).clicked() {
                            out.push(RowAction::ToggleExpand(l.id));
                        }
                    } else {
                        ui.label(RichText::new(kind_icon(l)).color(theme::TEXT_DIM));
                    }
                    if let Some(m) = &l.mask {
                        let txt = RichText::new(ph::SQUARE_HALF).color(if selected && editing_mask { theme::ACCENT } else if m.enabled { theme::TEXT } else { theme::TEXT_DIM });
                        let r = ui.add(egui::Button::new(txt).frame(false)).on_hover_text("Mask — click to paint on it, ⇧-click to disable");
                        if r.clicked() {
                            if ui.input(|i| i.modifiers.shift) {
                                out.push(RowAction::ToggleMask(l.id));
                            } else {
                                out.push(RowAction::EditMask(l.id, !(selected && editing_mask)));
                            }
                        }
                    }
                    match renaming {
                        Some((id, name)) if *id == l.id => {
                            let r = ui.text_edit_singleline(name);
                            if r.lost_focus() {
                                out.push(RowAction::Rename(l.id, name.clone()));
                                *renaming = None;
                            } else {
                                r.request_focus();
                            }
                        }
                        _ => {
                            let name = RichText::new(&l.name).color(if l.visible { theme::TEXT } else { theme::TEXT_DIM });
                            let r = ui.add(egui::Label::new(name).sense(Sense::click()).truncate());
                            if r.double_clicked() {
                                *renaming = Some((l.id, l.name.clone()));
                            } else if r.clicked() {
                                out.push(RowAction::Select(l.id));
                            }
                        }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if l.lock.any() {
                            ui.label(RichText::new(ph::LOCK).small());
                        }
                        if l.opacity < 1.0 {
                            ui.label(RichText::new(format!("{:.0}%", l.opacity * 100.0)).small().color(theme::TEXT_DIM));
                        }
                        if l.blend != BlendMode::Normal {
                            ui.label(RichText::new(l.blend.name()).small().color(theme::TEXT_DIM));
                        }
                    });
                });
            })
            .response;
        let resp = ui.interact(resp.rect, ui.id().with(("layer-row", l.id.0)), Sense::click_and_drag());
        if resp.clicked() {
            out.push(RowAction::Select(l.id));
        }
        resp.dnd_set_drag_payload(l.id);
        if let Some(dragged) = resp.dnd_release_payload::<LayerId>() {
            let into = l.is_group() && ui.input(|i| i.pointer.hover_pos()).is_some_and(|p| (p.y - resp.rect.center().y).abs() < resp.rect.height() * 0.25);
            out.push(RowAction::Drop { dragged: *dragged, target: l.id, into });
        }
        if resp.dnd_hover_payload::<LayerId>().is_some() {
            ui.painter().hline(resp.rect.x_range(), resp.rect.top(), egui::Stroke::new(2.0, theme::ACCENT));
        }
        if let LayerKind::Group(g) = &l.kind
            && g.expanded {
                rows(ui, &g.children, depth + 1, active, editing_mask, renaming, out);
            }
    }
}

pub fn ui(s: &mut AppState, ui: &mut Ui) {
    let Some(d) = s.doc() else { return super::no_doc(ui) };
    let doc = d.doc().clone();
    let active = doc.active_layer;
    // Active layer blend / opacity.
    if let Some(l) = doc.active() {
        ui.horizontal(|ui| {
            let mut blend = l.blend;
            ComboBox::from_id_salt("blend").selected_text(blend.name()).width(120.0).show_ui(ui, |ui| {
                for m in BlendMode::ALL {
                    ui.selectable_value(&mut blend, m, m.name());
                }
            });
            if blend != l.blend {
                let id = l.id;
                s.doc_mut().unwrap().edit("Blend Mode", |d| d.layer_mut(id).unwrap().blend = blend);
            }
            let mut op = l.opacity * 100.0;
            let r = ui.add(egui::DragValue::new(&mut op).range(0.0..=100.0).suffix("%").prefix("Opacity "));
            if r.changed() {
                let id = l.id;
                s.doc_mut().unwrap().edit_merge("Opacity", 0x0ac1 ^ id.0, |d| d.layer_mut(id).unwrap().opacity = op / 100.0);
            }
            if r.drag_stopped() {
                s.doc_mut().unwrap().history.seal();
            }
        });
        let mut lock = l.lock;
        ui.horizontal(|ui| {
            ui.label(RichText::new("Lock").small().color(theme::TEXT_DIM));
            ui.toggle_value(&mut lock.pixels, "Pixels");
            ui.toggle_value(&mut lock.alpha, "Alpha");
            ui.toggle_value(&mut lock.position, "Position");
        });
        if lock != l.lock {
            let id = l.id;
            s.doc_mut().unwrap().edit("Lock", |d| d.layer_mut(id).unwrap().lock = lock);
        }
        ui.separator();
    }
    let mut actions = vec![];
    let renaming_id = ui.id().with("renaming");
    let mut renaming: Option<(LayerId, String)> = ui.data(|d| d.get_temp(renaming_id)).flatten();
    let editing_mask = s.settings.tools.edit_mask;
    egui::ScrollArea::vertical().max_height(ui.available_height() - 34.0).auto_shrink([false, true]).show(ui, |ui| {
        rows(ui, &doc.layers, 0, active, editing_mask, &mut renaming, &mut actions);
    });
    ui.data_mut(|d| d.insert_temp(renaming_id, renaming));
    for a in actions {
        let edit_mask = s.settings.tools.edit_mask;
        let d = s.docs.get_mut(s.active).unwrap();
        match a {
            RowAction::Select(id) => {
                d.set_active_layer(id);
                if edit_mask && d.doc().layer(id).is_some_and(|l| l.mask.is_none()) {
                    s.settings.tools.edit_mask = false;
                }
            }
            RowAction::ToggleVisible(id) => d.edit("Toggle Visibility", |d| {
                let l = d.layer_mut(id).unwrap();
                l.visible = !l.visible;
            }),
            RowAction::ToggleExpand(id) => {
                let mut doc = d.doc().clone();
                if let Some(LayerKind::Group(g)) = doc.layer_mut(id).map(|l| &mut l.kind) {
                    g.expanded = !g.expanded;
                }
                d.history.replace_current(doc);
                d.touch();
            }
            RowAction::Rename(id, name) => d.edit("Rename Layer", |d| d.layer_mut(id).unwrap().name = name),
            RowAction::Drop { dragged, target, into } => {
                if dragged != target {
                    d.edit("Reorder Layers", |d| {
                        d.reparent_layer(dragged, target, into);
                    });
                }
            }
            RowAction::EditMask(id, on) => {
                d.set_active_layer(id);
                s.settings.tools.edit_mask = on;
            }
            RowAction::ToggleMask(id) => d.edit("Toggle Mask", |d| {
                if let Some(m) = &mut d.layer_mut(id).unwrap().mask {
                    m.enabled = !m.enabled;
                }
            }),
        }
    }
    ui.separator();
    ui.horizontal(|ui| {
        if ui.button(ph::PLUS).on_hover_text("New layer").clicked() {
            cmd::new_layer(s);
        }
        if ui.button(ph::FOLDER_PLUS).on_hover_text("Group layer").clicked() {
            cmd::group_layer(s);
        }
        ui.menu_button(ph::GRADIENT, |ui| {
            if ui.button("Solid colour fill layer").clicked() {
                cmd::new_fill_layer(s, false);
                ui.close();
            }
            if ui.button("Gradient fill layer").clicked() {
                cmd::new_fill_layer(s, true);
                ui.close();
            }
        })
        .response
        .on_hover_text("New fill layer");
        if ui.button(ph::SQUARE_HALF).on_hover_text("Add layer mask (from selection if any)").clicked() {
            cmd::add_mask(s, true);
        }
        if ui.button(ph::COPY).on_hover_text("Duplicate layer").clicked() {
            cmd::duplicate_layer(s);
        }
        if ui.button(ph::ARROW_LINE_DOWN).on_hover_text("Merge down").clicked() {
            cmd::merge_down(s);
        }
        if ui.button(ph::TRASH).on_hover_text("Delete layer").clicked() {
            cmd::delete_layer(s);
        }
    });
}
