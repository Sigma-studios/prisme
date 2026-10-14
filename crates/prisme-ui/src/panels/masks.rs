//! Saved masks: store selections, edit them (paint/fill with a red quick-mask overlay),
//! load them back as selections, attach them to layers or use them in the node graph.

use crate::app::AppState;
use crate::theme;
use egui::{RichText, Ui};
use egui_phosphor::regular as ph;
use prisme_core::layer::LayerMask;
use prisme_core::{MaskId, SavedMask, SelectionOp};
use prisme_tools::ToolKind;
use std::sync::Arc;

/// Start/stop painting on a saved mask.
pub fn set_editing(s: &mut AppState, id: Option<MaskId>) {
    s.settings.tools.edit_saved_mask = id;
    if id.is_some() {
        s.settings.tools.edit_mask = false;
        if !matches!(s.tool, ToolKind::Brush | ToolKind::Eraser | ToolKind::Fill | ToolKind::Gradient) {
            s.set_tool(ToolKind::Brush);
        }
        s.toast("Editing mask — paint white to add, black to remove (Eraser removes). Click Done when finished.");
    }
    if let Some(d) = s.doc_mut() {
        d.touch();
    }
}

type GridOp<'a> = &'a dyn Fn(&prisme_core::TileGrid, Option<&prisme_core::TileGrid>) -> prisme_core::TileGrid;

fn edit_grid(d: &mut crate::doc::DocState, id: MaskId, label: &str, f: GridOp) {
    d.edit(label, |doc| {
        let sel = doc.selection.clone();
        if let Some(x) = doc.masks.iter_mut().find(|x| x.id == id) {
            let mut g = f(&x.mask, sel.as_deref());
            g.remove_clear_tiles();
            x.mask = Arc::new(g);
        }
    });
}

pub fn ui(s: &mut AppState, ui: &mut Ui) {
    let editing = s.settings.tools.edit_saved_mask;
    let Some(d) = s.doc_mut() else { return super::no_doc(ui) };
    let has_sel = d.doc().selection.is_some();
    let mut start_edit: Option<Option<MaskId>> = None;
    if let Some(mid) = editing {
        if let Some(m) = d.doc().saved_mask(mid) {
            let name = m.name.clone();
            egui::Frame::new().fill(theme::ACCENT_DIM.gamma_multiply(0.5)).corner_radius(5).inner_margin(6).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{} Editing “{name}”", ph::PENCIL_SIMPLE)).strong());
                    if ui.button("Done").clicked() {
                        start_edit = Some(None);
                    }
                });
                ui.label(RichText::new("Brush / Fill / Gradient paint the mask: white = inside, black = outside. Red tint = outside the mask.").small());
            });
        } else {
            start_edit = Some(None);
        }
    }
    ui.horizontal(|ui| {
        if ui.add_enabled(has_sel, egui::Button::new(format!("{} Save selection as mask", ph::PLUS))).clicked() {
            crate::commands::save_selection_mask(d);
        }
        if ui.button("New empty mask").clicked() {
            d.edit("New Mask", |doc| {
                let id = doc.alloc_mask_id();
                let n = doc.masks.len() + 1;
                doc.masks.push(SavedMask { id, name: format!("Mask {n}"), mask: Arc::new(prisme_core::mask::new_mask_grid()) });
            });
        }
    });
    ui.separator();
    let masks: Vec<SavedMask> = d.doc().masks.clone();
    if masks.is_empty() {
        ui.label(
            RichText::new("No saved masks.\nSave a selection (or create an empty mask) to reuse it: load it as a selection, attach it to a layer as its mask, or feed it into the node graph with a Saved Mask node.")
                .small()
                .color(theme::TEXT_DIM),
        );
    }
    // Infinite canvas: invert within the content and the mask itself.
    let (bounds, infinite) = (d.doc().extent(), d.doc().info.is_infinite());
    let mut export = None;
    for m in masks {
        let is_editing = editing == Some(m.id);
        egui::Frame::new().fill(if is_editing { theme::BG3 } else { theme::BG2 }).corner_radius(5).inner_margin(5).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(ph::SQUARE_HALF);
                let mut name = m.name.clone();
                if ui.add(egui::TextEdit::singleline(&mut name).desired_width(120.0)).lost_focus() && name != m.name {
                    d.edit("Rename Mask", |doc| {
                        if let Some(x) = doc.masks.iter_mut().find(|x| x.id == m.id) {
                            x.name = name.clone();
                        }
                    });
                }
                if ui
                    .selectable_label(is_editing, format!("{} Edit", ph::PENCIL_SIMPLE))
                    .on_hover_text("Paint on this mask with Brush / Eraser / Fill / Gradient")
                    .clicked()
                {
                    start_edit = Some(if is_editing { None } else { Some(m.id) });
                }
                if ui.small_button(ph::TRASH).on_hover_text("Delete mask").clicked() {
                    d.edit("Delete Mask", |doc| {
                        doc.masks.retain(|x| x.id != m.id);
                        doc.scrub_graph_refs();
                    });
                    if is_editing {
                        start_edit = Some(None);
                    }
                }
            });
            ui.horizontal_wrapped(|ui| {
                if ui.small_button("Select").on_hover_text("Load as the selection (⇧-click: add to selection)").clicked() {
                    let add = ui.input(|i| i.modifiers.shift);
                    d.edit("Load Mask", |doc| {
                        let grid = if add { prisme_core::mask::combine(doc.selection.as_deref(), &m.mask, SelectionOp::Add) } else { (*m.mask).clone() };
                        doc.selection = Some(Arc::new(grid));
                    });
                }
                if ui.small_button(format!("{} Export", ph::EXPORT)).on_hover_text("Load as the selection and export it (Export Selection…): saved masks work as reusable boards").clicked() {
                    export = Some(m.mask.clone());
                }
                if ui
                    .small_button("→ Layer mask")
                    .on_hover_text("Attach to the selected layer as its mask: white areas stay visible, black areas become transparent (non-destructive)")
                    .clicked()
                {
                    d.edit("Apply Mask to Layer", |doc| {
                        if let Some(l) = doc.active_mut() {
                            l.mask = Some(LayerMask { grid: m.mask.clone(), enabled: true, default_visible: false });
                        }
                    });
                }
                if ui.small_button("→ Node").on_hover_text("Add a Saved Mask node to the graph").clicked() {
                    d.edit("Add Mask Node", |doc| {
                        let mut p = crate::panels::nodes::smart_params("mask", doc);
                        p.insert("mask".into(), prisme_core::ParamValue::Mask(Some(m.id)));
                        doc.graph.add_node("mask", p, [-250.0, 200.0]);
                    });
                }
                if ui.small_button("Invert").clicked() {
                    edit_grid(d, m.id, "Invert Mask", &|g, _| prisme_core::mask::invert(Some(g), if infinite { bounds.union(&g.tile_bounds()) } else { bounds }));
                }
                if ui.add_enabled(has_sel, egui::Button::new("+ Sel").small()).on_hover_text("Add the current selection to this mask").clicked() {
                    edit_grid(d, m.id, "Add Selection to Mask", &|g, s| prisme_core::mask::combine(Some(g), s.unwrap(), SelectionOp::Add));
                }
                if ui.add_enabled(has_sel, egui::Button::new("− Sel").small()).on_hover_text("Remove the current selection from this mask").clicked() {
                    edit_grid(d, m.id, "Subtract Selection from Mask", &|g, s| prisme_core::mask::combine(Some(g), s.unwrap(), SelectionOp::Subtract));
                }
                if ui.small_button("Feather").on_hover_text("Soften the mask edge by 4 px").clicked() {
                    edit_grid(d, m.id, "Feather Mask", &|g, _| prisme_core::mask::feather(g, 4.0));
                }
                if ui.small_button("Clear").clicked() {
                    edit_grid(d, m.id, "Clear Mask", &|_, _| prisme_core::mask::new_mask_grid());
                }
            });
        });
    }
    if let Some(e) = start_edit {
        set_editing(s, e);
    }
    if let Some(m) = export {
        crate::commands::edit(s, "Load Mask", |doc| doc.selection = Some(m));
        crate::dialogs::open_export(s, true);
    }
}

/// Document copy whose graph tints everything outside the edited saved mask red
/// (quick-mask preview) on top of the current output.
pub fn overlay_doc(doc: &prisme_core::Document, mid: MaskId, show_nodes: bool) -> Option<prisme_core::Document> {
    doc.saved_mask(mid)?;
    let mut d = doc.clone();
    let g = &mut d.graph;
    let out = g.output;
    let base = if show_nodes { g.input(out, 0) } else { None };
    let base = match base {
        Some(b) => b,
        None => (g.add_node(prisme_core::graph::KIND_LAYER_STACK, prisme_graph::default_params(prisme_core::graph::KIND_LAYER_STACK), [0.0; 2]), 0),
    };
    let mut mp = prisme_graph::default_params("mask");
    mp.insert("mask".into(), prisme_core::ParamValue::Mask(Some(mid)));
    let m = g.add_node("mask", mp, [0.0; 2]);
    let inv = g.add_node("mask_invert", Default::default(), [0.0; 2]);
    let mut cp = prisme_graph::default_params("color");
    cp.insert("color".into(), prisme_core::ParamValue::Color([1.0, 0.05, 0.05, 1.0]));
    let col = g.add_node("color", cp, [0.0; 2]);
    let mut bp = prisme_graph::default_params("blend");
    bp.insert("opacity".into(), prisme_core::ParamValue::Float(0.5));
    let bl = g.add_node("blend", bp, [0.0; 2]);
    g.connect(m, 0, inv, 0);
    g.connect(base.0, base.1, bl, 0);
    g.connect(col, 0, bl, 1);
    g.connect(inv, 0, bl, 2);
    g.connect(bl, 0, out, 0);
    Some(d)
}
