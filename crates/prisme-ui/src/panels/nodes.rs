//! Node graph editor. An egui-snarl `Snarl<NodeId>` mirrors the document graph; edits
//! made in the editor are recorded and committed to the document history.

use crate::app::AppState;
use crate::theme;
use egui::{Color32, RichText, Ui};
use egui_snarl::ui::{PinInfo, SnarlStyle, SnarlViewer};
use egui_snarl::{InPin, NodeId as SNode, OutPin, Snarl};
use prisme_core::{Document, NodeId};
use prisme_graph::{registry, Category, SocketType};
use std::collections::HashMap;

#[derive(Default)]
pub struct NodeEditor {
    snarl: Snarl<NodeId>,
    map: HashMap<NodeId, SNode>,
    /// Graph positions as of the last sync (to tell graph edits from user drags).
    synced_pos: HashMap<NodeId, [f32; 2]>,
    wires: Vec<(NodeId, u16, NodeId, u16)>,
    /// Open right-click menu on a link: (from, out, to, in, screen position).
    link_menu: Option<(NodeId, u16, NodeId, u16, egui::Pos2)>,
}

enum Edit {
    Connect(NodeId, u16, NodeId, u16),
    Disconnect(NodeId, u16),
    DisconnectOut(NodeId, u16),
    Remove(NodeId),
    /// Add a node: optional upstream link (from node, from output, new node's input) and
    /// downstream link (to node, to input, new node's output).
    Add(String, [f32; 2], Option<(NodeId, u16, u16)>, Option<(NodeId, u16, u16)>),
    Bypass(NodeId),
    Preview(NodeId),
    Select(NodeId),
    Params(NodeId, std::collections::BTreeMap<String, prisme_core::ParamValue>),
    Collapse(NodeId, bool),
}

struct Viewer<'a> {
    doc: &'a Document,
    edits: Vec<Edit>,
    thumbs: HashMap<NodeId, (egui::TextureId, egui::Rect)>,
    preview: Option<NodeId>,
    selected: Option<NodeId>,
    /// Screen rects of nodes this frame (hit testing / tests).
    rects: Vec<(NodeId, egui::Rect)>,
    /// Link right-clicked this frame (from, out, to, in).
    link_click: Option<(NodeId, u16, NodeId, u16)>,
}

fn pin(ty: SocketType, connected: bool) -> PinInfo {
    let (c, shape) = match ty {
        SocketType::Image => (Color32::from_rgb(240, 190, 70), PinInfo::circle()),
        SocketType::Mask => (Color32::from_rgb(170, 170, 180), PinInfo::square()),
    };
    let _ = connected;
    shape.with_fill(c).with_wire_color(c)
}

/// Compact editor for one simple parameter shown on the node itself.
fn inline_param(ui: &mut Ui, doc: &Document, nid: NodeId, pd: &prisme_graph::ParamDesc, params: &mut std::collections::BTreeMap<String, prisme_core::ParamValue>) -> bool {
    use prisme_core::ParamValue as PV;
    use prisme_graph::ParamKind as PK;
    let id = ui.id().with(("inline", nid.0, pd.key));
    let label = |ui: &mut Ui| {
        ui.add_sized([72.0, 16.0], egui::Label::new(RichText::new(pd.label).small().color(theme::TEXT_DIM)).truncate());
    };
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        match &pd.kind {
            PK::Float { min, max, default, logarithmic } => {
                label(ui);
                let mut v = params.get(pd.key).map(|p| p.as_f32()).unwrap_or(*default);
                let r = ui.add(egui::Slider::new(&mut v, *min..=*max).logarithmic(*logarithmic).max_decimals(2).show_value(true));
                if r.double_clicked() {
                    v = *default;
                }
                if r.changed() || r.double_clicked() {
                    params.insert(pd.key.into(), PV::Float(v));
                    changed = true;
                }
            }
            PK::Int { min, max, default } => {
                label(ui);
                let mut v = match params.get(pd.key) {
                    Some(PV::Int(i)) => *i,
                    _ => *default,
                };
                if ui.add(egui::Slider::new(&mut v, *min..=*max)).changed() {
                    params.insert(pd.key.into(), PV::Int(v));
                    changed = true;
                }
            }
            PK::Bool(default) => {
                let mut v = match params.get(pd.key) {
                    Some(PV::Bool(b)) => *b,
                    _ => *default,
                };
                if ui.checkbox(&mut v, RichText::new(pd.label).small()).changed() {
                    params.insert(pd.key.into(), PV::Bool(v));
                    changed = true;
                }
            }
            PK::Choice { options, default } => {
                label(ui);
                let mut v = match params.get(pd.key) {
                    Some(PV::Choice(c)) => *c,
                    _ => *default,
                };
                let before = v;
                egui::ComboBox::from_id_salt(id).width(110.0).selected_text(options.get(v as usize).copied().unwrap_or("?")).show_ui(ui, |ui| {
                    for (i, o) in options.iter().enumerate() {
                        ui.selectable_value(&mut v, i as u32, *o);
                    }
                });
                if v != before {
                    params.insert(pd.key.into(), PV::Choice(v));
                    changed = true;
                }
            }
            PK::Color(default) => {
                label(ui);
                let mut c = match params.get(pd.key) {
                    Some(PV::Color(c)) => *c,
                    _ => *default,
                };
                if crate::widgets::color_button(ui, id, &mut c, true) {
                    params.insert(pd.key.into(), PV::Color(c));
                    changed = true;
                }
            }
            PK::Layer => {
                label(ui);
                let cur = match params.get(pd.key) {
                    Some(PV::Layer(l)) => *l,
                    _ => None,
                };
                let mut v = cur;
                let name = v.and_then(|id| doc.layer(id)).map(|l| l.name.clone()).unwrap_or("— choose —".into());
                egui::ComboBox::from_id_salt(id).width(110.0).selected_text(name).show_ui(ui, |ui| {
                    for l in doc.all_layers().into_iter().rev() {
                        ui.selectable_value(&mut v, Some(l.id), &l.name);
                    }
                });
                if v != cur {
                    params.insert(pd.key.into(), PV::Layer(v));
                    changed = true;
                }
            }
            PK::Mask => {
                label(ui);
                let cur = match params.get(pd.key) {
                    Some(PV::Mask(m)) => *m,
                    _ => None,
                };
                let mut v = cur;
                let name = v.and_then(|id| doc.saved_mask(id)).map(|m| m.name.clone()).unwrap_or("— choose —".into());
                egui::ComboBox::from_id_salt(id).width(110.0).selected_text(name).show_ui(ui, |ui| {
                    for m in &doc.masks {
                        ui.selectable_value(&mut v, Some(m.id), &m.name);
                    }
                });
                if v != cur {
                    params.insert(pd.key.into(), PV::Mask(v));
                    changed = true;
                }
            }
            _ => {}
        }
    });
    changed
}

fn out_type(doc: &Document, node: NodeId, idx: u16) -> SocketType {
    doc.graph.nodes.get(&node).and_then(|n| registry().get(&n.kind)).and_then(|d| d.outputs.get(idx as usize)).map(|s| s.ty).unwrap_or(SocketType::Image)
}

fn in_type(doc: &Document, node: NodeId, idx: u16) -> SocketType {
    doc.graph.nodes.get(&node).and_then(|n| registry().get(&n.kind)).and_then(|d| d.inputs.get(idx as usize)).map(|s| s.ty).unwrap_or(SocketType::Image)
}

/// Add-node menu by category. With `from` (an upstream output and its type) only nodes with
/// an input of that type are listed; with `to` (a downstream input) only nodes with a
/// matching output. Both together = insert into a link.
fn add_menu(ui: &mut Ui, edits: &mut Vec<Edit>, pos: [f32; 2], from: Option<(NodeId, u16, SocketType)>, to: Option<(NodeId, u16, SocketType)>) {
    let mut any = false;
    for cat in Category::ALL {
        let items: Vec<_> = registry()
            .by_category(cat)
            .filter_map(|k| {
                let ni = match from {
                    Some((_, _, ty)) => Some(k.inputs.iter().position(|s| s.ty == ty)? as u16),
                    None => None,
                };
                let no = match to {
                    Some((_, _, ty)) => Some(k.outputs.iter().position(|s| s.ty == ty)? as u16),
                    None => None,
                };
                Some((k, ni, no))
            })
            .collect();
        if items.is_empty() {
            continue;
        }
        any = true;
        ui.menu_button(cat.name(), |ui| {
            for (k, ni, no) in items {
                if ui.button(k.name).on_hover_text(k.description).clicked() {
                    let f = from.zip(ni).map(|((n, o, _), i)| (n, o, i));
                    let t = to.zip(no).map(|((n, i, _), o)| (n, i, o));
                    edits.push(Edit::Add(k.id.to_string(), pos, f, t));
                    ui.close();
                }
            }
        });
    }
    if !any {
        ui.label(RichText::new("No compatible nodes").color(theme::TEXT_DIM));
    }
}

impl SnarlViewer<NodeId> for Viewer<'_> {
    fn title(&mut self, node: &NodeId) -> String {
        self.doc.graph.nodes.get(node).and_then(|n| registry().get(&n.kind)).map(|d| d.name.to_string()).unwrap_or("?".into())
    }

    fn show_header(&mut self, node: SNode, _inputs: &[InPin], _outputs: &[OutPin], ui: &mut Ui, snarl: &mut Snarl<NodeId>) {
        let id = snarl[node];
        let Some(n) = self.doc.graph.nodes.get(&id) else { return };
        let title = self.title(&id);
        let foldable = !prisme_graph::main_params(&n.kind).is_empty() || prisme_graph::has_complex_params(&n.kind);
        if foldable {
            let icon = if n.collapsed { egui_phosphor::regular::CARET_RIGHT } else { egui_phosphor::regular::CARET_DOWN };
            if ui.add(egui::Button::new(RichText::new(icon).small()).frame(false)).on_hover_text(if n.collapsed { "Show parameters" } else { "Hide parameters" }).clicked() {
                self.edits.push(Edit::Collapse(id, !n.collapsed));
            }
        }
        let mut txt = RichText::new(title).strong();
        if n.bypass {
            txt = txt.strikethrough().color(theme::TEXT_DIM);
        }
        if self.selected == Some(id) {
            txt = txt.color(theme::ACCENT);
        }
        let mut label = n.kind.clone();
        if let Some(prisme_core::ParamValue::Layer(Some(l))) = n.params.get("layer")
            && let Some(layer) = self.doc.layer(*l) {
                label = layer.name.clone();
            }
        // Hover-only so drags on the title move the node (selection: see final_node_rect).
        ui.add(egui::Label::new(txt).selectable(false)).on_hover_text(label);
        if self.preview == Some(id) {
            ui.label(RichText::new(egui_phosphor::regular::EYE).color(theme::ACCENT));
        }
    }

    fn inputs(&mut self, node: &NodeId) -> usize {
        self.doc.graph.nodes.get(node).and_then(|n| registry().get(&n.kind)).map(|d| d.inputs.len()).unwrap_or(0)
    }

    fn outputs(&mut self, node: &NodeId) -> usize {
        self.doc.graph.nodes.get(node).and_then(|n| registry().get(&n.kind)).map(|d| d.outputs.len()).unwrap_or(0)
    }

    fn show_input(&mut self, p: &InPin, ui: &mut Ui, snarl: &mut Snarl<NodeId>) -> impl egui_snarl::ui::SnarlPin + 'static {
        let id = snarl[p.id.node];
        let desc = self.doc.graph.nodes.get(&id).and_then(|n| registry().get(&n.kind));
        let s = desc.and_then(|d| d.inputs.get(p.id.input)).cloned();
        match s {
            Some(s) => {
                ui.label(RichText::new(s.name).small().color(if s.optional { theme::TEXT_DIM } else { theme::TEXT }));
                pin(s.ty, !p.remotes.is_empty())
            }
            None => pin(SocketType::Image, false),
        }
    }

    fn show_output(&mut self, p: &OutPin, ui: &mut Ui, snarl: &mut Snarl<NodeId>) -> impl egui_snarl::ui::SnarlPin + 'static {
        let id = snarl[p.id.node];
        let desc = self.doc.graph.nodes.get(&id).and_then(|n| registry().get(&n.kind));
        let s = desc.and_then(|d| d.outputs.get(p.id.output)).cloned();
        match s {
            Some(s) => {
                ui.label(RichText::new(s.name).small());
                pin(s.ty, !p.remotes.is_empty())
            }
            None => pin(SocketType::Image, false),
        }
    }

    fn has_body(&mut self, node: &NodeId) -> bool {
        let kind = self.doc.graph.nodes.get(node).map(|n| n.kind.as_str()).unwrap_or("");
        self.thumbs.contains_key(node) || !prisme_graph::main_params(kind).is_empty() || prisme_graph::has_complex_params(kind)
    }

    fn show_body(&mut self, node: SNode, _i: &[InPin], _o: &[OutPin], ui: &mut Ui, snarl: &mut Snarl<NodeId>) {
        let id = snarl[node];
        let Some(n) = self.doc.graph.nodes.get(&id) else { return };
        ui.vertical(|ui| {
            if let Some((tex, uv)) = self.thumbs.get(&id) {
                let aspect = uv.width() / uv.height().max(1e-3);
                let h = 64.0;
                // Hover-only so the node can be dragged by its thumbnail.
                let r = ui.add(egui::Image::new((*tex, egui::vec2(h * aspect.clamp(0.4, 2.5), h))).uv(*uv));
                if r.contains_pointer() && ui.input(|i| i.pointer.button_double_clicked(egui::PointerButton::Primary)) {
                    self.edits.push(Edit::Preview(id));
                }
            }
            if n.collapsed {
                return;
            }
            ui.set_max_width(220.0);
            let mains = prisme_graph::main_params(&n.kind);
            let mut params = n.params.clone();
            let mut changed = false;
            for pd in &mains {
                changed |= inline_param(ui, self.doc, id, pd, &mut params);
            }
            if changed {
                self.edits.push(Edit::Params(id, params));
                self.edits.push(Edit::Select(id));
            }
            if prisme_graph::has_complex_params(&n.kind) && ui.small_button("More in Properties…").clicked() {
                self.edits.push(Edit::Select(id));
            }
        });
    }

    /// Clicking anywhere on a node (not consumed by a widget) selects it.
    fn final_node_rect(&mut self, node: SNode, rect: egui::Rect, ui: &mut Ui, snarl: &mut Snarl<NodeId>) {
        let to_screen = ui.ctx().layer_transform_to_global(ui.layer_id()).unwrap_or_default();
        self.rects.push((snarl[node], to_screen * rect));
        let clicked = ui.input(|i| i.pointer.primary_clicked());
        if clicked && ui.ctx().rect_contains_pointer(ui.layer_id(), rect) {
            self.edits.push(Edit::Select(snarl[node]));
        }
    }

    fn has_graph_menu(&mut self, _pos: egui::Pos2, _snarl: &mut Snarl<NodeId>) -> bool {
        true
    }

    fn show_graph_menu(&mut self, pos: egui::Pos2, ui: &mut Ui, _snarl: &mut Snarl<NodeId>) {
        ui.label(RichText::new("Add node").strong());
        add_menu(ui, &mut self.edits, [pos.x, pos.y], None, None);
    }

    fn has_dropped_wire_menu(&mut self, _src: egui_snarl::ui::AnyPins, _snarl: &mut Snarl<NodeId>) -> bool {
        true
    }

    fn show_dropped_wire_menu(&mut self, pos: egui::Pos2, ui: &mut Ui, src: egui_snarl::ui::AnyPins, snarl: &mut Snarl<NodeId>) {
        // Only offer nodes that can take (or feed) the dragged wire.
        let (from, to) = match src {
            egui_snarl::ui::AnyPins::Out(pins) => (
                pins.first().map(|p| {
                    let (n, o) = (snarl[p.node], p.output as u16);
                    (n, o, out_type(self.doc, n, o))
                }),
                None,
            ),
            egui_snarl::ui::AnyPins::In(pins) => (
                None,
                pins.first().map(|p| {
                    let (n, i) = (snarl[p.node], p.input as u16);
                    (n, i, in_type(self.doc, n, i))
                }),
            ),
        };
        ui.label(RichText::new("Add & connect").strong());
        add_menu(ui, &mut self.edits, [pos.x, pos.y], from, to);
    }


    fn has_node_menu(&mut self, _node: &NodeId) -> bool {
        true
    }

    fn show_node_menu(&mut self, node: SNode, _i: &[InPin], _o: &[OutPin], ui: &mut Ui, snarl: &mut Snarl<NodeId>) {
        let id = snarl[node];
        if ui.button("Preview on canvas").clicked() {
            self.edits.push(Edit::Preview(id));
            ui.close();
        }
        if ui.button("Toggle bypass").clicked() {
            self.edits.push(Edit::Bypass(id));
            ui.close();
        }
        if id != self.doc.graph.output && ui.button("Delete").clicked() {
            self.edits.push(Edit::Remove(id));
            ui.close();
        }
    }

    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<NodeId>) {
        self.edits.push(Edit::Connect(snarl[from.id.node], from.id.output as u16, snarl[to.id.node], to.id.input as u16));
    }

    /// Snarl reports a right-click on a link as "disconnect": show the link menu instead.
    fn disconnect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<NodeId>) {
        self.link_click = Some((snarl[from.id.node], from.id.output as u16, snarl[to.id.node], to.id.input as u16));
    }

    fn drop_inputs(&mut self, p: &InPin, snarl: &mut Snarl<NodeId>) {
        self.edits.push(Edit::Disconnect(snarl[p.id.node], p.id.input as u16));
    }

    fn drop_outputs(&mut self, p: &OutPin, snarl: &mut Snarl<NodeId>) {
        self.edits.push(Edit::DisconnectOut(snarl[p.id.node], p.id.output as u16));
    }
}

fn sync(ed: &mut NodeEditor, doc: &Document) {
    // nodes
    ed.map.retain(|id, sn| {
        if doc.graph.nodes.contains_key(id) {
            true
        } else {
            ed.snarl.remove_node(*sn);
            false
        }
    });
    for (id, n) in &doc.graph.nodes {
        match ed.map.get(id) {
            None => {
                let sn = ed.snarl.insert_node(egui::pos2(n.pos[0], n.pos[1]), *id);
                ed.map.insert(*id, sn);
            }
            Some(sn) => {
                // Position changed by a document edit (undo, insert, load): move the snarl node.
                if ed.synced_pos.get(id).is_some_and(|p| *p != n.pos) {
                    if let Some(info) = ed.snarl.get_node_info_mut(*sn) {
                        info.pos = egui::pos2(n.pos[0], n.pos[1]);
                    }
                }
            }
        }
        ed.synced_pos.insert(*id, n.pos);
    }
    ed.synced_pos.retain(|id, _| doc.graph.nodes.contains_key(id));
    // wires
    let mut want: Vec<(NodeId, u16, NodeId, u16)> = doc.graph.links.iter().map(|l| (l.from, l.from_out, l.to, l.to_in)).collect();
    want.sort();
    if want != ed.wires {
        let existing: Vec<_> = ed.snarl.wires().collect();
        for (o, i) in existing {
            ed.snarl.disconnect(o, i);
        }
        for (f, fo, t, ti) in &want {
            if let (Some(a), Some(b)) = (ed.map.get(f), ed.map.get(t)) {
                ed.snarl.connect(egui_snarl::OutPinId { node: *a, output: *fo as usize }, egui_snarl::InPinId { node: *b, input: *ti as usize });
            }
        }
        ed.wires = want;
    }
}

pub fn ui(s: &mut AppState, ui: &mut Ui) {
    let Some(d) = s.doc() else { return super::no_doc(ui) };
    let doc_id = d.id;
    let doc = d.doc().clone();
    let version = d.version;
    let preview = d.preview_node;
    let selected = d.selected_node;
    let show_nodes = d.show_nodes;

    // Toolbar.
    let mut toggle_nodes = false;
    ui.horizontal(|ui| {
        let mut on = show_nodes;
        if ui.toggle_value(&mut on, "Node output").on_hover_text("Show the graph output (off: raw layer stack, no nodes). Shortcut: \\").clicked() {
            toggle_nodes = true;
        }
        if preview.is_some() && ui.button("Stop preview").clicked() {
            s.docs[s.active].preview_node = None;
            s.docs[s.active].touch();
        }
        ui.label(RichText::new("Right-click empty space: add node · right-click a link: insert or disconnect · drag a pin into space: add a compatible node · double-click a thumbnail: preview").small().color(theme::TEXT_DIM));
    });
    if toggle_nodes {
        let d = &mut s.docs[s.active];
        d.show_nodes = !d.show_nodes;
        d.touch();
    }

    // Thumbnails (one refresh per frame, not while interacting).
    let mut thumbs = HashMap::new();
    if let Some(g) = s.gpu.as_mut() {
        let busy = ui.input(|i| i.pointer.any_down());
        let mut refreshed = false;
        for id in doc.graph.nodes.keys() {
            let key = (doc_id, *id);
            let stale = g.thumbs.get(&key).is_none_or(|t| t.0 != version);
            if stale && !busy && !refreshed && *id != doc.graph.output {
                refreshed = true;
                if let Some(img) = g.engine.thumbnail(&doc, *id, 96) {
                    let mut r = g.rs.renderer.write();
                    if let Some((_, old, _)) = g.thumbs.remove(&key) {
                        r.free_texture(&old);
                    }
                    let tid = r.register_native_texture(&g.rs.device, img.view(), egui_wgpu::wgpu::FilterMode::Linear);
                    g.thumbs.insert(key, (version, tid, img));
                }
            }
            if let Some((_, tid, img)) = g.thumbs.get(&key) {
                let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(img.rect.width() as f32 / img.tex.w as f32, img.rect.height() as f32 / img.tex.h as f32));
                thumbs.insert(*id, (*tid, uv));
            }
        }
        if refreshed {
            ui.ctx().request_repaint();
        }
    }

    let ed = s.node_editors.entry(doc_id).or_default();
    sync(ed, &doc);
    let mut viewer = Viewer { doc: &doc, edits: vec![], thumbs, preview, selected, rects: vec![], link_click: None };
    let style = SnarlStyle { collapsible: Some(false), pin_placement: Some(egui_snarl::ui::PinPlacement::Edge), ..SnarlStyle::new() };
    ed.snarl.show(&mut viewer, &style, ("prisme-snarl", doc_id), ui);

    // Node moves.
    let mut moved: Vec<(NodeId, [f32; 2])> = vec![];
    for (sn, pos, id) in ed.snarl.nodes_pos_ids() {
        let _ = sn;
        if let Some(n) = doc.graph.nodes.get(id)
            && ((n.pos[0] - pos.x).abs() > 0.5 || (n.pos[1] - pos.y).abs() > 0.5) {
                moved.push((*id, [pos.x, pos.y]));
            }
    }
    let mut edits = viewer.edits;
    if let Some((f, fo, t, ti)) = viewer.link_click {
        let p = ui.input(|i| i.pointer.interact_pos()).unwrap_or_default();
        ed.link_menu = Some((f, fo, t, ti, p));
    }
    link_menu(ui, ed, &doc, &mut edits);
    let d = &mut s.docs[s.active];
    if !moved.is_empty() {
        let mut nd = d.history.current().clone();
        for (id, p) in &moved {
            if let Some(n) = nd.graph.nodes.get_mut(id) {
                n.pos = *p;
            }
        }
        // Positions are cosmetic: update in place without a history entry.
        d.history.replace_current(nd);
        if let Some(ed) = s.node_editors.get_mut(&doc_id) {
            for (id, p) in &moved {
                ed.synced_pos.insert(*id, *p);
            }
        }
    }
    for e in edits {
        match e {
            Edit::Connect(f, fo, t, ti) => d.edit("Connect Nodes", |doc| {
                doc.graph.connect(f, fo, t, ti);
            }),
            Edit::Disconnect(t, ti) => d.edit("Disconnect", |doc| doc.graph.disconnect_input(t, ti)),
            Edit::DisconnectOut(f, fo) => d.edit("Disconnect", |doc| doc.graph.links.retain(|l| !(l.from == f && l.from_out == fo))),
            Edit::Remove(id) => {
                d.edit("Delete Node", |doc| {
                    doc.graph.remove_node(id);
                });
                if d.preview_node == Some(id) {
                    d.preview_node = None;
                }
                if d.selected_node == Some(id) {
                    d.selected_node = None;
                }
            }
            Edit::Add(kind, pos, from, to) => {
                let mut new_id = None;
                let label = if from.is_some() && to.is_some() { "Insert Node" } else { "Add Node" };
                d.edit(label, |doc| {
                    let params = smart_params(&kind, doc);
                    let id = doc.graph.add_node(&kind, params, pos);
                    if let Some((f, fo, ni)) = from {
                        doc.graph.connect(f, fo, id, ni);
                    }
                    // Connecting into the downstream input replaces the old link there.
                    if let Some((t, ti, no)) = to {
                        doc.graph.connect(id, no, t, ti);
                    }
                    new_id = Some(id);
                });
                d.selected_node = new_id;
            }
            Edit::Bypass(id) => d.edit("Bypass Node", |doc| {
                if let Some(n) = doc.graph.nodes.get_mut(&id) {
                    n.bypass = !n.bypass;
                }
            }),
            Edit::Preview(id) => {
                d.preview_node = if d.preview_node == Some(id) { None } else { Some(id) };
                d.touch();
            }
            Edit::Select(id) => d.selected_node = Some(id),
            Edit::Params(id, params) => {
                let key = {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    ("node", id.0).hash(&mut h);
                    h.finish()
                };
                let name = d.doc().graph.nodes.get(&id).and_then(|n| registry().get(&n.kind)).map(|k| k.name).unwrap_or("Node");
                d.edit_merge(&format!("Edit {name}"), key, |doc| {
                    if let Some(n) = doc.graph.nodes.get_mut(&id) {
                        n.params = params;
                    }
                });
            }
            Edit::Collapse(id, c) => {
                // UI state: no history entry.
                let mut nd = d.history.current().clone();
                if let Some(n) = nd.graph.nodes.get_mut(&id) {
                    n.collapsed = c;
                }
                d.history.replace_current(nd);
                if let Some(w) = &mut d.working {
                    if let Some(n) = w.graph.nodes.get_mut(&id) {
                        n.collapsed = c;
                    }
                }
                d.touch();
            }
        }
    }
}

/// Default parameters for a new node, pre-filled from the document wherever there is
/// a sensible choice (active layer, most recent saved mask…) instead of leaving them empty.
pub fn smart_params(kind: &str, doc: &Document) -> std::collections::BTreeMap<String, prisme_core::ParamValue> {
    use prisme_core::ParamValue as PV;
    let mut params = prisme_graph::default_params(kind);
    let Some(desc) = registry().get(kind) else { return params };
    let active = doc.active_layer.or_else(|| doc.all_layers().last().map(|l| l.id));
    let mask = doc.masks.last().map(|m| m.id);
    let (w, h) = (doc.info.width as f32, doc.info.height as f32);
    for pd in &desc.params {
        match (&pd.kind, params.get(pd.key)) {
            (prisme_graph::ParamKind::Layer, Some(PV::Layer(None))) if active.is_some() => {
                params.insert(pd.key.into(), PV::Layer(active));
            }
            (prisme_graph::ParamKind::Mask, Some(PV::Mask(None))) if mask.is_some() => {
                params.insert(pd.key.into(), PV::Mask(mask));
            }
            _ => {}
        }
    }
    // Spatial defaults that should span the document rather than a fixed 512 px.
    match kind {
        "gradient" => {
            params.insert("start".into(), PV::Vec2([0.0, h * 0.5]));
            params.insert("end".into(), PV::Vec2([w, h * 0.5]));
        }
        "crop" => {
            params.insert("min".into(), PV::Vec2([0.0, 0.0]));
            params.insert("max".into(), PV::Vec2([w, h]));
        }
        _ => {}
    }
    params
}

/// Right-click menu for a link: insert a compatible node into it, or disconnect.
fn link_menu(ui: &mut Ui, ed: &mut NodeEditor, doc: &Document, edits: &mut Vec<Edit>) {
    let Some((f, fo, t, ti, pos)) = ed.link_menu else { return };
    if !doc.graph.links.iter().any(|l| l.from == f && l.from_out == fo && l.to == t && l.to_in == ti) {
        ed.link_menu = None;
        return;
    }
    let before = edits.len();
    let mut close = false;
    let area = egui::Area::new(ui.id().with("link-menu")).order(egui::Order::Foreground).fixed_pos(pos).show(ui.ctx(), |ui| {
        egui::Frame::menu(ui.style()).show(ui, |ui| {
            ui.set_min_width(150.0);
            ui.label(RichText::new("Connection").small().color(theme::TEXT_DIM));
            let mid = match (doc.graph.nodes.get(&f), doc.graph.nodes.get(&t)) {
                (Some(a), Some(b)) => [(a.pos[0] + b.pos[0]) * 0.5, (a.pos[1] + b.pos[1]) * 0.5 + 60.0],
                _ => [0.0, 0.0],
            };
            ui.menu_button(format!("{} Insert node", egui_phosphor::regular::PLUS), |ui| {
                add_menu(ui, edits, mid, Some((f, fo, out_type(doc, f, fo))), Some((t, ti, in_type(doc, t, ti))));
            });
            if ui.button(format!("{} Disconnect", egui_phosphor::regular::LINK_BREAK)).clicked() {
                edits.push(Edit::Disconnect(t, ti));
                close = true;
            }
        });
    });
    // Clicks on our area or its submenu (both foreground layers) keep it open.
    let press = ui.input(|i| if i.pointer.any_pressed() { i.pointer.interact_pos() } else { None });
    let clicked_outside = press.is_some_and(|p| !area.response.rect.contains(p) && ui.ctx().layer_id_at(p).is_none_or(|l| l.order < egui::Order::Foreground));
    if close || edits.len() > before || clicked_outside || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        ed.link_menu = None;
    }
}

/// Insert a node of `kind` between the output and its current input (menu "Add node").
pub fn insert_before_output(d: &mut crate::doc::DocState, kind: &str) {
    let mut new_id = None;
    d.edit("Add Node", |doc| {
        let out = doc.graph.output;
        let prev = doc.graph.input(out, 0);
        let pos = doc.graph.nodes[&out].pos;
        let params = smart_params(kind, doc);
        let id = doc.graph.add_node(kind, params, pos);
        let desc = registry().get(kind);
        if let (Some((p, po)), Some(desc)) = (prev, desc)
            && !desc.inputs.is_empty() {
                doc.graph.connect(p, po, id, 0);
            }
        if desc.is_some_and(|d| !d.outputs.is_empty()) {
            doc.graph.connect(id, 0, out, 0);
        }
        if let Some(n) = doc.graph.nodes.get_mut(&out) {
            n.pos[0] += 220.0;
        }
        new_id = Some(id);
    });
    d.selected_node = new_id;
}

#[cfg(test)]
mod tests {
    use super::*;
    thread_local! { static RECTS: std::cell::RefCell<Vec<(NodeId, egui::Rect)>> = const { std::cell::RefCell::new(vec![]) }; }

    /// Run one editor frame against `doc` (positions written back like `ui()` does).
    fn frame(ctx: &egui::Context, ed: &mut NodeEditor, doc: &mut Document, events: Vec<egui::Event>) {
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1200.0, 800.0))),
            events,
            ..Default::default()
        };
        let mut out = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default().show_inside(ui, |ui| {
                sync(ed, doc);
                let snapshot = doc.clone();
                let thumbs = snapshot.graph.nodes.keys().map(|id| (*id, (egui::TextureId::Managed(0), egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0))))).collect();
                let mut viewer = Viewer { doc: &snapshot, edits: vec![], thumbs, preview: None, selected: None, rects: vec![], link_click: None };
                let style = SnarlStyle { collapsible: Some(false), pin_placement: Some(egui_snarl::ui::PinPlacement::Edge), ..SnarlStyle::new() };
                ed.snarl.show(&mut viewer, &style, ("prisme-snarl", 1u64), ui);
                RECTS.with(|r| *r.borrow_mut() = viewer.rects.clone());
                for (_, pos, id) in ed.snarl.nodes_pos_ids() {
                    if let Some(n) = doc.graph.nodes.get_mut(id) {
                        if (n.pos[0] - pos.x).abs() > 0.5 || (n.pos[1] - pos.y).abs() > 0.5 {
                            n.pos = [pos.x, pos.y];
                            ed.synced_pos.insert(*id, n.pos);
                        }
                    }
                }
            });
        });
        out.textures_delta.clear();
    }

    #[test]
    fn nodes_can_be_dragged() {
        let ctx = egui::Context::default();
        let mut ed = NodeEditor::default();
        let mut doc = Document::with_background("t", 64, 64, None);
        for _ in 0..3 {
            frame(&ctx, &mut ed, &mut doc, vec![]);
        }
        let before: Vec<[f32; 2]> = doc.graph.nodes.values().map(|n| n.pos).collect();
        let rects = RECTS.with(|r| r.borrow().clone());
        for (nid, r) in rects.iter().take(2) {
          for frac in [0.12f32, 0.6] {
            // Drag from the title (12%) and from the thumbnail/body (60%).
            let before: Vec<[f32; 2]> = doc.graph.nodes.values().map(|n| n.pos).collect();
            let r = RECTS.with(|x| x.borrow().iter().find(|(i, _)| i == nid).map(|(_, r)| *r)).unwrap_or(*r);
            let p = egui::pos2(r.center().x, r.top() + r.height() * frac);
            let q = p + egui::vec2(60.0, 30.0);
            let btn = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
            frame(&ctx, &mut ed, &mut doc, vec![egui::Event::PointerMoved(p), btn(p, true)]);
            for k in 1..=5 {
                frame(&ctx, &mut ed, &mut doc, vec![egui::Event::PointerMoved(p + (q - p) * (k as f32 / 5.0))]);
            }
            frame(&ctx, &mut ed, &mut doc, vec![btn(q, false)]);
            let after: Vec<[f32; 2]> = doc.graph.nodes.values().map(|n| n.pos).collect();
            assert_ne!(before, after, "drag at {frac} of node {nid:?} did not move it");
          }
        }
    }
}
