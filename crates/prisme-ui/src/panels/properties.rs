//! Properties: the selected node's parameters (generated from the node registry schema)
//! and the active layer's type-specific settings (text, fill, vector).

use crate::app::AppState;
use crate::theme;
use crate::widgets::{color_button, curves::curve_editor, gradient::gradient_editor, levels, wheel3::offset_wheel};
use egui::{ComboBox, DragValue, RichText, Ui};
use prisme_core::gradient::{GradientGeometry, GradientKind};
use prisme_core::layer::{FillContent, LayerKind, TextAlign, TextFill, TextGlow, TextLayer, TextOutline, TextShadow, TextWarp};
use prisme_core::{Document, Gradient, LayerId, NodeId, ParamValue};
use prisme_graph::{registry, ParamKind, CURVE_CHANNELS, LEVEL_CHANNELS};
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

fn key_of(parts: impl Hash) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    parts.hash(&mut h);
    h.finish()
}

pub fn ui(s: &mut AppState, ui: &mut Ui) {
    let Some(d) = s.doc() else { return super::no_doc(ui) };
    let doc = d.doc().clone();
    let selected = d.selected_node.filter(|n| doc.graph.nodes.contains_key(n));
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        if let Some(n) = selected {
            node_params(s, ui, &doc, n);
            ui.separator();
        }
        if let Some(l) = doc.active() {
            let id = l.id;
            match &l.kind {
                LayerKind::Text(t) => text_props(s, ui, id, t.clone()),
                LayerKind::Fill(f) => fill_props(s, ui, id, f.content.clone()),
                LayerKind::Vector(v) => {
                    super::section(ui, "Vector layer (SVG)");
                    ui.label(format!("Natural size: {:.0} × {:.0}", v.size[0], v.size[1]));
                    ui.label(RichText::new("Stays vector until render; scales without loss. Use Free Transform to place it.").small().color(theme::TEXT_DIM));
                    if ui.button("Rasterize layer").clicked() {
                        crate::commands::rasterize_layer(s);
                    }
                }
                LayerKind::Raster(g) => {
                    super::section(ui, "Raster layer");
                    ui.label(format!("Storage: {}", if g.format == prisme_core::PixelFormat::F32 { "32-bit float" } else { "16-bit float" }));
                    ui.label(format!("Tiles: {} ({:.1} MB)", g.tiles.len(), g.byte_size() as f64 / 1e6));
                }
                LayerKind::Group(g) => {
                    super::section(ui, "Group");
                    ui.label(format!("{} layers", g.children.len()));
                }
            }
        } else if selected.is_none() {
            ui.label(RichText::new("Select a layer or a node to edit its properties.").color(theme::TEXT_DIM));
        }
        ui.separator();
        super::section(ui, "Document");
        let size = if doc.info.is_infinite() { "Infinite canvas".to_string() } else { format!("{} × {} px", doc.info.width, doc.info.height) };
        ui.label(format!("{size}  ·  {} layers", doc.all_layers().len()));
        if let Some(p) = &doc.info.source_profile {
            ui.label(RichText::new(format!("Source colour: {p}")).small().color(theme::TEXT_DIM));
        }
        ui.label(RichText::new("Working space: linear Rec.2020, float").small().color(theme::TEXT_DIM));
    });
}

// ------------------------------------------------------------------ nodes

fn node_params(s: &mut AppState, ui: &mut Ui, doc: &Document, nid: NodeId) {
    let node = doc.graph.nodes[&nid].clone();
    let Some(desc) = registry().get(&node.kind) else { return };
    ui.horizontal(|ui| {
        ui.label(RichText::new(desc.name).strong().size(15.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let d = s.doc_mut().unwrap();
            let preview = d.preview_node == Some(nid);
            if ui.selectable_label(preview, format!("{} Preview", egui_phosphor::regular::EYE)).on_hover_text("Show this node's output on the canvas").clicked() {
                d.preview_node = if preview { None } else { Some(nid) };
                d.touch();
            }
            let mut bypass = node.bypass;
            if ui.checkbox(&mut bypass, "Bypass").changed() {
                d.edit("Bypass Node", |doc| doc.graph.nodes.get_mut(&nid).unwrap().bypass = bypass);
            }
        });
    });
    ui.label(RichText::new(desc.description).small().color(theme::TEXT_DIM));
    let mut params = node.params.clone();
    let mut changed = false;
    let hist = s.gpu.as_ref().and_then(|g| g.scopes.latest.as_ref().map(|d| d.hist.clone()));
    for pd in &desc.params {
        changed |= param_ui(s, ui, doc, nid, pd.key, pd.label, &pd.kind, &mut params, hist.as_deref());
    }
    if changed {
        let d = s.doc_mut().unwrap();
        d.edit_merge(&format!("Edit {}", desc.name), key_of(("node", nid.0)), |doc| {
            if let Some(n) = doc.graph.nodes.get_mut(&nid) {
                n.params = params;
            }
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn param_ui(
    s: &mut AppState,
    ui: &mut Ui,
    doc: &Document,
    nid: NodeId,
    key: &str,
    label: &str,
    kind: &ParamKind,
    params: &mut BTreeMap<String, ParamValue>,
    hist: Option<&[[u32; 256]]>,
) -> bool {
    let id = ui.id().with(("param", nid.0, key));
    match kind {
        ParamKind::Float { min, max, default, logarithmic } => {
            let mut v = params.get(key).map(|p| p.as_f32()).unwrap_or(*default);
            let r = crate::widgets::slider_row(ui, label, &mut v, *min..=*max, *default, *logarithmic);
            if r.changed() {
                params.insert(key.into(), ParamValue::Float(v));
                return true;
            }
        }
        ParamKind::Int { min, max, default } => {
            let mut v = match params.get(key) {
                Some(ParamValue::Int(i)) => *i,
                _ => *default,
            };
            let changed = ui.horizontal(|ui| {
                ui.add_sized([110.0, 18.0], egui::Label::new(RichText::new(label).color(theme::TEXT_DIM)));
                ui.add(egui::Slider::new(&mut v, *min..=*max)).changed()
            }).inner;
            if changed {
                params.insert(key.into(), ParamValue::Int(v));
                return true;
            }
        }
        ParamKind::Bool(default) => {
            let mut v = match params.get(key) {
                Some(ParamValue::Bool(b)) => *b,
                _ => *default,
            };
            if ui.checkbox(&mut v, label).changed() {
                params.insert(key.into(), ParamValue::Bool(v));
                return true;
            }
        }
        ParamKind::Color(default) => {
            let mut c = match params.get(key) {
                Some(ParamValue::Color(c)) => *c,
                _ => *default,
            };
            let changed = ui.horizontal(|ui| {
                ui.add_sized([110.0, 18.0], egui::Label::new(RichText::new(label).color(theme::TEXT_DIM)));
                color_button(ui, id, &mut c, true)
            }).inner;
            if changed {
                params.insert(key.into(), ParamValue::Color(c));
                return true;
            }
        }
        ParamKind::Choice { options, default } => {
            let mut v = match params.get(key) {
                Some(ParamValue::Choice(c)) => *c,
                _ => *default,
            };
            let before = v;
            ui.horizontal(|ui| {
                ui.add_sized([110.0, 18.0], egui::Label::new(RichText::new(label).color(theme::TEXT_DIM)));
                ComboBox::from_id_salt(id).selected_text(options.get(v as usize).copied().unwrap_or("?")).show_ui(ui, |ui| {
                    for (i, o) in options.iter().enumerate() {
                        ui.selectable_value(&mut v, i as u32, *o);
                    }
                });
            });
            if v != before {
                params.insert(key.into(), ParamValue::Choice(v));
                return true;
            }
        }
        ParamKind::Vec2(default) => {
            let mut v = match params.get(key) {
                Some(ParamValue::Vec2(c)) => *c,
                _ => *default,
            };
            let changed = ui.horizontal(|ui| {
                ui.add_sized([110.0, 18.0], egui::Label::new(RichText::new(label).color(theme::TEXT_DIM)));
                let a = ui.add(DragValue::new(&mut v[0]).prefix("x ").speed(1.0)).changed();
                let b = ui.add(DragValue::new(&mut v[1]).prefix("y ").speed(1.0)).changed();
                a || b
            }).inner;
            if changed {
                params.insert(key.into(), ParamValue::Vec2(v));
                return true;
            }
        }
        ParamKind::Wheel => {
            // Wheels are laid out side by side by the caller pattern: render inline.
            let mut v = match params.get(key) {
                Some(ParamValue::Vec3(c)) => *c,
                _ => [0.0; 3],
            };
            if offset_wheel(ui, label, &mut v, 110.0) {
                params.insert(key.into(), ParamValue::Vec3(v));
                return true;
            }
        }
        ParamKind::Curve => {
            let mut c = match params.get(key) {
                Some(ParamValue::Curve(c)) => c.clone(),
                _ => vec![[0.0, 0.0], [1.0, 1.0]],
            };
            if curve_editor(ui, id, &mut c, theme::TEXT, hist.map(|h| &h[3])) {
                params.insert(key.into(), ParamValue::Curve(c));
                return true;
            }
        }
        ParamKind::Gradient => {
            let mut g = match params.get(key) {
                Some(ParamValue::Gradient(g)) => g.clone(),
                _ => Gradient::default(),
            };
            ui.label(RichText::new(label).color(theme::TEXT_DIM));
            if gradient_editor(ui, id, &mut g) {
                params.insert(key.into(), ParamValue::Gradient(g));
                return true;
            }
        }
        ParamKind::Layer => {
            let cur = match params.get(key) {
                Some(ParamValue::Layer(l)) => *l,
                _ => None,
            };
            let mut v = cur;
            ui.horizontal(|ui| {
                ui.add_sized([110.0, 18.0], egui::Label::new(RichText::new(label).color(theme::TEXT_DIM)));
                let name = v.and_then(|id| doc.layer(id)).map(|l| l.name.clone()).unwrap_or("— choose —".into());
                ComboBox::from_id_salt(id).selected_text(name).show_ui(ui, |ui| {
                    for l in doc.all_layers().into_iter().rev() {
                        ui.selectable_value(&mut v, Some(l.id), &l.name);
                    }
                });
            });
            if v != cur {
                params.insert(key.into(), ParamValue::Layer(v));
                return true;
            }
        }
        ParamKind::LayerSet => {
            let mut set = match params.get(key) {
                Some(ParamValue::LayerSet(v)) => v.clone(),
                _ => vec![],
            };
            ui.label(RichText::new("Included layers").color(theme::TEXT_DIM));
            let mut changed = false;
            for l in doc.layers.iter().rev() {
                let mut on = !set.contains(&l.id);
                if ui.checkbox(&mut on, &l.name).changed() {
                    if on {
                        set.retain(|x| *x != l.id);
                    } else {
                        set.push(l.id);
                    }
                    changed = true;
                }
            }
            if changed {
                params.insert(key.into(), ParamValue::LayerSet(set));
                return true;
            }
        }
        ParamKind::Mask => {
            let cur = match params.get(key) {
                Some(ParamValue::Mask(m)) => *m,
                _ => None,
            };
            let mut v = cur;
            ui.horizontal(|ui| {
                ui.add_sized([110.0, 18.0], egui::Label::new(RichText::new(label).color(theme::TEXT_DIM)));
                let name = v.and_then(|id| doc.saved_mask(id)).map(|m| m.name.clone()).unwrap_or("— choose —".into());
                ComboBox::from_id_salt(id).selected_text(name).show_ui(ui, |ui| {
                    for m in &doc.masks {
                        ui.selectable_value(&mut v, Some(m.id), &m.name);
                    }
                });
            });
            if v != cur {
                params.insert(key.into(), ParamValue::Mask(v));
                return true;
            }
        }
        ParamKind::Text => {
            let txt = match params.get(key) {
                Some(ParamValue::Text(t)) => t.clone(),
                _ => String::new(),
            };
            ui.horizontal(|ui| {
                ui.add_sized([110.0, 18.0], egui::Label::new(RichText::new(label).color(theme::TEXT_DIM)));
                match prisme_graph::parse_cube(&txt) {
                    Some((n, _)) => ui.label(format!("{n}³ LUT loaded")),
                    None => ui.label(RichText::new("none").color(theme::TEXT_DIM)),
                };
                if ui.button("Load .cube…").clicked() {
                    crate::jobs::load_lut(s, nid, key.to_string());
                }
            });
        }
        ParamKind::Custom("levels") => {
            let ch_id = id.with("ch");
            let mut ch: usize = ui.data(|d| d.get_temp(ch_id)).unwrap_or(0);
            ui.horizontal(|ui| {
                for (i, n) in ["RGB", "Red", "Green", "Blue"].iter().enumerate() {
                    ui.selectable_value(&mut ch, i, *n);
                }
            });
            ui.data_mut(|d| d.insert_temp(ch_id, ch));
            let c = LEVEL_CHANNELS[ch];
            let mut vals: Vec<f32> = ["in_black", "in_white", "gamma", "out_black", "out_white"]
                .iter()
                .map(|k| params.get(&format!("{c}.{k}")).map(|p| p.as_f32()).unwrap_or(if matches!(*k, "in_white" | "gamma" | "out_white") { 1.0 } else { 0.0 }))
                .collect();
            let tint = [theme::TEXT, egui::Color32::from_rgb(255, 90, 90), egui::Color32::from_rgb(90, 230, 110), egui::Color32::from_rgb(100, 150, 255)][ch];
            let h = hist.map(|h| if ch == 0 { &h[3] } else { &h[ch - 1] });
            let (a, rest) = vals.split_at_mut(1);
            let (b, rest) = rest.split_at_mut(1);
            let (g, rest) = rest.split_at_mut(1);
            let (ob, ow) = rest.split_at_mut(1);
            let changed = levels::levels_editor(
                ui,
                id.with(ch),
                levels::LevelsValue { in_black: &mut a[0], in_white: &mut b[0], gamma: &mut g[0], out_black: &mut ob[0], out_white: &mut ow[0] },
                h,
                tint,
            );
            if changed {
                for (k, v) in ["in_black", "in_white", "gamma", "out_black", "out_white"].iter().zip(&vals) {
                    params.insert(format!("{c}.{k}"), ParamValue::Float(*v));
                }
                return true;
            }
        }
        ParamKind::Custom("curves") => {
            let ch_id = id.with("ch");
            let mut ch: usize = ui.data(|d| d.get_temp(ch_id)).unwrap_or(0);
            ui.horizontal(|ui| {
                for (i, n) in ["RGB", "Red", "Green", "Blue", "Lightness"].iter().enumerate() {
                    ui.selectable_value(&mut ch, i, *n);
                }
            });
            ui.data_mut(|d| d.insert_temp(ch_id, ch));
            let k = format!("curve.{}", CURVE_CHANNELS[ch]);
            let mut c = match params.get(&k) {
                Some(ParamValue::Curve(c)) => c.clone(),
                _ => vec![[0.0, 0.0], [1.0, 1.0]],
            };
            let tint = [theme::TEXT, egui::Color32::from_rgb(255, 90, 90), egui::Color32::from_rgb(90, 230, 110), egui::Color32::from_rgb(100, 150, 255), theme::ACCENT][ch];
            let h = hist.map(|h| if ch == 0 || ch == 4 { &h[3] } else { &h[ch - 1] });
            let changed = curve_editor(ui, id.with(ch), &mut c, tint, h);
            if ui.small_button("Reset channel").clicked() {
                c = vec![[0.0, 0.0], [1.0, 1.0]];
                params.insert(k, ParamValue::Curve(c));
                return true;
            }
            if changed {
                params.insert(k, ParamValue::Curve(c));
                return true;
            }
        }
        ParamKind::Custom(_) => {}
    }
    false
}

// ------------------------------------------------------------------ layers

fn text_props(s: &mut AppState, ui: &mut Ui, id: LayerId, t: TextLayer) {
    let mut t2 = t.clone();
    super::section(ui, "Text");
    ui.add(egui::TextEdit::multiline(&mut t2.text).desired_rows(3).desired_width(f32::INFINITY));
    ui.horizontal(|ui| {
        ui.label("Font");
        let fams = s.fonts.families();
        let search_id = ui.id().with("font-search");
        let mut search: String = ui.data(|d| d.get_temp(search_id)).unwrap_or_default();
        // Stays open while typing in the search field; picking a font closes it.
        ComboBox::from_id_salt("font")
            .selected_text(&t2.font_family)
            .height(320.0)
            .width(170.0)
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .show_ui(ui, |ui| {
                let r = ui.add(egui::TextEdit::singleline(&mut search).hint_text(format!("{} Search fonts", egui_phosphor::regular::MAGNIFYING_GLASS)));
                if !r.has_focus() && search.is_empty() {
                    r.request_focus();
                }
                let q = search.to_lowercase();
                for f in ["sans-serif", "serif", "monospace"].iter().map(|s| s.to_string()).chain(fams.into_iter()) {
                    if (q.is_empty() || f.to_lowercase().contains(&q)) && ui.selectable_value(&mut t2.font_family, f.clone(), f).clicked() {
                        ui.close();
                    }
                }
            });
        ui.data_mut(|d| d.insert_temp(search_id, search));
        if ui.button(egui_phosphor::regular::PLUS).on_hover_text("Load a font file (TTF/OTF)").clicked() {
            crate::jobs::load_font(s);
        }
    });
    ui.horizontal(|ui| {
        ui.label("Size");
        ui.add(DragValue::new(&mut t2.size).range(1.0..=4000.0).speed(0.5).suffix(" px"));
        ui.label("Weight");
        ComboBox::from_id_salt("weight").selected_text(t2.weight.to_string()).width(60.0).show_ui(ui, |ui| {
            for w in [100u16, 200, 300, 400, 500, 600, 700, 800, 900] {
                ui.selectable_value(&mut t2.weight, w, w.to_string());
            }
        });
        ui.toggle_value(&mut t2.italic, RichText::new("I").italics());
    });
    ui.horizontal(|ui| {
        ui.label("Line");
        ui.add(DragValue::new(&mut t2.line_height).range(0.5..=4.0).speed(0.01));
        ui.label("Tracking");
        ui.add(DragValue::new(&mut t2.letter_spacing).range(-50.0..=200.0).speed(0.1));
        for (a, n) in [(TextAlign::Left, egui_phosphor::regular::TEXT_ALIGN_LEFT), (TextAlign::Center, egui_phosphor::regular::TEXT_ALIGN_CENTER), (TextAlign::Right, egui_phosphor::regular::TEXT_ALIGN_RIGHT)] {
            ui.selectable_value(&mut t2.align, a, n);
        }
    });
    ui.horizontal(|ui| {
        ui.label("Rotation");
        let mut deg = t2.rotation.to_degrees();
        if ui.add(DragValue::new(&mut deg).range(-180.0..=180.0).suffix("°")).changed() {
            t2.rotation = deg.to_radians();
        }
        ui.label("Position");
        ui.add(DragValue::new(&mut t2.position[0]).prefix("x "));
        ui.add(DragValue::new(&mut t2.position[1]).prefix("y "));
    });

    super::section(ui, "Fill");
    let mut is_grad = matches!(t2.fill, TextFill::Gradient { .. });
    ui.horizontal(|ui| {
        if ui.selectable_label(!is_grad, "Solid").clicked() && is_grad {
            t2.fill = TextFill::Solid(s.settings.fg);
            is_grad = false;
        }
        if ui.selectable_label(is_grad, "Gradient").clicked() && !is_grad {
            t2.fill = TextFill::Gradient { gradient: Gradient::two(s.settings.fg, s.settings.bg), angle: 0.0 };
        }
    });
    match &mut t2.fill {
        TextFill::Solid(c) => {
            color_button(ui, ui.id().with("tfill"), c, true);
        }
        TextFill::Gradient { gradient, angle } => {
            gradient_editor(ui, ui.id().with("tgrad"), gradient);
            let mut deg = angle.to_degrees();
            if ui.add(egui::Slider::new(&mut deg, -180.0..=180.0).text("Angle")).changed() {
                *angle = deg.to_radians();
            }
        }
    }

    super::section(ui, "Effects");
    let mut has = t2.outline.is_some();
    if ui.checkbox(&mut has, "Outline").changed() {
        t2.outline = has.then_some(TextOutline { width: 4.0, color: s.settings.bg });
    }
    if let Some(o) = &mut t2.outline {
        ui.horizontal(|ui| {
            ui.add(DragValue::new(&mut o.width).range(0.1..=200.0).suffix(" px"));
            color_button(ui, ui.id().with("tout"), &mut o.color, true);
        });
    }
    let mut has = t2.glow.is_some();
    if ui.checkbox(&mut has, "Glow").changed() {
        t2.glow = has.then_some(TextGlow { radius: 16.0, color: [1.0, 0.85, 0.3, 1.0], strength: 1.0, inner: false });
    }
    if let Some(g) = &mut t2.glow {
        ui.horizontal(|ui| {
            ui.add(DragValue::new(&mut g.radius).range(0.5..=400.0).prefix("r ").suffix(" px"));
            ui.add(DragValue::new(&mut g.strength).range(0.0..=4.0).speed(0.01).prefix("× "));
            color_button(ui, ui.id().with("tglow"), &mut g.color, true);
            ui.checkbox(&mut g.inner, "Inner");
        });
    }
    let mut has = t2.shadow.is_some();
    if ui.checkbox(&mut has, "Drop shadow").changed() {
        t2.shadow = has.then_some(TextShadow { offset: [6.0, 6.0], blur: 8.0, color: [0.0, 0.0, 0.0, 0.6] });
    }
    if let Some(sh) = &mut t2.shadow {
        ui.horizontal(|ui| {
            ui.add(DragValue::new(&mut sh.offset[0]).prefix("dx "));
            ui.add(DragValue::new(&mut sh.offset[1]).prefix("dy "));
            ui.add(DragValue::new(&mut sh.blur).range(0.0..=200.0).prefix("blur "));
            color_button(ui, ui.id().with("tshadow"), &mut sh.color, true);
        });
    }

    super::section(ui, "Warp");
    let names = ["None", "Arc", "Wave", "Bulge", "Flag"];
    let cur = match t2.warp {
        TextWarp::None => 0,
        TextWarp::Arc { .. } => 1,
        TextWarp::Wave { .. } => 2,
        TextWarp::Bulge { .. } => 3,
        TextWarp::Flag { .. } => 4,
    };
    let mut sel = cur;
    ComboBox::from_id_salt("warp").selected_text(names[cur]).show_ui(ui, |ui| {
        for (i, n) in names.iter().enumerate() {
            ui.selectable_value(&mut sel, i, *n);
        }
    });
    if sel != cur {
        t2.warp = match sel {
            1 => TextWarp::Arc { bend: 0.5 },
            2 => TextWarp::Wave { amplitude: t2.size * 0.2, wavelength: t2.size * 3.0 },
            3 => TextWarp::Bulge { amount: 0.4 },
            4 => TextWarp::Flag { amplitude: t2.size * 0.2, wavelength: t2.size * 3.0 },
            _ => TextWarp::None,
        };
    }
    match &mut t2.warp {
        TextWarp::Arc { bend } => {
            ui.add(egui::Slider::new(bend, -1.0..=1.0).text("Bend"));
        }
        TextWarp::Wave { amplitude, wavelength } | TextWarp::Flag { amplitude, wavelength } => {
            ui.add(egui::Slider::new(amplitude, 0.0..=500.0).text("Amplitude"));
            ui.add(egui::Slider::new(wavelength, 10.0..=4000.0).logarithmic(true).text("Wavelength"));
        }
        TextWarp::Bulge { amount } => {
            ui.add(egui::Slider::new(amount, -1.0..=1.0).text("Amount"));
        }
        TextWarp::None => {}
    }

    super::section(ui, "3D");
    let td = &mut t2.three_d;
    ui.checkbox(&mut td.enabled, "Extrude & wrap on sphere");
    if td.enabled {
        ui.add(egui::Slider::new(&mut td.depth, 0.0..=400.0).text("Depth"));
        ui.add(egui::Slider::new(&mut td.radius, 0.0..=5000.0).logarithmic(true).text("Sphere radius"));
        let mut lon = td.longitude.to_degrees();
        let mut lat = td.latitude.to_degrees();
        if ui.add(egui::Slider::new(&mut lon, -180.0..=180.0).text("Longitude")).changed() {
            td.longitude = lon.to_radians();
        }
        if ui.add(egui::Slider::new(&mut lat, -85.0..=85.0).text("Latitude")).changed() {
            td.latitude = lat.to_radians();
        }
        for (i, n) in ["Yaw", "Pitch", "Roll"].iter().enumerate() {
            let mut deg = td.rotation[i].to_degrees();
            if ui.add(egui::Slider::new(&mut deg, -180.0..=180.0).text(*n)).changed() {
                td.rotation[i] = deg.to_radians();
            }
        }
        ui.add(egui::Slider::new(&mut td.perspective, 0.0..=2.0).text("Perspective"));
        ui.horizontal(|ui| {
            ui.label("Sides");
            color_button(ui, ui.id().with("side"), &mut td.side_color, false);
            let mut grad = td.side_gradient.is_some();
            if ui.checkbox(&mut grad, "gradient").changed() {
                td.side_gradient = grad.then(|| Gradient::two(td.side_color, [0.0, 0.0, 0.0, 1.0]));
            }
        });
        if let Some(g) = &mut td.side_gradient {
            gradient_editor(ui, ui.id().with("sidegrad"), g);
        }
        ui.horizontal(|ui| {
            ui.checkbox(&mut td.show_sphere, "Show sphere");
            if td.show_sphere {
                color_button(ui, ui.id().with("sphere"), &mut td.sphere_color, false);
            }
        });
    }
    if t2 != t {
        s.doc_mut().unwrap().edit_merge("Edit Text", key_of(("text", id.0)), |d| {
            if let Some(LayerKind::Text(x)) = d.layer_mut(id).map(|l| &mut l.kind) {
                *x = t2;
            }
        });
    }
    ui.add_space(4.0);
    if ui.button("Rasterize layer").clicked() {
        crate::commands::rasterize_layer(s);
    }
}

fn fill_props(s: &mut AppState, ui: &mut Ui, id: LayerId, content: FillContent) {
    super::section(ui, "Fill layer");
    let mut c2 = content.clone();
    let is_grad = matches!(c2, FillContent::Gradient(..));
    ui.horizontal(|ui| {
        if ui.selectable_label(!is_grad, "Solid").clicked() && is_grad {
            c2 = FillContent::Solid(s.settings.fg);
        }
        if ui.selectable_label(is_grad, "Gradient").clicked() && !is_grad {
            let (w, h) = s.doc().map(|d| (d.doc().info.width as f32, d.doc().info.height as f32)).unwrap_or((512.0, 512.0));
            c2 = FillContent::Gradient(Gradient::two(s.settings.fg, s.settings.bg), GradientGeometry { start: [0.0, h * 0.5], end: [w, h * 0.5], ..Default::default() });
        }
    });
    match &mut c2 {
        FillContent::Solid(c) => {
            color_button(ui, ui.id().with("fillc"), c, true);
        }
        FillContent::Gradient(g, geo) => {
            gradient_editor(ui, ui.id().with("fillg"), g);
            ComboBox::from_id_salt("fillkind").selected_text(geo.kind.name()).show_ui(ui, |ui| {
                for k in GradientKind::ALL {
                    ui.selectable_value(&mut geo.kind, k, k.name());
                }
            });
            ui.horizontal(|ui| {
                ui.label("Start");
                ui.add(DragValue::new(&mut geo.start[0]));
                ui.add(DragValue::new(&mut geo.start[1]));
                ui.label("End");
                ui.add(DragValue::new(&mut geo.end[0]));
                ui.add(DragValue::new(&mut geo.end[1]));
            });
            ui.checkbox(&mut geo.reverse, "Reverse");
            ui.checkbox(&mut geo.dither, "Dither");
        }
    }
    if c2 != content {
        s.doc_mut().unwrap().edit_merge("Edit Fill", key_of(("fill", id.0)), |d| {
            if let Some(LayerKind::Fill(f)) = d.layer_mut(id).map(|l| &mut l.kind) {
                f.content = c2;
            }
        });
    }
}
