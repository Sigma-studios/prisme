//! Document commands, keyboard shortcut dispatch and menus.

use crate::actions::Action;
use crate::app::{AppState, Tab};
use crate::doc::DocState;
use egui::Ui;
use egui_dock::DockState;
use prisme_core::gradient::GradientGeometry;
use prisme_core::layer::{FillContent, FillLayer, GroupLayer, Layer, LayerKind, LayerMask};
use prisme_core::{Document, Gradient, IRect, SavedMask, TileGrid};
use prisme_tools::transform::{merge_over, split_by_selection, Resample};
use prisme_tools::{ToolCx, ToolKey, ToolKind, ToolOutcome};
use std::sync::Arc;

// ------------------------------------------------------------------ shortcuts

pub fn handle_shortcuts(s: &mut AppState, ctx: &egui::Context) {
    if ctx.input(|i| i.pointer.any_released()) {
        for d in &mut s.docs {
            d.history.seal();
        }
    }
    // On-canvas text editing captures the keyboard.
    if s.text_editing().is_some() && !ctx.egui_wants_keyboard_input() {
        let events = ctx.input(|i| i.events.clone());
        let mut keys = vec![];
        for e in events {
            match e {
                egui::Event::Text(t) => keys.push(ToolKey::Text(t)),
                egui::Event::Key { key, pressed: true, modifiers, .. } => {
                    if modifiers.command {
                        continue;
                    }
                    let k = match key {
                        egui::Key::Enter => ToolKey::Enter,
                        egui::Key::Escape => ToolKey::Escape,
                        egui::Key::Backspace => ToolKey::Backspace,
                        egui::Key::Delete => ToolKey::Delete,
                        egui::Key::ArrowLeft => ToolKey::Left,
                        egui::Key::ArrowRight => ToolKey::Right,
                        egui::Key::Home => ToolKey::Home,
                        egui::Key::End => ToolKey::End,
                        _ => continue,
                    };
                    keys.push(k);
                }
                _ => {}
            }
        }
        for k in keys {
            tool_key(s, &k);
        }
        // Still allow Cmd shortcuts (undo, save…).
        let actions: Vec<Action> = s.settings.keymap.triggered(ctx).into_iter().filter(|a| s.settings.keymap.get(*a).is_some_and(|b| b.command)).collect();
        for a in actions {
            run_action(s, a, ctx);
        }
        ctx.input_mut(|i| i.events.retain(|e| !matches!(e, egui::Event::Text(_) | egui::Event::Key { .. })));
        return;
    }
    if ctx.egui_wants_keyboard_input() {
        return;
    }
    // Keys the active tool may consume first.
    let tool_keys = [
        (egui::Key::Enter, ToolKey::Enter),
        (egui::Key::Escape, ToolKey::Escape),
        (egui::Key::Backspace, ToolKey::Backspace),
        (egui::Key::Delete, ToolKey::Delete),
        (egui::Key::ArrowLeft, ToolKey::Left),
        (egui::Key::ArrowRight, ToolKey::Right),
        (egui::Key::ArrowUp, ToolKey::Up),
        (egui::Key::ArrowDown, ToolKey::Down),
    ];
    for (key, tk) in tool_keys {
        let pressed = ctx.input(|i| i.key_pressed(key) && i.modifiers.is_none());
        if pressed && tool_key(s, &tk) {
            ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, key));
        }
    }
    for a in s.settings.keymap.triggered(ctx) {
        run_action(s, a, ctx);
    }
    if ctx.input(|i| i.key_pressed(egui::Key::Escape))
        && let Some(d) = s.doc_mut()
            && d.doc().selection.is_some() {
                d.edit("Deselect", |doc| doc.selection = None);
            }
}

pub fn script_key(s: &mut AppState, key: &ToolKey) {
    tool_key(s, key);
}

/// Send a key to the active tool. Returns true if it was handled.
fn tool_key(s: &mut AppState, key: &ToolKey) -> bool {
    let tool_kind = s.tool;
    let fg = s.settings.fg;
    let bg = s.settings.bg;
    let opts = s.settings.tools.clone();
    let Some(d) = s.docs.get_mut(s.active) else { return false };
    let Some(tool) = s.tools.get_mut(&tool_kind) else { return false };
    let had_working = d.working.is_some();
    if !had_working {
        d.working = Some(d.history.current().clone());
    }
    let out = {
        let mut cx = ToolCx { doc: d.working.as_mut().unwrap(), opts: &opts, fg, bg, strokes: None };
        tool.key(&mut cx, key)
    };
    let busy = tool.busy();
    let handled = out != ToolOutcome::None;
    if !handled && !had_working {
        d.working = None;
        return false;
    }
    crate::canvas::apply_outcome(s, out, busy);
    handled
}

pub fn run_action(s: &mut AppState, a: Action, ctx: &egui::Context) {
    // Actions see the document with any finished GPU stroke landed.
    if let Some(d) = s.doc_mut() {
        d.finish_stroke();
    }
    match a {
        Action::New => s.dialogs.new_doc.open = true,
        Action::Open => crate::jobs::open_dialog(s),
        Action::Save => crate::jobs::save(s, false),
        Action::SaveAs => crate::jobs::save(s, true),
        Action::Export => crate::dialogs::open_export(s),
        Action::Close => {
            let i = s.active;
            close_doc(s, i)
        }
        Action::Undo => {
            if let Some(d) = s.doc_mut() {
                d.undo();
            }
            reset_tools(s);
        }
        Action::Redo => {
            if let Some(d) = s.doc_mut() {
                d.redo();
            }
            reset_tools(s);
        }
        Action::Cut => {
            copy(s, false);
            clear_selection_pixels(s, "Cut");
        }
        Action::Copy => copy(s, false),
        Action::CopyMerged => copy(s, true),
        Action::Paste => paste(s, false),
        Action::PasteNewDoc => paste(s, true),
        Action::Delete => clear_selection_pixels(s, "Clear"),
        Action::FillFg => fill_selection(s, false),
        Action::FillBg => fill_selection(s, true),
        Action::SelectAll => edit(s, "Select All", |d| d.selection = Some(Arc::new(prisme_core::mask::select_all(d.bounds())))),
        Action::Deselect => edit(s, "Deselect", |d| d.selection = None),
        Action::InvertSelection => edit(s, "Invert Selection", |d| {
            let b = d.bounds();
            let inv = prisme_core::mask::invert(d.selection.as_deref(), b);
            d.selection = Some(Arc::new(inv));
        }),
        Action::SaveSelectionMask => {
            if let Some(d) = s.doc_mut() {
                save_selection_mask(d);
            }
        }
        Action::NewLayer => new_layer(s),
        Action::DuplicateLayer => duplicate_layer(s),
        Action::DeleteLayer => delete_layer(s),
        Action::MergeDown => merge_down(s),
        Action::GroupLayer => group_layer(s),
        Action::AddMask => add_mask(s, true),
        Action::ImageSize => s.dialogs.open_image_size(s.docs.get(s.active)),
        Action::CanvasSize => s.dialogs.open_canvas_size(s.docs.get(s.active)),
        Action::ZoomIn | Action::ZoomOut => {
            if let Some(d) = s.doc_mut() {
                let p = d.view.pan;
                d.view.zoom_at(if a == Action::ZoomIn { 1.25 } else { 0.8 }, p);
            }
        }
        Action::ZoomFit => {
            if let Some(d) = s.doc_mut() {
                d.view.fitted = false;
                d.view.zoom = 1.0;
                d.view.pan = [0.0, 0.0];
            }
        }
        Action::Zoom100 => {
            if let Some(d) = s.doc_mut() {
                let p = d.view.pan;
                let z = 1.0 / ctx.pixels_per_point() / d.view.zoom;
                d.view.zoom_at(z, p);
            }
        }
        Action::RotateViewLeft | Action::RotateViewRight => {
            if let Some(d) = s.doc_mut() {
                d.view.rotation += if a == Action::RotateViewLeft { -15f32.to_radians() } else { 15f32.to_radians() };
            }
        }
        Action::ResetView => {
            if let Some(d) = s.doc_mut() {
                d.view.rotation = 0.0;
            }
        }
        Action::ToggleNodeOutput => {
            if let Some(d) = s.doc_mut() {
                d.show_nodes = !d.show_nodes;
                d.touch();
            }
        }
        Action::SwapColors => std::mem::swap(&mut s.settings.fg, &mut s.settings.bg),
        Action::ResetColors => {
            s.settings.fg = [0.0, 0.0, 0.0, 1.0];
            s.settings.bg = [1.0, 1.0, 1.0, 1.0];
        }
        Action::BrushSmaller | Action::BrushBigger => {
            let k = if a == Action::BrushBigger { 1.2 } else { 1.0 / 1.2 };
            let o = &mut s.settings.tools;
            let b = match s.tool {
                ToolKind::Eraser => &mut o.eraser,
                ToolKind::Clone | ToolKind::Heal => &mut o.clone_brush,
                ToolKind::Smudge | ToolKind::Blur => &mut o.retouch_brush,
                _ => &mut o.brush,
            };
            b.size = (b.size * k).clamp(1.0, 2000.0);
            if b.size < 10.0 {
                b.size = b.size.round().max(1.0);
            }
        }
        Action::PickScreenColor => crate::jobs::pick_screen(s, false),
        Action::Preferences => s.dialogs.prefs_open = true,
        Action::Tool(t) => s.set_tool(t),
    }
}

fn reset_tools(s: &mut AppState) {
    for t in s.tools.values_mut() {
        t.reset();
    }
}

pub fn edit(s: &mut AppState, label: &str, f: impl FnOnce(&mut Document)) {
    if let Some(d) = s.doc_mut() {
        d.edit(label, f);
    }
}

pub fn close_doc(s: &mut AppState, i: usize) {
    if i >= s.docs.len() {
        return;
    }
    s.docs[i].finish_stroke();
    if s.docs[i].is_dirty() && s.dialogs.confirm_close != Some(i) {
        s.dialogs.confirm_close = Some(i);
        return;
    }
    s.dialogs.confirm_close = None;
    let d = s.docs.remove(i);
    s.node_editors.remove(&d.id);
    prisme_platform::storage_delete(&format!("autosave/{}", d.id));
    if let Some(g) = &mut s.gpu {
        let ids: Vec<_> = g.thumbs.keys().filter(|k| k.0 == d.id).copied().collect();
        let mut r = g.rs.renderer.write();
        for k in ids {
            if let Some((_, t, _)) = g.thumbs.remove(&k) {
                r.free_texture(&t);
            }
        }
    }
    if s.active >= s.docs.len() {
        s.active = s.docs.len().saturating_sub(1);
    }
}

// ------------------------------------------------------------------ layers

pub fn new_layer(s: &mut AppState) {
    edit(s, "New Layer", |d| {
        let id = d.alloc_layer_id();
        let n = d.all_layers().len() + 1;
        let l = Layer::new(id, format!("Layer {n}"), LayerKind::Raster(Arc::new(TileGrid::rgba(d.info.format))));
        let above = d.active_layer;
        d.insert_layer_above(l, above);
        d.active_layer = Some(id);
    });
}

pub fn group_layer(s: &mut AppState) {
    edit(s, "Group Layer", |d| {
        if let Some(a) = d.active_layer {
            if let Some(g) = d.group_layer(a) {
                d.active_layer = Some(g);
            }
        } else {
            let id = d.alloc_layer_id();
            d.layers.push(Layer::new(id, "Group", LayerKind::Group(GroupLayer { children: vec![], expanded: true })));
            d.active_layer = Some(id);
        }
    });
}

pub fn new_fill_layer(s: &mut AppState, gradient: bool) {
    let (fg, bg) = (s.settings.fg, s.settings.bg);
    edit(s, "New Fill Layer", |d| {
        let id = d.alloc_layer_id();
        let (w, h) = (d.info.width as f32, d.info.height as f32);
        let content = if gradient {
            FillContent::Gradient(Gradient::two(fg, bg), GradientGeometry { start: [0.0, h * 0.5], end: [w, h * 0.5], ..Default::default() })
        } else {
            FillContent::Solid(fg)
        };
        let mut l = Layer::new(id, if gradient { "Gradient Fill" } else { "Color Fill" }, LayerKind::Fill(FillLayer { content }));
        if let Some(sel) = &d.selection {
            l.mask = Some(LayerMask { grid: sel.clone(), enabled: true, default_visible: false });
        }
        let above = d.active_layer;
        d.insert_layer_above(l, above);
        d.active_layer = Some(id);
    });
}

pub fn add_mask(s: &mut AppState, from_selection: bool) {
    edit(s, "Add Layer Mask", |d| {
        let sel = d.selection.clone();
        if let Some(l) = d.active_mut() {
            l.mask = Some(match (from_selection, sel) {
                (true, Some(sel)) => LayerMask { grid: sel, enabled: true, default_visible: false },
                _ => LayerMask { grid: Arc::new(prisme_core::mask::new_mask_grid()), enabled: true, default_visible: true },
            });
        }
    });
}

fn reid(l: &mut Layer, d: &mut Document) {
    l.id = d.alloc_layer_id();
    if let LayerKind::Group(g) = &mut l.kind {
        for c in &mut g.children {
            reid(c, d);
        }
    }
}

pub fn duplicate_layer(s: &mut AppState) {
    edit(s, "Duplicate Layer", |d| {
        let Some(l) = d.active().cloned() else { return };
        let mut c = l.clone();
        reid(&mut c, d);
        c.name = format!("{} copy", l.name);
        let id = c.id;
        d.insert_layer_above(c, Some(l.id));
        d.active_layer = Some(id);
    });
}

pub fn delete_layer(s: &mut AppState) {
    edit(s, "Delete Layer", |d| {
        if let Some(a) = d.active_layer {
            d.remove_layer(a);
            d.scrub_graph_refs();
        }
    });
}

/// CPU rasterization of any layer kind at full resolution (straight alpha), including
/// opacity and mask when `apply_props` is set.
pub fn rasterize(s: &AppState, d: &Document, l: &Layer, apply_props: bool) -> TileGrid {
    let fmt = d.info.format.max(d.effective_format());
    let mut g = match &l.kind {
        LayerKind::Raster(g) => (**g).clone(),
        LayerKind::Text(t) => s.rasterizer.text.rasterize(t, 0).unwrap_or_else(|| TileGrid::rgba(fmt)),
        LayerKind::Vector(v) => s.rasterizer.vector.rasterize(v, 0).unwrap_or_else(|| TileGrid::rgba(fmt)),
        LayerKind::Fill(f) => {
            let mut g = TileGrid::rgba(fmt);
            let b = d.bounds();
            match &f.content {
                FillContent::Solid(c) => g.map_rect(&b, |_, _, p| p.copy_from_slice(c)),
                FillContent::Gradient(gr, geo) => prisme_tools::paint::draw_gradient(&mut g, b, gr, geo, 1.0, None),
            }
            g
        }
        LayerKind::Group(gr) => {
            let mut acc = TileGrid::rgba(fmt);
            for c in &gr.children {
                if c.visible {
                    acc = merge_over(&acc, &rasterize(s, d, c, true));
                }
            }
            acc
        }
    };
    if apply_props && (l.opacity < 1.0 || l.mask.as_ref().is_some_and(|m| m.enabled)) {
        let b = g.tile_bounds();
        let mask = l.mask.clone().filter(|m| m.enabled);
        g.map_rect(&b, |x, y, p| {
            let mut a = l.opacity;
            if let Some(m) = &mask {
                let v = m.grid.get_pixel(x, y)[0];
                a *= if m.grid.get_tile(prisme_core::TileCoord::of_pixel(x, y)).is_none() && m.default_visible { 1.0 } else { v };
            }
            p[3] *= a;
        });
        g.remove_clear_tiles();
    }
    g
}

pub fn rasterize_layer(s: &mut AppState) {
    let Some(d) = s.doc() else { return };
    let doc = d.doc().clone();
    let Some(l) = doc.active().cloned() else { return };
    if matches!(l.kind, LayerKind::Raster(_)) {
        return;
    }
    let g = rasterize(s, &doc, &l, false);
    edit(s, "Rasterize Layer", |d| {
        if let Some(x) = d.layer_mut(l.id) {
            x.kind = LayerKind::Raster(Arc::new(g));
        }
    });
}

pub fn merge_down(s: &mut AppState) {
    let Some(d) = s.doc() else { return };
    let doc = d.doc().clone();
    let Some(a) = doc.active_layer else { return };
    let Some(path) = doc.path_of(a) else { return };
    let (last, parent) = path.split_last().unwrap();
    if *last == 0 {
        s.toast("No layer below to merge into");
        return;
    }
    let mut sib: &[Layer] = &doc.layers;
    for &i in parent {
        sib = sib[i].children();
    }
    let top = sib[*last].clone();
    let below = sib[*last - 1].clone();
    let below_g = rasterize(s, &doc, &below, true);
    let top_g = rasterize(s, &doc, &top, true);
    let merged = merge_over(&below_g, &top_g);
    if top.blend != prisme_core::BlendMode::Normal {
        s.toast("Merged with normal blending (blend modes are applied by the GPU compositor only)");
    }
    edit(s, "Merge Down", |d| {
        d.remove_layer(top.id);
        if let Some(b) = d.layer_mut(below.id) {
            b.kind = LayerKind::Raster(Arc::new(merged));
            b.opacity = 1.0;
            b.mask = None;
        }
        d.active_layer = Some(below.id);
        d.scrub_graph_refs();
    });
}

pub fn flatten(s: &mut AppState) {
    let Some(d) = s.doc() else { return };
    let doc = d.doc().clone();
    let mut acc = TileGrid::rgba(doc.effective_format());
    for l in &doc.layers {
        if l.visible {
            acc = merge_over(&acc, &rasterize(s, &doc, l, true));
        }
    }
    edit(s, "Flatten Image", |d| {
        let id = d.alloc_layer_id();
        d.layers = vec![Layer::new(id, "Background", LayerKind::Raster(Arc::new(acc)))];
        d.active_layer = Some(id);
        d.scrub_graph_refs();
    });
}

// ------------------------------------------------------------------ selection & pixels

pub fn save_selection_mask(d: &mut DocState) {
    d.edit("Save Selection as Mask", |doc| {
        if let Some(sel) = doc.selection.clone() {
            let id = doc.alloc_mask_id();
            let n = doc.masks.len() + 1;
            doc.masks.push(SavedMask { id, name: format!("Mask {n}"), mask: sel });
        }
    });
}

pub fn selection_modify(s: &mut AppState, op: &str, amount: f32) {
    let label = match op {
        "grow" => "Grow Selection",
        "shrink" => "Shrink Selection",
        "feather" => "Feather Selection",
        "border" => "Border Selection",
        _ => "Smooth Selection",
    };
    edit(s, label, |d| {
        let Some(sel) = d.selection.clone() else { return };
        let g = match op {
            "grow" => prisme_core::mask::grow(&sel, amount),
            "shrink" => prisme_core::mask::grow(&sel, -amount),
            "feather" => prisme_core::mask::feather(&sel, amount),
            "border" => prisme_core::mask::border(&sel, amount),
            _ => prisme_core::mask::harden(&prisme_core::mask::feather(&sel, amount)),
        };
        d.selection = Some(Arc::new(g));
    });
}

pub fn select_layer_alpha(s: &mut AppState) {
    edit(s, "Select Layer Alpha", |d| {
        let Some(g) = d.active().and_then(|l| l.raster().cloned()) else { return };
        let mut m = prisme_core::mask::new_mask_grid();
        for (c, t) in &g.tiles {
            let v = t.f32_view();
            let a: Vec<f32> = v.chunks(4).map(|p| p[3]).collect();
            m.insert(*c, prisme_core::Tile::from_f32(prisme_core::PixelFormat::F16, 1, &a));
        }
        m.remove_clear_tiles();
        d.selection = Some(Arc::new(m));
    });
}

fn clear_selection_pixels(s: &mut AppState, label: &str) {
    edit(s, label, |d| {
        let Some(sel) = d.selection.clone() else { return };
        if let Some(g) = d.active_mut().and_then(|l| l.raster_mut()) {
            let (_, rest) = split_by_selection(g, &sel);
            *g = rest;
        }
    });
}

fn fill_selection(s: &mut AppState, bg: bool) {
    let c = if bg { s.settings.bg } else { s.settings.fg };
    edit(s, "Fill", |d| {
        let b = d.bounds();
        let sel = d.selection.clone().unwrap_or_else(|| Arc::new(prisme_core::mask::select_all(b)));
        if let Some(g) = d.active_mut().and_then(|l| l.raster_mut()) {
            prisme_tools::paint::fill_region(g, &sel, c, 1.0, None);
        }
    });
}

pub fn crop_to_selection(s: &mut AppState) {
    edit(s, "Crop to Selection", |d| {
        if let Some(sel) = &d.selection {
            let b = sel.content_bounds().intersect(&d.bounds());
            if !b.is_empty() {
                d.crop(b, false);
            }
        }
    });
}

pub fn straighten(s: &mut AppState, angle: f32) {
    // Rotate by -angle (snapping to the nearest axis).
    let mut a = angle;
    while a > std::f32::consts::FRAC_PI_4 {
        a -= std::f32::consts::FRAC_PI_2;
    }
    while a < -std::f32::consts::FRAC_PI_4 {
        a += std::f32::consts::FRAC_PI_2;
    }
    edit(s, "Straighten", |d| prisme_tools::transform::rotate_document(d, -a, Resample::Bicubic));
}

// ------------------------------------------------------------------ clipboard

fn to_srgb8(px: &[f32]) -> Vec<u8> {
    let c = prisme_color::srgb();
    px.chunks(4)
        .flat_map(|p| {
            let e = c.from_working([p[0], p[1], p[2]]);
            let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            [q(e[0]), q(e[1]), q(e[2]), q(p[3])]
        })
        .collect()
}

fn copy(s: &mut AppState, merged: bool) {
    let Some(d) = s.doc() else { return };
    let doc = d.doc().clone();
    let src = if merged {
        let mut acc = TileGrid::rgba(doc.effective_format());
        for l in &doc.layers {
            if l.visible {
                acc = merge_over(&acc, &rasterize(s, &doc, l, true));
            }
        }
        acc
    } else {
        match doc.active() {
            Some(l) => rasterize(s, &doc, l, false),
            None => return,
        }
    };
    let (grid, bounds) = match &doc.selection {
        Some(sel) => {
            let (f, _) = split_by_selection(&src, sel);
            let b = sel.content_bounds().intersect(&doc.bounds());
            (f, b)
        }
        None => (src.clone(), src.content_bounds().intersect(&doc.bounds())),
    };
    if bounds.is_empty() {
        s.toast("Nothing to copy");
        return;
    }
    let px = grid.read_rect(&bounds);
    let img = prisme_platform::ClipboardImage { width: bounds.width() as u32, height: bounds.height() as u32, rgba: to_srgb8(&px) };
    s.clipboard = Some((grid, bounds));
    s.jobs.push(crate::jobs::Job::ClipboardSet(prisme_platform::clipboard_set_image(img)));
    s.toast(format!("Copied {} × {}", bounds.width(), bounds.height()));
}

fn paste(s: &mut AppState, new_doc: bool) {
    s.jobs.push(crate::jobs::Job::ClipboardGet { pending: prisme_platform::clipboard_get_image(), new_doc });
}

/// Paste pixels as a new layer (centered on the view when they came from outside).
pub fn paste_grid(s: &mut AppState, grid: TileGrid, bounds: IRect, new_doc: bool) {
    if new_doc || s.docs.is_empty() {
        let mut d = Document::new("Pasted", bounds.width() as u32, bounds.height() as u32, grid.format);
        let id = d.alloc_layer_id();
        d.layers.push(Layer::new(id, "Pasted", LayerKind::Raster(Arc::new(grid.translated(-bounds.x0, -bounds.y0)))));
        d.active_layer = Some(id);
        s.add_doc(DocState::new(d));
        return;
    }
    edit(s, "Paste", |d| {
        let id = d.alloc_layer_id();
        let mut g = grid;
        // Keep the original position if it lies inside the canvas, else centre it.
        if bounds.intersect(&d.bounds()).is_empty() || bounds.x0 == 0 && bounds.y0 == 0 && (bounds.width() as u32) < d.info.width {
            let dx = (d.info.width as i32 - bounds.width()) / 2 - bounds.x0;
            let dy = (d.info.height as i32 - bounds.height()) / 2 - bounds.y0;
            g = prisme_tools::transform::translate(&g, dx, dy);
        }
        let above = d.active_layer;
        d.insert_layer_above(Layer::new(id, "Pasted", LayerKind::Raster(Arc::new(g))), above);
        d.active_layer = Some(id);
    });
}

// ------------------------------------------------------------------ menus

fn item(ui: &mut Ui, s: &AppState, label: &str, a: Action) -> bool {
    let sc = s.settings.keymap.label(ui.ctx(), a);
    ui.add(egui::Button::new(label).shortcut_text(sc)).clicked()
}

pub fn menu_bar(s: &mut AppState, ui: &mut Ui, dock: &mut DockState<Tab>) {
    let ctx = ui.ctx().clone();
    let mut run: Option<Action> = None;
    let has_doc = !s.docs.is_empty();
    ui.label(egui::RichText::new(format!("{} Prisme", egui_phosphor::regular::DIAMOND)).strong().color(crate::theme::ACCENT));
    ui.menu_button("File", |ui| {
        for (l, a) in [("New…", Action::New), ("Open…", Action::Open)] {
            if item(ui, s, l, a) {
                run = Some(a);
            }
        }
        open_recent_menu(s, ui);
        ui.add_enabled_ui(has_doc, |ui| {
            if ui.button("Place image as layer…").clicked() {
                crate::jobs::place_dialog(s);
                ui.close();
            }
            ui.separator();
            for (l, a) in [("Save", Action::Save), ("Save As…", Action::SaveAs), ("Export PNG…", Action::Export), ("Close", Action::Close)] {
                if item(ui, s, l, a) {
                    run = Some(a);
                }
            }
        });
    });
    ui.menu_button("Edit", |ui| {
        ui.add_enabled_ui(has_doc, |ui| {
            let (cu, cr) = s.doc().map(|d| (d.history.can_undo(), d.history.can_redo())).unwrap_or((false, false));
            ui.add_enabled_ui(cu, |ui| {
                if item(ui, s, "Undo", Action::Undo) {
                    run = Some(Action::Undo);
                }
            });
            ui.add_enabled_ui(cr, |ui| {
                if item(ui, s, "Redo", Action::Redo) {
                    run = Some(Action::Redo);
                }
            });
            ui.separator();
            for (l, a) in [
                ("Cut", Action::Cut),
                ("Copy", Action::Copy),
                ("Copy Merged", Action::CopyMerged),
                ("Paste", Action::Paste),
                ("Paste as New Image", Action::PasteNewDoc),
                ("Clear", Action::Delete),
            ] {
                if item(ui, s, l, a) {
                    run = Some(a);
                }
            }
            ui.separator();
            for (l, a) in [("Fill with Foreground", Action::FillFg), ("Fill with Background", Action::FillBg)] {
                if item(ui, s, l, a) {
                    run = Some(a);
                }
            }
        });
        ui.separator();
        if item(ui, s, "Preferences & Shortcuts…", Action::Preferences) {
            run = Some(Action::Preferences);
        }
    });
    ui.menu_button("Image", |ui| {
        ui.add_enabled_ui(has_doc, |ui| {
            if item(ui, s, "Image Size…", Action::ImageSize) {
                run = Some(Action::ImageSize);
            }
            if item(ui, s, "Canvas Size…", Action::CanvasSize) {
                run = Some(Action::CanvasSize);
            }
            ui.separator();
            for (l, op) in [("Rotate 90° CW", 0u8), ("Rotate 180°", 1), ("Rotate 90° CCW", 2), ("Flip Horizontal", 3), ("Flip Vertical", 4)] {
                if ui.button(l).clicked() {
                    edit(s, l, |d| d.orthogonal_transform(op));
                    ui.close();
                }
            }
            ui.separator();
            if ui.button("Crop to Selection").clicked() {
                crop_to_selection(s);
                ui.close();
            }
            if ui.button("Flatten Image").clicked() {
                flatten(s);
                ui.close();
            }
        });
    });
    ui.menu_button("Layer", |ui| {
        ui.add_enabled_ui(has_doc, |ui| {
            for (l, a) in [("New Layer", Action::NewLayer), ("Duplicate", Action::DuplicateLayer), ("Delete", Action::DeleteLayer), ("Group", Action::GroupLayer), ("Merge Down", Action::MergeDown)] {
                if item(ui, s, l, a) {
                    run = Some(a);
                }
            }
            ui.separator();
            if ui.button("New Solid Fill Layer").clicked() {
                new_fill_layer(s, false);
                ui.close();
            }
            if ui.button("New Gradient Fill Layer").clicked() {
                new_fill_layer(s, true);
                ui.close();
            }
            ui.separator();
            if ui.button("Add Mask (reveal all)").clicked() {
                add_mask(s, false);
                ui.close();
            }
            if ui.button("Add Mask from Selection").clicked() {
                add_mask(s, true);
                ui.close();
            }
            if ui.button("Delete Mask").clicked() {
                edit(s, "Delete Mask", |d| {
                    if let Some(l) = d.active_mut() {
                        l.mask = None;
                    }
                });
                ui.close();
            }
            ui.separator();
            if ui.button("Rasterize").clicked() {
                rasterize_layer(s);
                ui.close();
            }
        });
    });
    ui.menu_button("Select", |ui| {
        ui.add_enabled_ui(has_doc, |ui| {
            for (l, a) in [("All", Action::SelectAll), ("Deselect", Action::Deselect), ("Invert", Action::InvertSelection)] {
                if item(ui, s, l, a) {
                    run = Some(a);
                }
            }
            ui.separator();
            for (l, op) in [("Grow…", "grow"), ("Shrink…", "shrink"), ("Feather…", "feather"), ("Border…", "border"), ("Smooth…", "smooth")] {
                if ui.button(l).clicked() {
                    s.dialogs.modify_sel = Some((op.to_string(), 5.0));
                    ui.close();
                }
            }
            ui.separator();
            if ui.button("Select Layer Alpha").clicked() {
                select_layer_alpha(s);
                ui.close();
            }
            if item(ui, s, "Save Selection as Mask", Action::SaveSelectionMask) {
                run = Some(Action::SaveSelectionMask);
            }
        });
    });
    ui.menu_button("Nodes", |ui| {
        ui.add_enabled_ui(has_doc, |ui| {
            if ui.button("Show Node Editor").clicked() {
                crate::app::show_node_editor(dock);
                ui.close();
            }
            if item(ui, s, "Toggle Node Output", Action::ToggleNodeOutput) {
                run = Some(Action::ToggleNodeOutput);
            }
            ui.separator();
            ui.label(egui::RichText::new("Insert before output").small());
            for cat in prisme_graph::Category::ALL {
                ui.menu_button(cat.name(), |ui| {
                    for k in prisme_graph::registry().by_category(cat) {
                        if ui.button(k.name).on_hover_text(k.description).clicked() {
                            if let Some(d) = s.doc_mut() {
                                crate::panels::nodes::insert_before_output(d, k.id);
                            }
                            ui.close();
                        }
                    }
                });
            }
        });
    });
    ui.menu_button("View", |ui| {
        for (l, a) in [
            ("Zoom In", Action::ZoomIn),
            ("Zoom Out", Action::ZoomOut),
            ("Fit on Screen", Action::ZoomFit),
            ("Actual Pixels", Action::Zoom100),
            ("Rotate View Left", Action::RotateViewLeft),
            ("Rotate View Right", Action::RotateViewRight),
            ("Reset View Rotation", Action::ResetView),
        ] {
            if item(ui, s, l, a) {
                run = Some(a);
            }
        }
        ui.separator();
        ui.checkbox(&mut s.settings.pixel_grid, "Pixel grid");
        let mut dp = s.settings.use_display_profile;
        if ui.checkbox(&mut dp, "Use display colour profile").changed() {
            s.set_display_profile(dp);
        }
        ui.checkbox(&mut s.settings.dither_display, "Dither display (no banding)");
        ui.separator();
        ui.menu_button("Panels", |ui| {
            for t in Tab::ALL {
                if t == Tab::Canvas {
                    continue;
                }
                let open = dock.find_tab(&t).is_some();
                if ui.selectable_label(open, t.title()).clicked() {
                    if let Some(loc) = dock.find_tab(&t) {
                        dock.remove_tab(loc);
                    } else {
                        dock.push_to_focused_leaf(t);
                    }
                }
            }
        });
        if ui.button("Reset Layout").clicked() {
            *dock = crate::app::default_dock_pub();
            ui.close();
        }
        ui.separator();
        let mut dark = s.settings.theme == crate::theme::ThemeKind::Dark;
        if ui.checkbox(&mut dark, "Dark theme").changed() {
            s.settings.theme = if dark { crate::theme::ThemeKind::Dark } else { crate::theme::ThemeKind::Light };
            crate::theme::apply(&ctx, s.settings.theme);
        }
    });
    ui.menu_button("Help", |ui| {
        if ui.button("Keyboard Shortcuts…").clicked() {
            s.dialogs.prefs_open = true;
            ui.close();
        }
        if ui.button("About Prisme").clicked() {
            s.dialogs.about_open = true;
            ui.close();
        }
    });
    if let Some(a) = run {
        run_action(s, a, &ctx);
    }
}

/// File ▸ Open Recent submenu.
pub fn open_recent_menu(s: &mut AppState, ui: &mut Ui) {
    let recent = s.settings.recent_files.clone();
    ui.add_enabled_ui(!recent.is_empty() && !cfg!(target_arch = "wasm32"), |ui| {
        ui.menu_button("Open Recent", |ui| {
            for p in &recent {
                let name = std::path::Path::new(p).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| p.clone());
                if ui.button(name).on_hover_text(p).clicked() {
                    crate::jobs::open_recent(s, p);
                    ui.close();
                }
            }
            ui.separator();
            if ui.button("Clear Recent").clicked() {
                s.settings.recent_files.clear();
                ui.close();
            }
        });
    });
}

pub fn canvas_context_menu(s: &mut AppState, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    let mut run = None;
    for (l, a) in [("Undo", Action::Undo), ("Redo", Action::Redo), ("Copy", Action::Copy), ("Paste", Action::Paste), ("Deselect", Action::Deselect), ("Select All", Action::SelectAll), ("Fit on Screen", Action::ZoomFit)] {
        if item(ui, s, l, a) {
            run = Some(a);
            ui.close();
        }
    }
    if let Some(a) = run {
        run_action(s, a, &ctx);
    }
}
