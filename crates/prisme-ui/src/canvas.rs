//! The canvas: document tabs, view navigation, tool input, GPU rendering and overlays.

use crate::app::AppState;
use crate::theme;
use egui::{Color32, Pos2, Rect, Sense, Shape, Stroke, Ui, Vec2};
use prisme_core::IRect;
use prisme_gpu::{DisplayParams, DisplayRenderer, EvalFormat, EvalRequest, GpuImage, RenderSource};
use prisme_tools::{Modifiers, Overlay, Phase, PointerKind, ToolCursor, ToolCx, ToolInput, ToolKind, ToolOutcome};

struct CanvasCallback {
    image: GpuImage,
    params: DisplayParams,
}

impl egui_wgpu::CallbackTrait for CanvasCallback {
    fn prepare(
        &self,
        device: &egui_wgpu::wgpu::Device,
        _queue: &egui_wgpu::wgpu::Queue,
        _sd: &egui_wgpu::ScreenDescriptor,
        _enc: &mut egui_wgpu::wgpu::CommandEncoder,
        res: &mut egui_wgpu::CallbackResources,
    ) -> Vec<egui_wgpu::wgpu::CommandBuffer> {
        if let Some(r) = res.get_mut::<DisplayRenderer>() {
            r.prepare(device, &self.params, &self.image);
        }
        vec![]
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut egui_wgpu::wgpu::RenderPass<'static>, res: &egui_wgpu::CallbackResources) {
        if let Some(r) = res.get::<DisplayRenderer>() {
            r.paint(pass);
        }
    }
}

#[derive(Clone, Copy, Default)]
struct DragState {
    tool_active: bool,
    panning: bool,
    last_pos: Option<Pos2>,
}

pub fn canvas_tab(s: &mut AppState, ui: &mut Ui) {
    doc_tabs(s, ui);
    if s.docs.is_empty() {
        welcome(s, ui);
        return;
    }
    let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
    let ctx = ui.ctx().clone();
    let ppp = ctx.pixels_per_point();
    let center = rect.center();
    let active = s.active;

    // Fit on first show (and Fit on Screen): the page, or the content of an infinite canvas.
    {
        let v = s.docs[active].view;
        if !v.fitted && v.zoom == 1.0 && v.pan == [0.0, 0.0] {
            let r = crate::commands::doc_extent(s, s.docs[active].doc());
            s.docs[active].view.fit(r, rect.size());
        }
    }

    // ---------------------------------------------------------------- navigation
    let hovered = resp.hovered() || resp.contains_pointer();
    let (zoom_delta, scroll, pointer) = ctx.input(|i| (i.zoom_delta(), i.smooth_scroll_delta(), i.pointer.hover_pos()));
    if hovered {
        let d = &mut s.docs[active];
        if zoom_delta != 1.0 {
            let anchor = pointer.map(|p| d.view.screen_to_doc(center, p)).unwrap_or(d.view.pan);
            d.view.zoom_at(zoom_delta, anchor);
        } else if scroll != Vec2::ZERO {
            let (sn, cs) = (-d.view.rotation).sin_cos();
            let dx = -scroll.x / d.view.zoom;
            let dy = -scroll.y / d.view.zoom;
            d.view.pan[0] += dx * cs - dy * sn;
            d.view.pan[1] += dx * sn + dy * cs;
            d.view.fitted = false;
        }
    }

    // ---------------------------------------------------------------- tool input
    let space = ctx.input(|i| i.key_down(egui::Key::Space)) && s.text_editing().is_none();
    let drag_id = ui.id().with("canvas-drag");
    let mut ds: DragState = ctx.data(|d| d.get_temp(drag_id)).unwrap_or_default();
    let events = ctx.input(|i| i.events.clone());
    let mods = ctx.input(|i| i.modifiers);
    let time = ctx.input(|i| i.time);
    let tmods = Modifiers { shift: mods.shift, alt: mods.alt, primary: mods.command };
    let mut pressure = 1.0f32;
    let mut pointer_kind = PointerKind::Mouse;
    for ev in &events {
        if let egui::Event::Touch { force: Some(f), .. } = ev {
            pressure = f.clamp(0.0, 1.0);
            pointer_kind = PointerKind::Pen;
        }
    }
    let mut tool_events: Vec<(Phase, Pos2, bool)> = vec![];
    for ev in &events {
        match ev {
            egui::Event::PointerButton { pos, button, pressed, .. } => {
                let inside = rect.contains(*pos) && hovered;
                match (button, pressed) {
                    (egui::PointerButton::Middle, true) if inside => {
                        ds.panning = true;
                        ds.last_pos = Some(*pos);
                    }
                    (egui::PointerButton::Primary, true) if inside && (space || s.tool == ToolKind::Hand && false) => {
                        ds.panning = true;
                        ds.last_pos = Some(*pos);
                    }
                    (egui::PointerButton::Primary, true) if inside => {
                        ds.tool_active = true;
                        tool_events.push((Phase::Down, *pos, false));
                    }
                    (egui::PointerButton::Secondary, true) if inside => {
                        tool_events.push((Phase::Down, *pos, true));
                        tool_events.push((Phase::Up, *pos, true));
                    }
                    (egui::PointerButton::Primary | egui::PointerButton::Middle, false) => {
                        if ds.panning {
                            ds.panning = false;
                        } else if ds.tool_active && *button == egui::PointerButton::Primary {
                            ds.tool_active = false;
                            tool_events.push((Phase::Up, *pos, false));
                        }
                    }
                    _ => {}
                }
            }
            egui::Event::PointerMoved(pos) => {
                if ds.panning {
                    if let Some(last) = ds.last_pos {
                        let d = &mut s.docs[active];
                        let delta = *pos - last;
                        let (sn, cs) = (-d.view.rotation).sin_cos();
                        let dx = -delta.x / d.view.zoom;
                        let dy = -delta.y / d.view.zoom;
                        d.view.pan[0] += dx * cs - dy * sn;
                        d.view.pan[1] += dx * sn + dy * cs;
                        d.view.fitted = false;
                    }
                    ds.last_pos = Some(*pos);
                } else if ds.tool_active {
                    tool_events.push((Phase::Move, *pos, false));
                } else if rect.contains(*pos) && hovered {
                    tool_events.push((Phase::Hover, *pos, false));
                }
            }
            _ => {}
        }
    }
    ctx.data_mut(|d| d.insert_temp(drag_id, ds));

    for (phase, pos, secondary) in tool_events {
        let view = s.docs[active].view;
        let p = view.screen_to_doc(center, pos);
        let ev = ToolInput {
            phase,
            pos: p,
            pressure,
            tilt: 0.0,
            tilt_dir: 0.0,
            pointer: pointer_kind,
            mods: tmods,
            time,
            zoom: view.zoom,
            secondary,
        };
        if secondary {
            continue; // context menus are handled by egui below
        }
        run_tool(s, ev);
    }
    // Airbrush ticks.
    if s.tools.get(&s.tool).is_some_and(|t| t.ticking()) {
        let fg = s.settings.fg;
        let bg = s.settings.bg;
        let opts = s.settings.tools.clone();
        let tool = s.tool;
        if let (Some(t), Some(d)) = (s.tools.get_mut(&tool), s.docs.get_mut(active))
            && let Some(w) = d.working.as_mut() {
                let mut ops = vec![];
                let gpu = s.gpu.as_mut();
                let mut cx = ToolCx { doc: w, opts: &opts, fg, bg, strokes: gpu.is_some().then_some(&mut ops), view: d.view_rect };
                let out = t.tick(&mut cx, time);
                if let Some(g) = gpu {
                    crate::stroke::feed(d, &mut g.engine, ops);
                }
                if out == ToolOutcome::Preview {
                    d.touch();
                }
            }
        ctx.request_repaint();
    }
    s.hover_doc = pointer.filter(|p| rect.contains(*p)).map(|p| s.docs[active].view.screen_to_doc(center, p));

    // ---------------------------------------------------------------- render
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::from_rgb(21, 22, 24));
    render(s, &ctx, rect, ppp, &painter);

    // ---------------------------------------------------------------- overlays
    overlays(s, &ctx, rect, &painter);

    // Cursor.
    let blocked = blocked_reason(s);
    if hovered {
        let icon = if ds.panning || space {
            egui::CursorIcon::Grabbing
        } else if blocked.is_some() {
            egui::CursorIcon::NotAllowed
        } else {
            match s.tools.get(&s.tool).map(|t| t.cursor()) {
                Some(ToolCursor::Crosshair) | Some(ToolCursor::Brush) => egui::CursorIcon::Crosshair,
                Some(ToolCursor::Move) => egui::CursorIcon::Move,
                Some(ToolCursor::Grab) => egui::CursorIcon::Grab,
                Some(ToolCursor::Text) => egui::CursorIcon::Text,
                Some(ToolCursor::Picker) => egui::CursorIcon::Crosshair,
                Some(ToolCursor::ZoomIn) => egui::CursorIcon::ZoomIn,
                _ => egui::CursorIcon::Default,
            }
        };
        ctx.set_cursor_icon(icon);
    }

    resp.context_menu(|ui| crate::commands::canvas_context_menu(s, ui));
}

/// Why the current tool can't edit its target right now (locked layer, wrong layer type…).
pub fn blocked_reason(s: &AppState) -> Option<String> {
    let tool = s.tool;
    let doc = s.doc()?.doc();
    let paint = tool.uses_brush() || tool == ToolKind::Fill || (tool == ToolKind::Gradient && !s.settings.tools.gradient_as_layer);
    let moves = matches!(tool, ToolKind::Move | ToolKind::Transform);
    if tool == ToolKind::Liquify {
        let l = doc.active()?;
        return if l.lock.pixels {
            Some(format!("“{}” has locked pixels", l.name))
        } else {
            l.raster().is_none().then(|| format!("“{}” is a {} layer — Liquify works on raster layer pixels", l.name, l.kind.type_name().to_lowercase()))
        };
    }
    if !paint && !moves {
        return None;
    }
    if paint && s.settings.tools.edit_saved_mask.is_some_and(|m| doc.saved_mask(m).is_some()) {
        return None;
    }
    let l = doc.active()?;
    if paint {
        if l.lock.pixels {
            return Some(format!("“{}” has locked pixels", l.name));
        }
        if s.settings.tools.edit_mask {
            return l.mask.is_none().then(|| "The layer has no mask".to_string());
        }
        if l.raster().is_none() {
            return Some(format!("“{}” is a {} layer — rasterize it to paint", l.name, l.kind.type_name().to_lowercase()));
        }
    }
    if moves && l.lock.position {
        return Some(format!("“{}” has a locked position", l.name));
    }
    None
}

/// Feed one event to the current tool and apply its outcome.
pub fn run_tool(s: &mut AppState, ev: ToolInput) {
    let tool_kind = s.tool;
    let fg = s.settings.fg;
    let bg = s.settings.bg;
    let opts = s.settings.tools.clone();
    let Some(doc) = s.doc() else { return };
    if tool_kind == ToolKind::Crop && doc.doc().info.is_infinite() {
        if ev.phase == Phase::Down {
            s.toast("Crop needs a fixed-size canvas (this one is infinite): use Export Selection");
        }
        return;
    }
    let Some(d) = s.docs.get_mut(s.active) else { return };
    let Some(tool) = s.tools.get_mut(&tool_kind) else { return };
    // New interactions start from the landed previous stroke.
    if ev.phase != Phase::Hover {
        d.finish_stroke();
    }
    if d.working.is_none() {
        d.working = Some(d.history.current().clone());
    }
    // With a GPU, dab tools emit stroke ops instead of writing tiles (crate::stroke).
    let mut ops = vec![];
    let gpu = s.gpu.as_mut();
    let outcome = {
        let mut cx = ToolCx { doc: d.working.as_mut().unwrap(), opts: &opts, fg, bg, strokes: gpu.is_some().then_some(&mut ops), view: d.view_rect };
        tool.input(&mut cx, &ev)
    };
    if let Some(g) = gpu {
        crate::stroke::feed(d, &mut g.engine, ops);
    }
    let busy = tool.busy();
    apply_outcome(s, outcome, busy);
}

pub fn apply_outcome(s: &mut AppState, outcome: ToolOutcome, busy: bool) {
    let changed = matches!(outcome, ToolOutcome::Preview | ToolOutcome::Commit(_) | ToolOutcome::Cancel | ToolOutcome::EditText(_));
    let tool_kind = s.tool;
    let text_editing = s.text_editing().is_some();
    let Some(d) = s.docs.get_mut(s.active) else { return };
    match outcome {
        ToolOutcome::Preview => d.touch(),
        ToolOutcome::Commit(label) => {
            d.commit_working(&label);
            if tool_kind == ToolKind::Text && text_editing {
                // keep editing on a fresh working copy
                d.working = Some(d.history.current().clone());
            }
        }
        ToolOutcome::Cancel => d.cancel_working(),
        ToolOutcome::EditText(_) => {
            // New text layers are committed immediately so the edit session builds on them.
            d.touch();
        }
        ToolOutcome::Pan(delta) => {
            d.view.pan[0] -= delta[0];
            d.view.pan[1] -= delta[1];
            d.view.fitted = false;
        }
        ToolOutcome::Zoom { at, factor } => d.view.zoom_at(factor, at),
        ToolOutcome::Message(m) => {
            s.toast(m);
        }
        ToolOutcome::SampleColor { pos, raw, to_bg } => sample_color(s, pos, raw, to_bg),
        ToolOutcome::None => {}
    }
    // Drop untouched working copies when the tool is idle (nothing to commit).
    let text_editing = s.text_editing().is_some();
    // A GPU Liquify session previews from its stroke: between strokes there's no working
    // copy, so panel edits go straight to the history (and survive the bake).
    let ticking = s.tools.get(&s.tool).is_some_and(|t| t.ticking());
    if let Some(d) = s.docs.get_mut(s.active)
        && (!busy || !ticking && d.stroke.as_ref().is_some_and(|st| st.gpu.is_liquify()))
        && !text_editing && d.working.is_some() {
            d.working = None;
            if changed {
                d.touch();
            }
        }
}

fn sample_color(s: &mut AppState, pos: [f32; 2], raw: bool, to_bg: bool) {
    let Some(d) = s.doc() else { return };
    let doc = d.doc();
    let (x, y) = (pos[0].floor() as i32, pos[1].floor() as i32);
    if !doc.info.is_infinite() && !doc.bounds().contains(x, y) {
        return;
    }
    if raw {
        // Active layer's stored value: no nodes, no colour transforms.
        let c = match doc.active() {
            Some(l) if l.raster().is_some() => l.raster().unwrap().get_pixel(x, y),
            _ => prisme_tools::flatten::flatten_pixel(doc, x, y),
        };
        if c[3] > 0.0 {
            set_color(s, [c[0], c[1], c[2], 1.0], to_bg);
        }
        return;
    }
    // Displayed colour: read back from the rendered image.
    let (image, mip) = (d.render.image.clone(), d.render.mip);
    if let (Some(img), Some(g)) = (image, s.gpu.as_mut()) {
        let k = 1 << mip;
        let r = IRect::new(x / k, y / k, x / k + 1, y / k + 1);
        if img.rect.intersect(&r) == r {
            g.pick = Some((prisme_gpu::Readback::start(&g.rs.device, &g.rs.queue, &img, r), to_bg));
        }
    }
}

pub fn set_color(s: &mut AppState, c: [f32; 4], to_bg: bool) {
    if to_bg {
        s.settings.bg = c;
    } else {
        s.settings.fg = c;
        let rc = &mut s.settings.recent_colors;
        rc.retain(|x| x != &c);
        rc.insert(0, c);
        rc.truncate(16);
    }
}

fn render(s: &mut AppState, ctx: &egui::Context, rect: Rect, ppp: f32, painter: &egui::Painter) {
    let use_dither = s.settings.dither_display;
    let output = s.hdr.output(s.settings.hdr_highlights);
    let scope_range = if s.settings.scope_range >= 1.0 { s.settings.scope_range } else { s.hdr.scope_range() };
    let pixel_grid = s.settings.pixel_grid;
    let editing_mask = s.settings.tools.edit_saved_mask;
    let active = s.active;
    let interacting = s.tools.get(&s.tool).is_some_and(|t| t.busy()) || ctx.input(|i| i.pointer.any_down());
    let Some(g) = s.gpu.as_mut() else {
        painter.text(rect.center(), egui::Align2::CENTER_CENTER, "GPU unavailable", egui::FontId::proportional(16.0), theme::WARN);
        return;
    };
    let d = &mut s.docs[active];
    // A stroke just landed: its tiles are already on the GPU, don't upload them again.
    if let Some((p, ids)) = d.adopt.take() {
        g.engine.adopt_tiles(&p, &ids);
    }
    let doc = d.working.as_ref().unwrap_or_else(|| d.history.current());
    let infinite = doc.info.is_infinite();
    let center = rect.center();
    let zoom_px = d.view.zoom * ppp;
    let corners = [rect.left_top(), rect.right_top(), rect.right_bottom(), rect.left_bottom()].map(|p| d.view.screen_to_doc(center, p));
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for c in corners {
        x0 = x0.min(c[0]);
        y0 = y0.min(c[1]);
        x1 = x1.max(c[0]);
        y1 = y1.max(c[1]);
    }
    let view_r = IRect::from_f32(x0, y0, x1, y1);
    d.view_rect = view_r;
    // Infinite canvas: no page, the evaluated area is just the view.
    let page = if infinite { view_r } else { doc.bounds() };
    let (w, h) = (page.width(), page.height());
    let max_mip = ((w.max(h) as f32).log2().floor() as u32).saturating_sub(4);
    let mip = if zoom_px >= 1.0 { 0 } else { ((1.0 / zoom_px).log2().floor() as u32).min(max_mip).min(10) };
    let visible = view_r.intersect(&page);
    if visible.is_empty() {
        return;
    }
    let k = 1i32 << mip;
    let level = |r: IRect| IRect::new(r.x0.div_euclid(k), r.y0.div_euclid(k), (r.x1 + k - 1).div_euclid(k), (r.y1 + k - 1).div_euclid(k));
    let level_doc = level(page);
    let clip = |r: IRect| if infinite { r } else { r.intersect(&level_doc) };
    let need = clip(level(visible));
    // Editing a saved mask: show it as a red quick-mask overlay on top of the output.
    let overlay = editing_mask.and_then(|mid| crate::panels::masks::overlay_doc(doc, mid, d.show_nodes));
    let doc = overlay.as_ref().unwrap_or(doc);
    let source = if overlay.is_some() {
        RenderSource::Graph
    } else if let Some(n) = d.preview_node.filter(|n| doc.graph.nodes.contains_key(n)) {
        RenderSource::Node(n)
    } else if d.show_nodes && !doc.graph.is_trivial() {
        RenderSource::Graph
    } else {
        RenderSource::LayerStack
    };
    let src_tag = match source {
        RenderSource::Graph if overlay.is_some() => 3u8,
        RenderSource::Graph => 0u8,
        RenderSource::LayerStack => 1,
        RenderSource::Node(_) => 2,
    };
    let reuse = matches!((&d.render.key, &d.render.image), (Some((v, r, m, t)), Some(_)) if *v == d.version && *m == mip && *t == src_tag && r.intersect(&need) == need);
    if !reuse {
        // Evaluate a tile-aligned rect with one tile of margin to absorb small pans.
        let ts = 256;
        let eval = IRect::new(need.x0.div_euclid(ts) * ts - ts, need.y0.div_euclid(ts) * ts - ts, (need.x1 + ts - 1).div_euclid(ts) * ts + ts, (need.y1 + ts - 1).div_euclid(ts) * ts + ts);
        // Grow it to a quarter-octave size (the texture pool's buckets): zooming out, the
        // next few frames still fit and skip evaluation, and the ones that don't reuse the
        // same few texture sizes at every level instead of allocating.
        let snap = |v: i32| {
            let p = 1i32 << (31 - v.max(ts).leading_zeros());
            let step = (p / 4).max(ts);
            (v + step - 1) / step * step
        };
        let (pw, ph) = (snap(eval.width()) - eval.width(), snap(eval.height()) - eval.height());
        let (l, t) = (pw / 2 / ts * ts, ph / 2 / ts * ts);
        let eval = clip(IRect::new(eval.x0 - l, eval.y0 - t, eval.x1 + pw - l, eval.y1 + ph - t));
        let max = g.engine.ctx.max_texture_size() as i32;
        let eval = if eval.width() > max || eval.height() > max { need } else { eval };
        let img = g.engine.evaluate_with(&EvalRequest { doc, rect: eval, mip, format: EvalFormat::F16, source }, d.stroke.as_ref().map(|st| &st.gpu));
        d.render.image = img;
        d.render.key = Some((d.version, eval, mip, src_tag));
        d.render.mip = mip;
        if let Some(gi) = &d.render.image {
            // Scopes refresh (throttled).
            let now = prisme_platform::now_seconds();
            if now - g.scope_time > 0.25 {
                g.scope_time = now;
                g.scopes.request(&g.rs.device, &g.rs.queue, gi, scope_range);
            }
        }
    } else if let Some(gi) = &d.render.image
        && g.scopes.latest.as_ref().is_some_and(|l| l.range != scope_range)
    {
        // Scope range changed (setting or display headroom): recompute.
        g.scopes.request(&g.rs.device, &g.rs.queue, gi, scope_range);
    }
    if g.scopes.poll(&g.rs.device) {
        ctx.request_repaint();
    }
    // Idle frames: make the tiles the next pan (a ring of half the view around it) and
    // zoom-out (the next level over the same area: twice the view per axis) need resident,
    // a few ms per frame.
    if reuse && !interacting && d.stroke.is_none() {
        let k2 = k * 2;
        let (cx, cy) = ((visible.x0 + visible.x1) / 2, (visible.y0 + visible.y1) / 2);
        let (hw, hh) = (visible.width().max(1), visible.height().max(1));
        let around = |s: i32| IRect::new(cx - hw * s / 2, cy - hh * s / 2, cx + hw * s / 2, cy + hh * s / 2);
        let clip_doc = |r: IRect| if infinite { r } else { r.intersect(&page) };
        let at = |r: IRect, k: i32| IRect::new(r.x0.div_euclid(k), r.y0.div_euclid(k), (r.x1 + k - 1).div_euclid(k), (r.y1 + k - 1).div_euclid(k));
        let mut regions = vec![(mip, at(clip_doc(around(2)), k))];
        if mip < max_mip.min(10) {
            regions.push((mip + 1, at(clip_doc(around(2)), k2)));
        }
        let deadline = prisme_platform::now_seconds() + 0.003;
        if !g.engine.prefetch(doc, &regions, &|| prisme_platform::now_seconds() > deadline) {
            ctx.request_repaint();
        }
    }
    let Some(image) = d.render.image.clone() else { return };
    let params = DisplayParams {
        fb_to_doc: d.view.fb_to_doc(center, ppp),
        level_scale: 1.0 / k as f32,
        nearest: zoom_px >= 2.0,
        doc_size: if infinite { [-1.0; 2] } else { [w as f32, h as f32] },
        checker_px: 8.0 * ppp,
        display_space: prisme_color::RgbSpace::Srgb,
        exposure: 0.0,
        dither: use_dither,
        output,
        bg: theme::CANVAS_BG,
        checker_a: [0.80, 0.80, 0.80],
        checker_b: [0.62, 0.62, 0.62],
    };
    painter.add(egui_wgpu::Callback::new_paint_callback(rect, CanvasCallback { image, params }));

    // Pixel grid at high zoom.
    if pixel_grid && zoom_px >= 10.0 && d.view.rotation == 0.0 {
        let stroke = Stroke::new(1.0 / ppp, Color32::from_black_alpha(40));
        let v = visible;
        if (v.width() + v.height()) < 600 {
            for x in v.x0..=v.x1 {
                painter.line_segment([d.view.doc_to_screen(center, [x as f32, v.y0 as f32]), d.view.doc_to_screen(center, [x as f32, v.y1 as f32])], stroke);
            }
            for y in v.y0..=v.y1 {
                painter.line_segment([d.view.doc_to_screen(center, [v.x0 as f32, y as f32]), d.view.doc_to_screen(center, [v.x1 as f32, y as f32])], stroke);
            }
        }
    }
}

fn overlays(s: &mut AppState, ctx: &egui::Context, rect: Rect, painter: &egui::Painter) {
    let active = s.active;
    let center = rect.center();
    let time = ctx.input(|i| i.time);
    let tool = s.tool;
    let brush_size = match tool {
        ToolKind::Brush => Some(s.settings.tools.brush.size),
        ToolKind::Eraser => Some(s.settings.tools.eraser.size),
        ToolKind::Clone | ToolKind::Heal => Some(s.settings.tools.clone_brush.size),
        ToolKind::Smudge | ToolKind::Blur => Some(s.settings.tools.retouch_brush.size),
        ToolKind::Liquify => Some(s.settings.tools.liquify.size),
        _ => None,
    };
    let hover = s.hover_doc;
    let text_edit = s.text_editing();
    let blocked = blocked_reason(s);
    let brush_size = if blocked.is_some() { None } else { brush_size };
    let d = &mut s.docs[active];
    let view = d.view;
    let to_s = |p: [f32; 2]| view.doc_to_screen(center, p);
    let doc = d.working.as_ref().unwrap_or_else(|| d.history.current());
    // Document border (an infinite canvas has no page).
    if !doc.info.is_infinite() {
        let (w, h) = (doc.info.width as f32, doc.info.height as f32);
        painter.add(Shape::closed_line(vec![to_s([0.0, 0.0]), to_s([w, 0.0]), to_s([w, h]), to_s([0.0, h])], Stroke::new(1.0, Color32::from_black_alpha(160))));
    }

    // Selection marching ants.
    if let Some(sel) = &doc.selection {
        let key = crate::outline::grid_key(sel);
        if d.outline.as_ref().map(|o| o.0) != Some(key) {
            let b = if doc.info.is_infinite() { sel.tile_bounds() } else { doc.bounds() };
            d.outline = Some((key, crate::outline::contours(sel, b)));
        }
        let offset = (time * 16.0) as f32 % 8.0;
        if let Some((_, lines)) = &d.outline {
            for l in lines {
                let pts: Vec<Pos2> = l.iter().map(|p| to_s(*p)).collect();
                painter.add(Shape::line(pts.clone(), Stroke::new(1.0, Color32::BLACK)));
                painter.extend(Shape::dashed_line_with_offset(&pts, Stroke::new(1.0, Color32::WHITE), &[4.0], &[4.0], offset));
            }
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(80));
    }

    // Tool overlays.
    let ov = s.tools.get(&tool).map(|t| t.overlays(doc)).unwrap_or_default();
    for o in ov {
        draw_overlay(painter, &o, &to_s, rect);
    }

    // Text caret.
    if let Some((id, caret)) = text_edit
        && let Some(prisme_core::LayerKind::Text(t)) = doc.layer(id).map(|l| &l.kind) {
            let carets = s.rasterizer.text.caret_positions(t);
            if let Some(c) = carets.get(caret.min(carets.len().saturating_sub(1))) {
                let blink = (time * 2.0) as i64 % 2 == 0;
                if blink {
                    painter.line_segment([to_s([c[0], c[1]]), to_s([c[0], c[2]])], Stroke::new(1.5, theme::ACCENT));
                }
                ctx.request_repaint_after(std::time::Duration::from_millis(250));
            }
        }

    // Lock badge next to the cursor when the tool can't edit its target.
    if let (Some(reason), Some(p)) = (&blocked, hover) {
        let c = to_s(p) + Vec2::new(16.0, 16.0);
        painter.circle_filled(c, 11.0, Color32::from_black_alpha(200));
        painter.circle_stroke(c, 11.0, Stroke::new(1.0, theme::WARN));
        painter.text(c, egui::Align2::CENTER_CENTER, egui_phosphor::regular::LOCK, egui::FontId::proportional(13.0), theme::WARN);
        let galley = painter.layout_no_wrap(reason.clone(), egui::FontId::proportional(11.0), Color32::WHITE);
        let at = c + Vec2::new(16.0, -galley.size().y * 0.5);
        painter.rect_filled(Rect::from_min_size(at, galley.size()).expand(3.0), 3.0, Color32::from_black_alpha(190));
        painter.galley(at, galley, Color32::WHITE);
    }

    // Brush footprint.
    if let (Some(size), Some(p)) = (brush_size, hover) {
        let r = size * 0.5 * view.zoom;
        let c = to_s(p);
        if r > 3.0 {
            painter.circle_stroke(c, r, Stroke::new(1.5, Color32::from_black_alpha(160)));
            painter.circle_stroke(c, r, Stroke::new(0.8, Color32::WHITE));
        } else {
            painter.line_segment([c - Vec2::X * 5.0, c + Vec2::X * 5.0], Stroke::new(1.0, Color32::WHITE));
            painter.line_segment([c - Vec2::Y * 5.0, c + Vec2::Y * 5.0], Stroke::new(1.0, Color32::WHITE));
        }
    }
}

fn draw_overlay(p: &egui::Painter, o: &Overlay, to_s: &dyn Fn([f32; 2]) -> Pos2, clip: Rect) {
    let black = Stroke::new(2.5, Color32::from_black_alpha(140));
    let white = Stroke::new(1.0, Color32::WHITE);
    let accent = Stroke::new(1.5, theme::ACCENT);
    let both = |pts: Vec<Pos2>, closed: bool, dashed: bool| {
        let mut path = pts;
        if closed && !path.is_empty() {
            path.push(path[0]);
        }
        p.add(Shape::line(path.clone(), black));
        if dashed {
            p.extend(Shape::dashed_line(&path, white, 5.0, 4.0));
        } else {
            p.add(Shape::line(path, white));
        }
    };
    match o {
        Overlay::Rect { r, dashed } => both(vec![to_s([r[0], r[1]]), to_s([r[2], r[1]]), to_s([r[2], r[3]]), to_s([r[0], r[3]])], true, *dashed),
        Overlay::Ellipse { r } => {
            let (cx, cy, rx, ry) = ((r[0] + r[2]) * 0.5, (r[1] + r[3]) * 0.5, (r[2] - r[0]) * 0.5, (r[3] - r[1]) * 0.5);
            let pts = (0..96).map(|i| {
                let a = i as f32 / 96.0 * std::f32::consts::TAU;
                to_s([cx + rx * a.cos(), cy + ry * a.sin()])
            }).collect();
            both(pts, true, true);
        }
        Overlay::Polyline { points, closed, dashed } => both(points.iter().map(|q| to_s(*q)).collect(), *closed, *dashed),
        Overlay::Quad(q) => both(q.iter().map(|x| to_s(*x)).collect(), true, false),
        Overlay::Line { a, b, label } => {
            both(vec![to_s(*a), to_s(*b)], false, false);
            if let Some(l) = label {
                let pos = to_s(*b) + Vec2::new(10.0, 10.0);
                let galley = p.layout_no_wrap(l.clone(), egui::FontId::proportional(12.0), Color32::WHITE);
                let r = Rect::from_min_size(pos, galley.size()).expand(4.0);
                p.rect_filled(r, 4.0, Color32::from_black_alpha(190));
                p.galley(pos, galley, Color32::WHITE);
            }
        }
        Overlay::Circle { c, radius } => {
            let cs = to_s(*c);
            let r = (to_s([c[0] + radius, c[1]]) - cs).length();
            p.circle_stroke(cs, r, black);
            p.circle_stroke(cs, r, white);
        }
        Overlay::Handles(hs) => {
            for h in hs {
                let c = to_s(*h);
                let r = Rect::from_center_size(c, Vec2::splat(8.0));
                p.rect_filled(r, 1.0, Color32::WHITE);
                p.rect_stroke(r, 1.0, accent, egui::StrokeKind::Outside);
            }
        }
        Overlay::Crosshair(c) => {
            let cs = to_s(*c);
            for (a, b) in [(Vec2::new(-8.0, 0.0), Vec2::new(8.0, 0.0)), (Vec2::new(0.0, -8.0), Vec2::new(0.0, 8.0))] {
                p.line_segment([cs + a, cs + b], black);
                p.line_segment([cs + a, cs + b], white);
            }
            p.circle_stroke(cs, 4.0, white);
        }
        Overlay::Shade(r) => {
            let q = [to_s([r[0], r[1]]), to_s([r[2], r[1]]), to_s([r[2], r[3]]), to_s([r[0], r[3]])];
            let (minx, miny) = (q.iter().map(|p| p.x).fold(f32::MAX, f32::min), q.iter().map(|p| p.y).fold(f32::MAX, f32::min));
            let (maxx, maxy) = (q.iter().map(|p| p.x).fold(f32::MIN, f32::max), q.iter().map(|p| p.y).fold(f32::MIN, f32::max));
            let inner = Rect::from_min_max(Pos2::new(minx, miny), Pos2::new(maxx, maxy));
            let shade = Color32::from_black_alpha(130);
            p.rect_filled(Rect::from_min_max(clip.min, Pos2::new(clip.max.x, inner.min.y)), 0.0, shade);
            p.rect_filled(Rect::from_min_max(Pos2::new(clip.min.x, inner.max.y), clip.max), 0.0, shade);
            p.rect_filled(Rect::from_min_max(Pos2::new(clip.min.x, inner.min.y), Pos2::new(inner.min.x, inner.max.y)), 0.0, shade);
            p.rect_filled(Rect::from_min_max(Pos2::new(inner.max.x, inner.min.y), Pos2::new(clip.max.x, inner.max.y)), 0.0, shade);
        }
        Overlay::Label { pos, text } => {
            let at = to_s(*pos) + Vec2::new(10.0, 10.0);
            let galley = p.layout_no_wrap(text.clone(), egui::FontId::proportional(12.0), Color32::WHITE);
            p.rect_filled(Rect::from_min_size(at, galley.size()).expand(4.0), 4.0, Color32::from_black_alpha(190));
            p.galley(at, galley, Color32::WHITE);
        }
    }
}

fn doc_tabs(s: &mut AppState, ui: &mut Ui) {
    if s.docs.is_empty() {
        return;
    }
    let mut close = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        for i in 0..s.docs.len() {
            let selected = i == s.active;
            let title = s.docs[i].title();
            let frame = egui::Frame::new()
                .fill(if selected { theme::BG2 } else { theme::BG1 })
                .corner_radius(egui::CornerRadius { nw: 5, ne: 5, sw: 0, se: 0 })
                .inner_margin(egui::Margin::symmetric(8, 3));
            frame.show(ui, |ui| {
                if ui.add(egui::Label::new(egui::RichText::new(title).color(if selected { theme::TEXT } else { theme::TEXT_DIM })).sense(Sense::click())).clicked()
                    && !selected {
                        crate::stroke::commit_liquify(s);
                        if let Some(d) = s.doc_mut() {
                            d.cancel_working();
                        }
                        s.active = i;
                    }
                if ui.add(egui::Button::new(egui_phosphor::regular::X).frame(false).small()).clicked() {
                    close = Some(i);
                }
            });
        }
    });
    if let Some(i) = close {
        crate::commands::close_doc(s, i);
    }
}

fn welcome(s: &mut AppState, ui: &mut Ui) {
    ui.vertical_centered(|ui| {
        ui.add_space(ui.available_height() * 0.25);
        ui.label(egui::RichText::new("Prisme").size(42.0).color(theme::ACCENT).strong());
        ui.label(egui::RichText::new("Fast, colour-accurate, node-based image editing").color(theme::TEXT_DIM));
        ui.add_space(20.0);
        ui.horizontal(|ui| {
            ui.add_space((ui.available_width() - 260.0).max(0.0) / 2.0);
            if ui.add(egui::Button::new("New image…").min_size(Vec2::new(120.0, 32.0))).clicked() {
                s.dialogs.new_doc.open = true;
            }
            if ui.add(egui::Button::new("Open…").min_size(Vec2::new(120.0, 32.0))).clicked() {
                crate::jobs::open_dialog(s);
            }
        });
        ui.add_space(12.0);
        ui.label(egui::RichText::new("…or drop images anywhere").small().color(theme::TEXT_DIM));
        let recent: Vec<String> = s.settings.recent_files.iter().take(8).cloned().collect();
        if !recent.is_empty() && !cfg!(target_arch = "wasm32") {
            ui.add_space(18.0);
            ui.label(egui::RichText::new("RECENT").small().strong().color(theme::TEXT_DIM));
            for p in recent {
                let name = std::path::Path::new(&p).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| p.clone());
                if ui.link(name).on_hover_text(&p).clicked() {
                    crate::jobs::open_recent(s, &p);
                }
            }
        }
    });
}
