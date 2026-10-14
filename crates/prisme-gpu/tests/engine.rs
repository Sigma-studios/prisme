//! Headless GPU tests: compositing correctness against a CPU reference, and every node
//! kind compiles/runs without validation errors.

use prisme_core::graph::ParamValue;
use prisme_core::layer::{BlendMode, Layer, LayerKind};
use prisme_core::{Document, IRect, PixelFormat, TileGrid};
use prisme_gpu::{Engine, EvalFormat, EvalRequest, GpuContext, Readback, RenderSource};
use std::sync::Arc;

fn ctx() -> Option<GpuContext> {
    pollster::block_on(GpuContext::headless())
}

fn eval(engine: &mut Engine, doc: &Document, rect: IRect, mip: u32, source: RenderSource) -> Vec<f32> {
    let img = engine.evaluate(&EvalRequest { doc, rect, mip, format: EvalFormat::F32, source }).unwrap();
    let rb = Readback::start(&engine.ctx.device, &engine.ctx.queue, &img, rect);
    rb.wait(&engine.ctx.device)
}

fn two_layer_doc() -> Document {
    let mut doc = Document::with_background("t", 300, 200, Some([0.2, 0.4, 0.6, 1.0]));
    let id = doc.alloc_layer_id();
    let mut g = TileGrid::rgba(PixelFormat::F32);
    g.map_rect(&IRect::new(50, 50, 280, 150), |_, _, p| p.copy_from_slice(&[0.9, 0.1, 0.05, 0.5]));
    let mut l = Layer::new(id, "top", LayerKind::Raster(Arc::new(g)));
    l.opacity = 0.8;
    doc.layers.push(l);
    doc
}

#[test]
fn composite_matches_cpu() {
    let Some(ctx) = ctx() else { eprintln!("no GPU adapter; skipping"); return };
    let mut engine = Engine::new(ctx);
    let doc = two_layer_doc();
    let px = eval(&mut engine, &doc, doc.bounds(), 0, RenderSource::LayerStack);
    let at = |x: usize, y: usize| { let i = (y * 300 + x) * 4; [px[i], px[i + 1], px[i + 2], px[i + 3]] };
    // outside top layer: background
    let bg = at(10, 10);
    assert!((bg[0] - 0.2).abs() < 1e-3 && (bg[3] - 1.0).abs() < 1e-4, "{bg:?}");
    // inside: normal blend with alpha 0.5*0.8 = 0.4 (premultiplied output)
    let a = 0.4;
    let c = at(100, 100);
    let expect = [0.9 * a + 0.2 * (1.0 - a), 0.1 * a + 0.4 * (1.0 - a), 0.05 * a + 0.6 * (1.0 - a)];
    for k in 0..3 {
        assert!((c[k] - expect[k]).abs() < 1e-3, "{c:?} vs {expect:?}");
    }
    // graph output == layer stack for default graph
    let px2 = eval(&mut engine, &doc, doc.bounds(), 0, RenderSource::Graph);
    assert_eq!(px.len(), px2.len());
    let max_diff = px.iter().zip(&px2).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 1e-5);
}

#[test]
fn mip_levels_average() {
    let Some(ctx) = ctx() else { return };
    let mut engine = Engine::new(ctx);
    let mut doc = Document::with_background("t", 1024, 1024, Some([0.0, 0.0, 0.0, 1.0]));
    let id = doc.active_layer.unwrap();
    // checkerboard of 1px -> mip 1 should be uniform grey 0.5
    doc.layer_mut(id).unwrap().raster_mut().unwrap().map_rect(&IRect::new(0, 0, 1024, 1024), |x, y, p| {
        let v = ((x + y) % 2) as f32;
        p.copy_from_slice(&[v, v, v, 1.0]);
    });
    let px = eval(&mut engine, &doc, IRect::new(0, 0, 512, 512), 1, RenderSource::LayerStack);
    for v in px.chunks(4).take(1000) {
        assert!((v[0] - 0.5).abs() < 1e-3, "{v:?}");
    }
}

#[test]
fn blend_modes_run() {
    let Some(ctx) = ctx() else { return };
    let mut engine = Engine::new(ctx);
    let mut doc = two_layer_doc();
    for m in BlendMode::ALL {
        let id = doc.layers[1].id;
        doc.layer_mut(id).unwrap().blend = m;
        let px = eval(&mut engine, &doc, IRect::new(90, 90, 110, 110), 0, RenderSource::LayerStack);
        assert!(px.iter().all(|v| v.is_finite()), "{m:?}");
    }
}

#[test]
fn every_node_kind_evaluates() {
    let Some(ctx) = ctx() else { return };
    let mut engine = Engine::new(ctx);
    let base = two_layer_doc();
    for desc in &prisme_graph::registry().kinds {
        if desc.id == prisme_core::graph::KIND_OUTPUT { continue; }
        let mut doc = base.clone();
        let mut params = prisme_graph::default_params(desc.id);
        if desc.id == "layer" { params.insert("layer".into(), ParamValue::Layer(Some(doc.layers[1].id))); }
        let n = doc.graph.add_node(desc.id, params, [0.0, 0.0]);
        let stack = doc.graph.input(doc.graph.output, 0).unwrap().0;
        if !desc.inputs.is_empty() {
            doc.graph.connect(stack, 0, n, 0);
            if desc.id == "blend" || desc.id == "mix" {
                doc.graph.connect(stack, 0, n, 1);
            }
        }
        doc.graph.connect(n, 0, doc.graph.output, 0);
        for mip in [0, 1] {
            let s = 1 << mip;
            let px = eval(&mut engine, &doc, IRect::new(0, 0, 300 / s, 200 / s), mip, RenderSource::Graph);
            assert!(px.iter().all(|v| v.is_finite()), "{} produced non-finite values", desc.id);
        }
    }
}

#[test]
fn invert_twice_is_identity() {
    let Some(ctx) = ctx() else { return };
    let mut engine = Engine::new(ctx);
    let mut doc = two_layer_doc();
    let stack = doc.graph.input(doc.graph.output, 0).unwrap().0;
    let a = doc.graph.add_node("invert", Default::default(), [0.0; 2]);
    let b = doc.graph.add_node("invert", Default::default(), [0.0; 2]);
    doc.graph.connect(stack, 0, a, 0);
    doc.graph.connect(a, 0, b, 0);
    doc.graph.connect(b, 0, doc.graph.output, 0);
    let px = eval(&mut engine, &doc, doc.bounds(), 0, RenderSource::Graph);
    let raw = eval(&mut engine, &doc, doc.bounds(), 0, RenderSource::LayerStack);
    let max_diff = px.iter().zip(&raw).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 1e-4, "{max_diff}");
}

#[test]
fn cache_reuses_upstream() {
    let Some(ctx) = ctx() else { return };
    let mut engine = Engine::new(ctx);
    let mut doc = two_layer_doc();
    let stack = doc.graph.input(doc.graph.output, 0).unwrap().0;
    let blur = doc.graph.add_node("gaussian_blur", prisme_graph::default_params("gaussian_blur"), [0.0; 2]);
    let exp = doc.graph.add_node("exposure", prisme_graph::default_params("exposure"), [0.0; 2]);
    doc.graph.connect(stack, 0, blur, 0);
    doc.graph.connect(blur, 0, exp, 0);
    doc.graph.connect(exp, 0, doc.graph.output, 0);
    let _ = eval(&mut engine, &doc, doc.bounds(), 0, RenderSource::Graph);
    doc.graph.nodes.get_mut(&exp).unwrap().params.insert("exposure".into(), ParamValue::Float(1.0));
    let _ = eval(&mut engine, &doc, doc.bounds(), 0, RenderSource::Graph);
    // only exposure + output re-ran
    assert_eq!(engine.last_stats.nodes_run, 2, "{:?}", engine.last_stats);
}

#[test]
fn blur_extends_edges_by_default() {
    let Some(ctx) = ctx() else { return };
    let mut engine = Engine::new(ctx);
    let mut doc = Document::with_background("t", 200, 120, Some([0.3, 0.5, 0.7, 1.0]));
    let stack = doc.graph.input(doc.graph.output, 0).unwrap().0;
    let mut p = prisme_graph::default_params("gaussian_blur");
    p.insert("radius".into(), ParamValue::Float(20.0));
    let blur = doc.graph.add_node("gaussian_blur", p, [0.0; 2]);
    doc.graph.connect(stack, 0, blur, 0);
    doc.graph.connect(blur, 0, doc.graph.output, 0);
    let px = eval(&mut engine, &doc, doc.bounds(), 0, RenderSource::Graph);
    assert!((px[3] - 1.0).abs() < 1e-3, "corner alpha {} (should stay opaque)", px[3]);
    doc.graph.nodes.get_mut(&blur).unwrap().params.insert("edges".into(), ParamValue::Choice(1));
    let px = eval(&mut engine, &doc, doc.bounds(), 0, RenderSource::Graph);
    assert!(px[3] < 0.6, "transparent edges should fade, got {}", px[3]);
}

/// Infinite canvas: content far from the origin renders, filters read transparency past the
/// content (no page edge to extend), and a region rendered in tiles (export / Copy Merged
/// jobs) equals the one-shot render, with the graph (Levels) and blend modes applied.
#[test]
fn infinite_canvas_far_content() {
    let Some(ctx) = ctx() else { return };
    let mut engine = Engine::new(ctx);
    let mut doc = Document::infinite("t", 300, 200, None);
    let id = doc.active_layer.unwrap();
    doc.layer_mut(id).unwrap().raster_mut().unwrap().map_rect(&IRect::new(-5000, -5000, -4900, -4900), |_, _, p| p.copy_from_slice(&[0.8, 0.3, 0.1, 1.0]));
    let top = doc.alloc_layer_id();
    let mut g = TileGrid::rgba(PixelFormat::F32);
    g.map_rect(&IRect::new(-4950, -4950, -4850, -4850), |_, _, p| p.copy_from_slice(&[0.2, 0.6, 0.9, 0.7]));
    let mut l = Layer::new(top, "top", LayerKind::Raster(Arc::new(g)));
    l.blend = BlendMode::Multiply;
    doc.layers.push(l);
    let stack = doc.graph.input(doc.graph.output, 0).unwrap().0;
    let mut p = prisme_graph::default_params("gaussian_blur");
    p.insert("radius".into(), ParamValue::Float(8.0));
    let blur = doc.graph.add_node("gaussian_blur", p, [0.0; 2]);
    let mut p = prisme_graph::default_params("levels");
    p.insert("m.gamma".into(), ParamValue::Float(1.6));
    let lv = doc.graph.add_node("levels", p, [0.0; 2]);
    doc.graph.connect(stack, 0, blur, 0);
    doc.graph.connect(blur, 0, lv, 0);
    doc.graph.connect(lv, 0, doc.graph.output, 0);
    let rect = IRect::new(-5040, -5040, -4800, -4800);
    let whole = eval(&mut engine, &doc, rect, 0, RenderSource::Graph);
    let at = |px: &[f32], x: i32, y: i32| { let i = (((y - rect.y0) * rect.width() + x - rect.x0) * 4) as usize; px[i + 3] };
    assert!(at(&whole, -4980, -4980) > 0.99, "content renders far from the origin");
    assert!(at(&whole, -5030, -4980) < 1e-4, "no edge extension past the content");
    assert!(at(&whole, -5000, -4980) > 0.3 && at(&whole, -5000, -4980) < 0.7, "blur fades into transparency");
    // Tiled render (the export / Copy Merged path) == one-shot render.
    let mut tiled = vec![0.0f32; whole.len()];
    let w = rect.width() as usize;
    for t in [IRect::new(-5040, -5040, -4900, -4800), IRect::new(-4900, -5040, -4800, -4800)] {
        let px = eval(&mut engine, &doc, t, 0, RenderSource::Graph);
        let tw = t.width() as usize;
        for y in 0..t.height() as usize {
            let d = ((t.y0 - rect.y0) as usize + y) * w + (t.x0 - rect.x0) as usize;
            tiled[d * 4..(d + tw) * 4].copy_from_slice(&px[y * tw * 4..(y + 1) * tw * 4]);
        }
    }
    let max_diff = whole.iter().zip(&tiled).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 1e-5, "tiled render differs by {max_diff}");
    // The graph is applied (differs from the raw layer stack).
    let raw = eval(&mut engine, &doc, rect, 0, RenderSource::LayerStack);
    assert!(whole.iter().zip(&raw).any(|(a, b)| (a - b).abs() > 0.01));
}
