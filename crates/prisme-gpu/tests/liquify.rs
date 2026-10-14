//! Liquify warp on the GPU (`k_liquify`: bake readback and the evaluation splice) against the
//! CPU reference `prisme_tools::liquify::warp_rect`. Uses any available adapter.

use prisme_core::layer::{Layer, LayerKind};
use prisme_core::{Document, IRect, PixelFormat, TileGrid};
use prisme_gpu::{Engine, EvalFormat, EvalRequest, GpuContext, GpuStroke, Readback, RenderSource, UploadKey};
use prisme_tools::liquify::{warp_rect, LiquifyDab, LiquifyField, LiquifyMode};
use std::sync::Arc;

fn engine() -> Option<Engine> {
    let ctx = pollster::block_on(GpuContext::headless());
    if ctx.is_none() {
        eprintln!("no GPU adapter; skipping");
    }
    ctx.map(Engine::new)
}

/// 300x200 doc: grey background + a raster layer with a gradient patch, half-transparent stripes.
fn doc() -> Document {
    let mut d = Document::with_background("t", 300, 200, Some([0.5, 0.5, 0.5, 1.0]));
    let id = d.alloc_layer_id();
    let mut g = TileGrid::rgba(PixelFormat::F32);
    g.map_rect(&IRect::new(60, 30, 240, 170), |x, y, p| {
        let a = if (x / 8) % 3 == 0 { 0.4 } else { 1.0 };
        p.copy_from_slice(&[x as f32 / 300.0, y as f32 / 200.0, ((x ^ y) & 15) as f32 / 15.0, a]);
    });
    d.layers.push(Layer::new(id, "paint", LayerKind::Raster(Arc::new(g))));
    d.active_layer = Some(id);
    d
}

fn field(rect: IRect) -> LiquifyField {
    let mut f = LiquifyField::new(rect);
    let dab = |mode, center, amount, motion| LiquifyDab { mode, center, radius: 45.0, amount, density: 0.5, motion };
    for i in 0..12 {
        f.apply_dab(&dab(LiquifyMode::Forward, [100.0 + i as f32 * 4.0, 90.0], 1.0, [4.0, 1.5]));
    }
    f.apply_dab(&dab(LiquifyMode::Bloat, [190.0, 120.0], 0.4, [0.0; 2]));
    f.apply_dab(&dab(LiquifyMode::TwirlCw, [150.0, 60.0], 0.8, [0.0; 2]));
    f.apply_dab(&dab(LiquifyMode::Freeze, [120.0, 140.0], 1.0, [0.0; 2]));
    f.apply_dab(&dab(LiquifyMode::Pucker, [120.0, 130.0], 0.5, [0.0; 2]));
    f
}

fn begin(e: &mut Engine, d: &Document, f: &LiquifyField, sel: Option<Arc<TileGrid>>) -> GpuStroke {
    let id = d.active_layer.unwrap();
    let base = d.layer(id).unwrap().raster().unwrap().clone();
    let mut s = e.liquify_begin(UploadKey::Content(id.0), base, sel, f.rect, f.step, [f.w, f.h]);
    // Upload in two uneven pieces (partial updates).
    let a = IRect::new(0, 0, f.w, f.h / 3);
    let b = IRect::new(0, f.h / 3, f.w, f.h);
    e.liquify_update(&mut s, a, &f.read(a));
    e.liquify_update(&mut s, b, &f.read(b));
    s
}

fn max_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len());
    let mut m = 0.0f32;
    let mut n = 0;
    for (x, y) in a.chunks(4).zip(b.chunks(4)) {
        // Fully transparent pixels carry no colour.
        if x[3] == 0.0 && y[3] == 0.0 {
            continue;
        }
        let d = x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0.0, f32::max);
        m = m.max(d);
        n += (d > 2e-3) as usize;
    }
    (m, n)
}

#[test]
fn bake_matches_cpu_reference() {
    let Some(mut e) = engine() else { return };
    let d = doc();
    let base = d.active().unwrap().raster().unwrap().clone();
    let f = field(IRect::new(0, 0, 300, 200));
    let r = f.active_rect();
    assert!(!r.is_empty());
    let mut sel = TileGrid::mask();
    sel.map_rect(&IRect::new(0, 0, 300, 200), |x, _, p| p[0] = (x as f32 / 300.0).clamp(0.0, 1.0));
    let sel = Arc::new(sel);
    for (with_sel, bicubic) in [(false, false), (false, true), (true, true)] {
        let s = begin(&mut e, &d, &f, with_sel.then(|| sel.clone()));
        let rb = e.liquify_readback(&s, r, bicubic).expect("readback");
        assert_eq!(rb.rect, r);
        let gpu = rb.wait(&e.ctx.device);
        let cpu = warp_rect(&base, &f, with_sel.then_some(&*sel), r, bicubic);
        let (m, n) = max_diff(&gpu, &cpu);
        // Only float rounding differs.
        assert!(n == 0 && m < 1e-4, "sel {with_sel} bicubic {bicubic}: max {m}, {n} px over tolerance");
    }
}

#[test]
fn preview_splices_the_warp_into_evaluation() {
    let Some(mut e) = engine() else { return };
    let d = doc();
    let id = d.active_layer.unwrap();
    let base = d.layer(id).unwrap().raster().unwrap().clone();
    let f = field(IRect::new(0, 0, 300, 200));
    let s = begin(&mut e, &d, &f, None);
    let rect = IRect::new(0, 0, 300, 200);
    let req = |doc| EvalRequest { doc, rect, mip: 0, format: EvalFormat::F32, source: RenderSource::LayerStack };
    let img = e.evaluate_with(&req(&d), Some(&s)).unwrap();
    let gpu = Readback::start(&e.ctx.device, &e.ctx.queue, &img, rect).wait(&e.ctx.device);
    // Reference: the layer replaced by the CPU warp (bilinear, like the preview).
    let mut warped = d.clone();
    let g = warped.layer_mut(id).unwrap().raster_mut().unwrap();
    g.write_rect(&rect, &warp_rect(&base, &f, None, rect, false));
    let cpu_img = e.evaluate(&req(&warped)).unwrap();
    let cpu = Readback::start(&e.ctx.device, &e.ctx.queue, &cpu_img, rect).wait(&e.ctx.device);
    let (m, n) = max_diff(&gpu, &cpu);
    assert!(n == 0 && m < 1e-3, "max {m}, {n} px over tolerance");
    // And it really is deformed.
    let plain = e.evaluate(&req(&d)).unwrap();
    let plain = Readback::start(&e.ctx.device, &e.ctx.queue, &plain, rect).wait(&e.ctx.device);
    assert!(max_diff(&gpu, &plain).1 > 1000);
    // Zoomed out (mip 2) evaluates too.
    let img = e.evaluate_with(&EvalRequest { doc: &d, rect: IRect::new(0, 0, 75, 50), mip: 2, format: EvalFormat::F16, source: RenderSource::LayerStack }, Some(&s));
    assert!(img.is_some());
}
