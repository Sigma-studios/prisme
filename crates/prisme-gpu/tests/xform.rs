//! Free Transform preview on the GPU (`k_xform` spliced into evaluation) against the CPU
//! reference `prisme_tools::transform::warp`. Uses any available adapter.

use prisme_core::geom::Homography;
use prisme_core::layer::{Layer, LayerKind, LayerMask};
use prisme_core::{Document, IRect, PixelFormat, TileGrid};
use prisme_gpu::{Engine, EvalFormat, EvalRequest, GpuContext, Readback, RenderSource, UploadKey};
use prisme_tools::transform::{merge_over, quad_bounds, split_by_selection, warp, Resample};
use std::sync::Arc;

fn engine() -> Option<Engine> {
    let ctx = pollster::block_on(GpuContext::headless());
    if ctx.is_none() {
        eprintln!("no GPU adapter; skipping");
    }
    ctx.map(Engine::new)
}

/// 300x200 doc: grey background + a raster layer with a gradient patch, half-transparent
/// stripes, and a hide-all mask revealing a disc.
fn doc() -> Document {
    let mut d = Document::with_background("t", 300, 200, Some([0.5, 0.5, 0.5, 1.0]));
    let id = d.alloc_layer_id();
    let mut g = TileGrid::rgba(PixelFormat::F32);
    g.map_rect(&IRect::new(60, 30, 240, 170), |x, y, p| {
        let a = if (x / 8) % 3 == 0 { 0.4 } else { 1.0 };
        p.copy_from_slice(&[x as f32 / 300.0, y as f32 / 200.0, ((x ^ y) & 15) as f32 / 15.0, a]);
    });
    let mut m = TileGrid::mask();
    m.map_rect(&IRect::new(60, 30, 240, 170), |x, y, p| {
        let r = ((x as f32 - 150.0).powi(2) + (y as f32 - 100.0).powi(2)).sqrt();
        p[0] = (1.0 - (r - 60.0) / 10.0).clamp(0.0, 1.0);
    });
    let mut l = Layer::new(id, "paint", LayerKind::Raster(Arc::new(g)));
    l.mask = Some(LayerMask { grid: Arc::new(m), enabled: true, default_visible: false });
    d.layers.push(l);
    d.active_layer = Some(id);
    d
}

fn max_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len());
    let mut m = 0.0f32;
    let mut n = 0;
    for (x, y) in a.chunks(4).zip(b.chunks(4)) {
        let d = x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0.0, f32::max);
        m = m.max(d);
        n += (d > 2e-3) as usize;
    }
    (m, n)
}

#[test]
fn preview_matches_cpu_warp() {
    let Some(mut e) = engine() else { return };
    let d = doc();
    let id = d.active_layer.unwrap();
    let layer = d.layer(id).unwrap().clone();
    let grid = layer.raster().unwrap().clone();
    let src = [60.0, 30.0, 240.0, 170.0];
    let quad = [[80.0, 20.0], [250.0, 50.0], [230.0, 190.0], [40.0, 150.0]];
    let h = Homography::rect_to_quad(src, quad);
    let bounds = quad_bounds(&quad).expand(1);
    let mut sel = TileGrid::mask();
    sel.map_rect(&IRect::new(0, 0, 150, 200), |_, _, p| p[0] = 1.0);
    let sel = Arc::new(sel);
    let rect = IRect::new(0, 0, 300, 200);
    fn req(doc: &Document, rect: IRect) -> EvalRequest<'_> {
        EvalRequest { doc, rect, mip: 0, format: EvalFormat::F32, source: RenderSource::LayerStack }
    }
    for with_sel in [false, true] {
        // What the tool hands over: the moving pixels (and mask), the rest left in the layer.
        let (source, rest) = match with_sel {
            true => {
                let (f, r) = split_by_selection(&grid, &sel);
                (Arc::new(f), r)
            }
            false => (grid.clone(), TileGrid::rgba(PixelFormat::F32)),
        };
        let mask = (!with_sel).then(|| (layer.mask.as_ref().unwrap().grid.clone(), 0.0));
        let mut s = e.xform_begin(UploadKey::Content(id.0), source.clone(), IRect::new(60, 30, 240, 170), mask);
        e.xform_update(&mut s, &h);
        let mut preview = d.clone();
        preview.layer_mut(id).unwrap().kind = LayerKind::Raster(Arc::new(rest.clone()));
        let img = e.evaluate_with(&req(&preview, rect), Some(&s)).unwrap();
        let gpu = Readback::start(&e.ctx.device, &e.ctx.queue, &img, rect).wait(&e.ctx.device);
        // Reference: the tool's CPU render at the same (bilinear) quality.
        let mut cpu_doc = d.clone();
        let l = cpu_doc.layer_mut(id).unwrap();
        l.kind = LayerKind::Raster(Arc::new(merge_over(&rest, &warp(&source, &h, bounds, Resample::Bilinear))));
        if !with_sel {
            let m = l.mask.as_mut().unwrap();
            m.grid = Arc::new(warp(&m.grid, &h, bounds, Resample::Bilinear));
        }
        let cpu_img = e.evaluate(&req(&cpu_doc, rect)).unwrap();
        let cpu = Readback::start(&e.ctx.device, &e.ctx.queue, &cpu_img, rect).wait(&e.ctx.device);
        let (m, n) = max_diff(&gpu, &cpu);
        assert!(n == 0 && m < 1e-3, "sel {with_sel}: max {m}, {n} px over tolerance");
        // It really moved.
        let plain = e.evaluate(&req(&d, rect)).unwrap();
        let plain = Readback::start(&e.ctx.device, &e.ctx.queue, &plain, rect).wait(&e.ctx.device);
        assert!(max_diff(&gpu, &plain).1 > 1000, "sel {with_sel}: preview didn't move");
        // Zoomed out evaluates too.
        let img = e.evaluate_with(&EvalRequest { doc: &preview, rect: IRect::new(0, 0, 75, 50), mip: 2, format: EvalFormat::F16, source: RenderSource::LayerStack }, Some(&s));
        assert!(img.is_some());
    }
}
