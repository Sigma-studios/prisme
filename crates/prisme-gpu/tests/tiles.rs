//! Resident tile store: GPU-built mip levels against the CPU reference `Tile::downsample`,
//! incremental updates, reuse across views, eviction. Uses any available adapter.

use prisme_core::layer::{Layer, LayerKind, LayerMask};
use prisme_core::tile::{Tile, TileCoord, TileGrid};
use prisme_core::{Document, IRect, PixelFormat};
use prisme_gpu::{Engine, EvalFormat, EvalRequest, GpuContext, Readback, RenderSource};
use std::sync::Arc;

fn engine() -> Option<Engine> {
    let ctx = pollster::block_on(GpuContext::headless());
    if ctx.is_none() {
        eprintln!("no GPU adapter; skipping");
    }
    ctx.map(Engine::new)
}

fn eval(e: &mut Engine, d: &Document, rect: IRect, mip: u32) -> Vec<f32> {
    let img = e.evaluate(&EvalRequest { doc: d, rect, mip, format: EvalFormat::F32, source: RenderSource::LayerStack }).unwrap();
    Readback::start(&e.ctx.device, &e.ctx.queue, &img, rect).wait(&e.ctx.device)
}

/// CPU mip pyramid level `mip` of `g` (the reference the old CPU path uploaded).
fn cpu_level(g: &TileGrid, mip: u32) -> TileGrid {
    let mut cur = g.clone();
    for _ in 0..mip {
        let mut next = TileGrid::new(g.format, g.channels);
        let parents: std::collections::HashSet<TileCoord> = cur.tiles.keys().map(|c| TileCoord::new(c.x.div_euclid(2), c.y.div_euclid(2))).collect();
        for p in parents {
            let kids = [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(dx, dy)| cur.get_tile(TileCoord::new(p.x * 2 + dx, p.y * 2 + dy)).map(|t| t.as_ref()));
            next.tiles.insert(p, Arc::new(Tile::downsample(kids, g.format, g.channels)));
        }
        cur = next;
    }
    cur
}

/// A layer with gradients, varying alpha, holes and isolated tiles.
fn grid() -> TileGrid {
    let mut g = TileGrid::rgba(PixelFormat::F32);
    g.map_rect(&IRect::new(-300, -200, 900, 700), |x, y, p| {
        let a = (((x * 7 + y * 3) & 63) as f32 / 63.0).max(0.05);
        p.copy_from_slice(&[(x + 300) as f32 / 1200.0, (y + 200) as f32 / 900.0, ((x ^ y) & 31) as f32 / 31.0, a]);
    });
    g.map_rect(&IRect::new(2000, 1500, 2100, 1600), |_, _, p| p.copy_from_slice(&[1.0, 0.0, 0.5, 1.0]));
    g.tiles.remove(&TileCoord::new(1, 1));
    g
}

fn doc(g: TileGrid) -> Document {
    let mut d = Document::infinite("t", 1000, 800, None);
    let id = d.active_layer.unwrap();
    d.layer_mut(id).unwrap().kind = LayerKind::Raster(Arc::new(g));
    d
}

fn premul(px: &[f32]) -> Vec<f32> {
    px.chunks(4).flat_map(|p| [p[0] * p[3], p[1] * p[3], p[2] * p[3], p[3]]).collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn levels_match_cpu_downsample() {
    let Some(mut e) = engine() else { return };
    let g = grid();
    let d = doc(g.clone());
    for mip in [1, 2, 3, 5] {
        let k = 1 << mip;
        let rect = IRect::new(-512 / k, -512 / k, 2560 / k, 2048 / k);
        let gpu = eval(&mut e, &d, rect, mip);
        let cpu = premul(&cpu_level(&g, mip).read_rect(&rect));
        let m = max_diff(&gpu, &cpu);
        assert!(m < 1e-5, "mip {mip}: max diff {m}");
    }
    let (up, built, _) = e.tile_stats();
    assert_eq!(up as usize, g.tiles.len(), "each tile uploaded once");
    assert!(built > 0);
}

#[test]
fn views_reuse_resident_tiles() {
    let Some(mut e) = engine() else { return };
    let d = doc(grid());
    let views = [(0, IRect::new(-300, -200, 700, 500)), (1, IRect::new(-150, -100, 450, 350)), (2, IRect::new(-75, -50, 225, 175)), (0, IRect::new(-300, -200, 700, 500))];
    for (mip, r) in views {
        eval(&mut e, &d, r, mip);
    }
    let before = e.tile_stats();
    // Pan around and zoom back and forth: nothing is uploaded or rebuilt.
    for (mip, r) in views.iter().rev().chain(views.iter()) {
        eval(&mut e, &d, r.translate(16 >> mip, 8 >> mip), *mip);
    }
    let after = e.tile_stats();
    assert_eq!((before.0, before.1), (after.0, after.1));
}

#[test]
fn edits_update_built_levels() {
    let Some(mut e) = engine() else { return };
    let mut g = grid();
    let rect = IRect::new(-128, -128, 512, 448);
    let mut d = doc(g.clone());
    eval(&mut e, &d, rect, 2);
    let (up0, _, _) = e.tile_stats();
    // Paint into one tile and erase another entirely.
    g.map_rect(&IRect::new(10, 10, 60, 60), |_, _, p| p.copy_from_slice(&[0.0, 1.0, 0.0, 1.0]));
    g.tiles.remove(&TileCoord::new(2, 2));
    d = doc(g.clone());
    let gpu = eval(&mut e, &d, rect, 2);
    let cpu = premul(&cpu_level(&g, 2).read_rect(&rect));
    assert!(max_diff(&gpu, &cpu) < 1e-5);
    assert_eq!(e.tile_stats().0, up0 + 1, "only the edited tile is uploaded again");
}

#[test]
fn reveal_all_mask_levels_read_missing_tiles_as_visible() {
    let Some(mut e) = engine() else { return };
    let mut m = TileGrid::mask();
    m.map_rect(&IRect::new(0, 0, 256, 256), |_, _, p| p[0] = 0.0);
    let mut d = Document::infinite("t", 512, 512, None);
    let id = d.active_layer.unwrap();
    let mut g = TileGrid::rgba(PixelFormat::F32);
    g.map_rect(&IRect::new(0, 0, 512, 512), |_, _, p| p.copy_from_slice(&[1.0, 1.0, 1.0, 1.0]));
    let l: &mut Layer = d.layer_mut(id).unwrap();
    l.kind = LayerKind::Raster(Arc::new(g));
    l.mask = Some(LayerMask { grid: Arc::new(m), enabled: true, default_visible: true });
    // Level 1: the masked tile is the top-left quarter of the only level tile; the other
    // three quarters have no mask tile and must stay visible.
    let px = eval(&mut e, &d, IRect::new(0, 0, 256, 256), 1);
    let a = |x: usize, y: usize| px[(y * 256 + x) * 4 + 3];
    assert!(a(10, 10) < 1e-6, "hidden: {}", a(10, 10));
    assert!(a(200, 10) > 0.999 && a(10, 200) > 0.999 && a(200, 200) > 0.999, "{} {} {}", a(200, 10), a(10, 200), a(200, 200));
}

#[test]
fn eviction_keeps_results_correct() {
    let Some(mut e) = engine() else { return };
    // Room for a couple of pages only: tiles keep getting evicted and coming back.
    e.set_tile_budget(16 << 20);
    let g = grid();
    let d = doc(g.clone());
    for round in 0..2 {
        for mip in [0, 1, 2] {
            let k = 1 << mip;
            let rect = IRect::new(-512 / k, -512 / k, 2560 / k, 2048 / k);
            let gpu = eval(&mut e, &d, rect, mip);
            let cpu = premul(&cpu_level(&g, mip).read_rect(&rect));
            let m = max_diff(&gpu, &cpu);
            assert!(m < 1e-5, "round {round} mip {mip}: {m}");
        }
    }
}

#[test]
fn prefetched_views_upload_and_build_nothing() {
    let Some(mut e) = engine() else { return };
    let d = doc(grid());
    let regions = [(0, IRect::new(-1024, -1024, 1536, 1280)), (1, IRect::new(-512, -512, 768, 640))];
    // Sliced: every call does at least one tile, so repeated calls finish.
    let mut calls = 0;
    while !e.prefetch(&d, &regions, &|| true) {
        calls += 1;
        assert!(calls < 1000);
    }
    assert!(calls > 1, "work was spread over several calls");
    // Views inside them (with their texture margin) at the canvas's precision.
    let before = e.tile_stats();
    for (mip, r) in [(0, IRect::new(-256, -256, 512, 512)), (1, IRect::new(-256, -256, 256, 256))] {
        e.evaluate(&EvalRequest { doc: &d, rect: r, mip, format: EvalFormat::F16, source: RenderSource::LayerStack }).unwrap();
    }
    assert_eq!((before.0, before.1), (e.tile_stats().0, e.tile_stats().1));
}

#[test]
fn prefetch_stays_under_its_share_of_the_budget() {
    let Some(mut e) = engine() else { return };
    e.set_tile_budget(64 << 20);
    let d = doc(grid());
    assert!(e.prefetch(&d, &[(0, IRect::new(-4096, -4096, 4096, 4096))], &|| false), "stops (done) when the store is full enough");
    let (_, _, bytes) = e.tile_stats();
    // One page past the threshold at most.
    assert!(bytes <= 64 << 20, "{bytes}");
}

fn render(e: &mut Engine, d: &Document, rect: IRect, mip: u32) -> Vec<f32> {
    let img = e.evaluate(&EvalRequest { doc: d, rect, mip, format: EvalFormat::F16, source: RenderSource::LayerStack }).unwrap();
    Readback::start(&e.ctx.device, &e.ctx.queue, &img, rect).wait(&e.ctx.device)
}

fn fresh(d: &Document, rect: IRect, mip: u32) -> Vec<f32> {
    render(&mut engine().unwrap(), d, rect, mip)
}

/// Infinite doc (fill + one raster layer, ids 1 and 2 like every new document) with `r` painted.
fn painted(r: IRect, c: [f32; 4]) -> Document {
    let mut d = Document::infinite("t", 1000, 800, Some([1.0; 4]));
    let id = d.active_layer.unwrap();
    d.layer_mut(id).unwrap().raster_mut().unwrap().map_rect(&r, |_, _, p| p.copy_from_slice(&c));
    d
}

#[test]
fn documents_never_see_each_others_pixels() {
    let Some(mut e) = engine() else { return };
    let view = IRect::new(-1024, -768, 2048, 1536);
    // A big painted document, shown at two levels...
    let a = painted(IRect::new(-900, -700, 1900, 1400), [0.9, 0.1, 0.1, 1.0]);
    render(&mut e, &a, view, 0);
    render(&mut e, &a, IRect::new(-512, -384, 1024, 768), 1);
    // ...then a new one with the same layer ids and a small stroke: only its own pixels.
    let b = painted(IRect::new(200, 200, 600, 300), [0.1, 0.2, 0.9, 1.0]);
    for (mip, r) in [(0, view), (1, IRect::new(-512, -384, 1024, 768))] {
        assert!(render(&mut e, &b, r, mip) == fresh(&b, r, mip), "mip {mip}: the other document's pixels show");
    }
}

#[test]
fn erased_tiles_never_reappear() {
    let Some(mut e) = engine() else { return };
    let view = IRect::new(0, 0, 1536, 1024);
    let mut d = painted(IRect::new(0, 0, 1500, 1000), [0.9, 0.1, 0.1, 1.0]);
    render(&mut e, &d, view, 0);
    // Erase everything but one corner tile: the content bounds shrink to it.
    let id = d.active_layer.unwrap();
    d.layer_mut(id).unwrap().raster_mut().unwrap().tiles.retain(|c, _| c.x == 0 && c.y == 0);
    assert!(render(&mut e, &d, view, 0) == fresh(&d, view, 0));
}

#[test]
fn releasing_a_document_gives_its_memory_back() {
    let Some(mut e) = engine() else { return };
    let view = IRect::new(-2048, -2048, 4096, 4096);
    let a = painted(IRect::new(-2000, -2000, 4000, 4000), [0.9, 0.1, 0.1, 1.0]);
    let b = painted(IRect::new(0, 0, 600, 300), [0.1, 0.2, 0.9, 1.0]);
    // `b` shares one tile with a clone of `a` (pasted content keeps its tiles).
    let mut b2 = a.clone();
    b2.uid = prisme_core::DocUid::fresh();
    for mip in [0, 1, 2] {
        let k = 1 << mip;
        render(&mut e, &a, IRect::new(view.x0 / k, view.y0 / k, view.x1 / k, view.y1 / k), mip);
    }
    render(&mut e, &b, IRect::new(0, 0, 1024, 512), 0);
    render(&mut e, &b2, IRect::new(0, 0, 256, 256), 0);
    let (_, _, full) = e.tile_stats();
    let pool_full = e.pool_bytes();
    e.release_document(a.uid);
    let (up, _, after) = e.tile_stats();
    assert!(after * 8 < full, "tile store kept {} of {} MiB", after >> 20, full >> 20);
    assert!(e.pool_bytes() < pool_full, "pool kept its large textures");
    // The other documents still render right without re-uploading (b2's tile was shared).
    assert!(render(&mut e, &b, IRect::new(0, 0, 1024, 512), 0) == fresh(&b, IRect::new(0, 0, 1024, 512), 0));
    assert!(render(&mut e, &b2, IRect::new(0, 0, 256, 256), 0) == fresh(&b2, IRect::new(0, 0, 256, 256), 0));
    assert_eq!(e.tile_stats().0, up, "live documents' tiles were freed");
}

#[test]
fn tiles_nothing_shows_any_more_are_trimmed() {
    let Some(mut e) = engine() else { return };
    let view = IRect::new(-2048, -2048, 4096, 4096);
    let mut d = painted(IRect::new(-2000, -2000, 4000, 4000), [0.9, 0.1, 0.1, 1.0]);
    render(&mut e, &d, view, 0);
    let (_, _, full) = e.tile_stats();
    // The big layer goes (deleted); the view only shows a small new one from now on.
    let id = d.active_layer.unwrap();
    *d.layer_mut(id).unwrap().raster_mut().unwrap() = TileGrid::rgba(PixelFormat::F16);
    d.layer_mut(id).unwrap().raster_mut().unwrap().map_rect(&IRect::new(0, 0, 300, 300), |_, _, p| p.copy_from_slice(&[0.1, 0.2, 0.9, 1.0]));
    let small = IRect::new(0, 0, 512, 512);
    render(&mut e, &d, small, 0);
    e.trim_tiles(1);
    let (up, _, after) = e.tile_stats();
    assert!(after * 8 < full, "kept {} of {} MiB", after >> 20, full >> 20);
    assert!(render(&mut e, &d, small, 0) == fresh(&d, small, 0));
    assert_eq!(e.tile_stats().0, up, "tiles in use were trimmed");
}
