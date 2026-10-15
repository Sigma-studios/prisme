//! GPU brush strokes against the CPU reference engine (`StrokeCanvas`, `RetouchCanvas`):
//! dabs are rendered on the GPU, read back and landed with `write_stroke`, then compared
//! with `replay_stroke` on the same document. Uses any available adapter.

use prisme_brush::{BlendTarget, BrushSettings, Dab, InputSample, RetouchMode, Stroke, StrokeMode, TipShape};
use prisme_core::layer::{Layer, LayerKind, LayerMask};
use prisme_core::{Document, IRect, PixelFormat, TileGrid};
use prisme_gpu::{Engine, EvalFormat, EvalRequest, GpuContext, Readback, RenderSource, UploadKey};
use prisme_tools::stroke::{dab_tiles, replay_stroke, write_stroke, PaintTarget, StrokeSpec};
use std::collections::HashSet;
use std::sync::Arc;

fn engine() -> Option<Engine> {
    let ctx = pollster::block_on(GpuContext::headless());
    if ctx.is_none() {
        eprintln!("no GPU adapter; skipping");
    }
    ctx.map(Engine::new)
}

/// 300x200 doc: opaque grey background + a raster layer (active) with a half-transparent blue patch.
fn doc(format: PixelFormat) -> Document {
    let mut d = Document::with_background("t", 300, 200, Some([0.5, 0.5, 0.5, 1.0]));
    let id = d.alloc_layer_id();
    let mut g = TileGrid::rgba(format);
    g.map_rect(&IRect::new(100, 40, 220, 160), |x, _, p| p.copy_from_slice(&[0.1, 0.2, 0.9, if x < 160 { 0.5 } else { 1.0 }]));
    d.layers.push(Layer::new(id, "paint", LayerKind::Raster(Arc::new(g))));
    d.active_layer = Some(id);
    d
}

fn dabs(settings: BrushSettings, pts: &[[f32; 2]]) -> Vec<Dab> {
    let mut s = Stroke::new(settings, 11);
    let mut out = vec![];
    for (i, p) in pts.iter().enumerate() {
        let mut smp = InputSample::mouse(*p, i as f64 * 0.02);
        smp.pressure = 0.4 + 0.6 * ((i % 5) as f32 / 4.0);
        out.extend(s.add(smp));
    }
    out
}

fn spec(d: &Document, target: PaintTarget, mode: StrokeMode, selection: Option<Arc<TileGrid>>) -> StrokeSpec {
    let (base, base_default) = if target.mask {
        let m = d.layer(target.layer).unwrap().mask.as_ref().unwrap();
        (m.grid.clone(), if m.default_visible { 1.0 } else { 0.0 })
    } else {
        (d.layer(target.layer).unwrap().raster().unwrap().clone(), 0.0)
    };
    StrokeSpec { target, base, base_default, selection, mode }
}

fn key(t: PaintTarget) -> UploadKey {
    if t.mask { UploadKey::LayerMask(t.layer.0) } else { UploadKey::Content(t.layer.0) }
}

/// Render on the GPU in uneven batches, read back, land.
fn gpu(e: &mut Engine, d: &Document, sp: &StrokeSpec, dabs: &[Dab]) -> Document {
    let mut s = e.stroke_begin(key(sp.target), sp.base.clone(), sp.base_default, sp.selection.clone(), sp.mode);
    for c in dabs.chunks(7) {
        e.stroke_dabs(&mut s, c);
    }
    let mut out = d.clone();
    let mut tiles = HashSet::new();
    dab_tiles(dabs, &mut tiles);
    for rb in e.stroke_readback(&s).expect("readback") {
        let px = rb.wait(&e.ctx.device);
        let mine = TileGrid::coords_in(&rb.rect).filter(|c| tiles.contains(c));
        assert!(write_stroke(&mut out, sp.target, sp.base_default, rb.rect, &px, sp.base.channels as usize, mine), "layout mismatch");
    }
    out
}

fn cpu(d: &Document, sp: &StrokeSpec, dabs: &[Dab]) -> Document {
    let mut out = d.clone();
    replay_stroke(&mut out, sp, dabs);
    out
}

fn grid(d: &Document, t: PaintTarget) -> TileGrid {
    let l = d.layer(t.layer).unwrap();
    if t.mask { (*l.mask.as_ref().unwrap().grid).clone() } else { (**l.raster().unwrap()).clone() }
}

/// (max abs difference, number of pixels differing by more than `tol`).
fn diff(a: &TileGrid, b: &TileGrid, tol: f32) -> (f32, usize) {
    let r = IRect::new(-64, -64, 364, 264);
    let (pa, pb) = (a.read_rect(&r), b.read_rect(&r));
    let ch = a.channels as usize;
    let mut max = 0.0f32;
    let mut n = 0;
    for (x, y) in pa.chunks(ch).zip(pb.chunks(ch)) {
        let d = x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        max = max.max(d);
        n += (d > tol) as usize;
    }
    (max, n)
}

fn target(d: &Document) -> PaintTarget {
    PaintTarget { layer: d.active_layer.unwrap(), mask: false, saved: None }
}

const LINE: [[f32; 2]; 6] = [[40.0, 60.0], [90.0, 80.0], [150.0, 100.0], [210.0, 90.0], [150.0, 120.0], [60.0, 110.0]];

#[test]
fn opacity_caps_flow_builds_up() {
    let Some(mut e) = engine() else { return };
    let d = doc(PixelFormat::F32);
    let t = target(&d);
    // Opacity caps a stroke that crosses itself many times.
    let hard = BrushSettings { size: 24.0, hardness: 1.0, flow: 1.0, opacity: 0.5, spacing: 0.05, ..Default::default() };
    let sp = spec(&d, t, StrokeMode::Paint { target: BlendTarget::Color([1.0, 0.0, 0.0, 1.0]), opacity: 0.5, lock_alpha: false }, None);
    let ds = dabs(hard, &[[20.0, 20.0], [60.0, 20.0], [20.0, 22.0], [60.0, 22.0]]);
    let g = grid(&gpu(&mut e, &d, &sp, &ds), t);
    let p = g.get_pixel(40, 21);
    assert!((p[3] - 0.5).abs() < 1e-5 && (p[0] - 1.0).abs() < 1e-5, "opacity cap: {p:?}");
    // Flow builds up: n coincident dabs with flow f give 1 - (1-f)^n coverage.
    let sp = spec(&d, t, StrokeMode::Paint { target: BlendTarget::Color([0.0, 1.0, 0.0, 1.0]), opacity: 1.0, lock_alpha: false }, None);
    let dab = Dab { center: [30.5, 150.5], radius: 8.0, angle: 0.0, roundness: 1.0, hardness: 1.0, alpha: 0.3, tip: TipShape::Round, pixel: false, grain: 0.0, seed: 1 };
    for n in [1usize, 3] {
        let g = grid(&gpu(&mut e, &d, &sp, &vec![dab; n]), t);
        let a = g.get_pixel(30, 150)[3];
        let expect = 1.0 - 0.7f32.powi(n as i32);
        assert!((a - expect).abs() < 1e-5, "flow x{n}: {a} vs {expect}");
    }
}

/// Zoomed out on an infinite canvas, a big brush sweeps a huge bounding box: the stroke
/// only holds the tiles it touches, stays on the GPU and lands like the CPU engine.
#[test]
fn long_sweep_stays_on_gpu() {
    let Some(mut e) = engine() else { return };
    let d = doc(PixelFormat::F16);
    let t = target(&d);
    let sp = spec(&d, t, StrokeMode::Paint { target: BlendTarget::Color([1.0, 0.2, 0.0, 1.0]), opacity: 0.8, lock_alpha: false }, None);
    // 31 dabs of radius 300 along a 30000 x 21000 px diagonal.
    let ds: Vec<Dab> = (0..=30)
        .map(|i| {
            let p = i as f32 * 1000.0 - 2000.0;
            Dab { center: [p, p * 0.7], radius: 300.0, angle: 0.0, roundness: 1.0, hardness: 0.5, alpha: 0.7, tip: TipShape::Round, pixel: false, grain: 0.0, seed: 1 }
        })
        .collect();
    let mut s = e.stroke_begin(key(t), sp.base.clone(), sp.base_default, None, sp.mode);
    for c in ds.chunks(5) {
        e.stroke_dabs(&mut s, c);
    }
    assert!(!s.overflow, "overflowed");
    // ~12 tiles per dab, not the 630 Mpx bounding box.
    assert!(s.tile_bytes() < 1 << 30, "{} MiB of tiles", s.tile_bytes() >> 20);
    drop(s);
    let (a, b) = (grid(&gpu(&mut e, &d, &sp, &ds), t), grid(&cpu(&d, &sp, &ds), t));
    let keys: HashSet<_> = a.tiles.keys().chain(b.tiles.keys()).copied().collect();
    assert!(keys.len() > 300, "{} tiles", keys.len());
    for c in keys {
        let r = c.rect();
        let max = a.read_rect(&r).iter().zip(b.read_rect(&r)).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(max < 2e-3, "tile {c:?}: max diff {max}");
    }
}

/// A stroke whose tiles would exceed the VRAM budget overflows to the CPU before allocating.
#[test]
fn over_budget_stroke_overflows() {
    let Some(mut e) = engine() else { return };
    let d = doc(PixelFormat::F16);
    let t = target(&d);
    let sp = spec(&d, t, StrokeMode::Paint { target: BlendTarget::Color([1.0, 0.0, 0.0, 1.0]), opacity: 1.0, lock_alpha: false }, None);
    let mut s = e.stroke_begin(key(t), sp.base.clone(), sp.base_default, None, sp.mode);
    let dab = Dab { center: [0.0, 0.0], radius: 20000.0, angle: 0.0, roundness: 1.0, hardness: 0.5, alpha: 1.0, tip: TipShape::Round, pixel: false, grain: 0.0, seed: 1 };
    e.stroke_dabs(&mut s, &[dab]);
    assert!(s.overflow && s.tile_bytes() == 0);
    assert!(e.stroke_readback(&s).is_none());
}

#[test]
fn paint_matches_cpu() {
    let Some(mut e) = engine() else { return };
    for format in [PixelFormat::F32, PixelFormat::F16] {
        let d = doc(format);
        let t = target(&d);
        let col = BlendTarget::Color([0.9, 0.4, 0.1, 0.8]);
        let soft = BrushSettings { size: 30.0, hardness: 0.2, flow: 0.4, opacity: 0.7, spacing: 0.1, ..Default::default() };
        for (lock_alpha, sel) in [(false, None), (true, None), (false, Some(Arc::new(prisme_core::mask::rect([0.0, 0.0, 130.0, 200.0], false))))] {
            let sp = spec(&d, t, StrokeMode::Paint { target: col, opacity: 0.7, lock_alpha }, sel);
            let ds = dabs(soft.clone(), &LINE);
            let (max, _) = diff(&grid(&gpu(&mut e, &d, &sp, &ds), t), &grid(&cpu(&d, &sp, &ds), t), 0.0);
            let tol = if format == PixelFormat::F16 { 2e-3 } else { 2e-4 };
            assert!(max < tol, "{format:?} lock={lock_alpha}: max diff {max}");
        }
    }
}

#[test]
fn erase_matches_cpu() {
    let Some(mut e) = engine() else { return };
    let d = doc(PixelFormat::F32);
    let t = target(&d);
    let sel = Arc::new(prisme_core::mask::rect([0.0, 0.0, 180.0, 200.0], false));
    let sp = spec(&d, t, StrokeMode::Paint { target: BlendTarget::Erase, opacity: 0.8, lock_alpha: false }, Some(sel));
    let ds = dabs(BrushSettings { size: 40.0, hardness: 0.6, flow: 0.5, ..Default::default() }, &LINE);
    let g = gpu(&mut e, &d, &sp, &ds);
    let (max, _) = diff(&grid(&g, t), &grid(&cpu(&d, &sp, &ds), t), 0.0);
    assert!(max < 2e-4, "max diff {max}");
    // Erased inside the selection, untouched outside it.
    let gg = grid(&g, t);
    assert!(gg.get_pixel(150, 100)[3] < 0.5, "{:?}", gg.get_pixel(150, 100));
    assert_eq!(gg.get_pixel(200, 90)[3], 1.0);
}

#[test]
fn mask_value_matches_cpu() {
    let Some(mut e) = engine() else { return };
    let mut d = doc(PixelFormat::F32);
    let id = d.active_layer.unwrap();
    // Reveal-all mask: missing tiles read as 1.0.
    d.layer_mut(id).unwrap().mask = Some(LayerMask { grid: Arc::new(TileGrid::mask()), enabled: true, default_visible: true });
    let t = PaintTarget { layer: id, mask: true, saved: None };
    let sp = spec(&d, t, StrokeMode::Paint { target: BlendTarget::Value(0.0), opacity: 1.0, lock_alpha: false }, None);
    let ds = dabs(BrushSettings { size: 20.0, hardness: 0.9, ..Default::default() }, &LINE);
    let g = gpu(&mut e, &d, &sp, &ds);
    let (max, _) = diff(&grid(&g, t), &grid(&cpu(&d, &sp, &ds), t), 0.0);
    assert!(max < 2e-3, "max diff {max}");
    assert!(grid(&g, t).get_pixel(250, 10)[0] == 1.0);
}

#[test]
fn tips_match_cpu() {
    let Some(mut e) = engine() else { return };
    let d = doc(PixelFormat::F32);
    let t = target(&d);
    for p in prisme_brush::presets() {
        let mut st = p.settings.clone();
        st.airbrush = false;
        let sp = spec(&d, t, StrokeMode::Paint { target: BlendTarget::Color([0.2, 0.8, 0.3, 1.0]), opacity: st.opacity, lock_alpha: false }, None);
        let ds = dabs(st, &LINE);
        let (max, n) = diff(&grid(&gpu(&mut e, &d, &sp, &ds), t), &grid(&cpu(&d, &sp, &ds), t), 2e-3);
        // Textured tips threshold noise: allow a handful of edge pixels to flip.
        assert!(n <= 40, "{}: {n} pixels differ (max {max})", p.name);
    }
}

#[test]
fn retouch_matches_cpu() {
    let Some(mut e) = engine() else { return };
    let d = doc(PixelFormat::F32);
    let t = target(&d);
    let brush = BrushSettings { size: 26.0, hardness: 0.5, spacing: 0.15, ..Default::default() };
    let sel = Some(Arc::new(prisme_core::mask::rect([0.0, 0.0, 190.0, 200.0], false)));
    for (mode, sel) in [
        (RetouchMode::Clone { offset: [60.4, -20.6] }, None),
        (RetouchMode::Clone { offset: [60.4, -20.6] }, sel.clone()),
        (RetouchMode::Heal { offset: [-35.0, 12.0] }, None),
        (RetouchMode::Smudge { strength: 0.6 }, None),
        (RetouchMode::Smudge { strength: 0.6 }, sel.clone()),
        (RetouchMode::Blur { strength: 0.8 }, None),
    ] {
        let sp = spec(&d, t, StrokeMode::Retouch(mode), sel.clone());
        let ds = dabs(brush.clone(), &LINE);
        let (max, n) = diff(&grid(&gpu(&mut e, &d, &sp, &ds), t), &grid(&cpu(&d, &sp, &ds), t), 2e-3);
        assert!(n == 0, "{mode:?} sel={}: {n} pixels differ (max {max})", sel.is_some());
    }
}

#[test]
fn preview_splices_stroke() {
    let Some(mut e) = engine() else { return };
    let d = doc(PixelFormat::F32);
    let t = target(&d);
    let sp = spec(&d, t, StrokeMode::Paint { target: BlendTarget::Color([1.0, 0.1, 0.1, 1.0]), opacity: 0.9, lock_alpha: false }, None);
    // Crosses the tile edge at x = 256 (the view splices tile by tile).
    let ds = dabs(BrushSettings { size: 30.0, ..Default::default() }, &[LINE.as_slice(), &[[200.0, 40.0], [290.0, 120.0], [240.0, 180.0]]].concat());
    let landed = cpu(&d, &sp, &ds);
    let mut s = e.stroke_begin(key(t), sp.base.clone(), 0.0, None, sp.mode);
    e.stroke_dabs(&mut s, &ds);
    for mip in [0u32, 1, 2, 4] {
        let k = 1 << mip;
        let rect = IRect::new(0, 0, 300 / k, 200 / k);
        let read = |e: &mut Engine, doc: &Document, s: Option<&prisme_gpu::GpuStroke>| {
            let img = e.evaluate_with(&EvalRequest { doc, rect, mip, format: EvalFormat::F32, source: RenderSource::LayerStack }, s).unwrap();
            Readback::start(&e.ctx.device, &e.ctx.queue, &img, rect).wait(&e.ctx.device)
        };
        let a = read(&mut e, &d, Some(&s));
        let b = read(&mut e, &landed, None);
        let max = a.iter().zip(&b).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max < 1e-3, "mip {mip}: max diff {max}");
    }
    // Without the stroke the cached result must not leak the preview.
    let img = e.evaluate(&EvalRequest { doc: &d, rect: IRect::new(0, 0, 300, 200), mip: 0, format: EvalFormat::F32, source: RenderSource::LayerStack }).unwrap();
    let px = Readback::start(&e.ctx.device, &e.ctx.queue, &img, IRect::new(40, 60, 41, 61)).wait(&e.ctx.device);
    assert!((px[0] - 0.5).abs() < 1e-4, "{px:?}");
}

/// Land through `stroke_pack` (f16 bits converted on the GPU) instead of the f32 read-back.
fn gpu_f16(e: &mut Engine, d: &Document, sp: &StrokeSpec, dabs: &[Dab]) -> Document {
    let mut s = e.stroke_begin(key(sp.target), sp.base.clone(), sp.base_default, sp.selection.clone(), sp.mode);
    for c in dabs.chunks(7) {
        e.stroke_dabs(&mut s, c);
    }
    let mut out = d.clone();
    let mut mine = HashSet::new();
    dab_tiles(dabs, &mut mine);
    let p = e.stroke_pack(&s).expect("f16 target");
    p.wait(&e.ctx.device);
    let tiles = p.take(&e.ctx.device).expect("tiles");
    let g = prisme_tools::stroke::target_grid(&mut out, sp.target).unwrap();
    for (c, t) in p.coords.iter().zip(tiles) {
        if mine.contains(c) {
            if sp.base_default == 0.0 && t.is_clear() {
                g.tiles.remove(c);
            } else {
                g.insert(*c, t);
            }
        }
    }
    out
}

fn tile_bits(g: &TileGrid) -> Vec<(i32, i32, Vec<u8>)> {
    let mut v: Vec<_> = g.tiles.iter().map(|(c, t)| (c.x, c.y, t.to_bytes())).collect();
    v.sort();
    v
}

#[test]
fn f16_landing_matches_f32_readback_bit_for_bit() {
    let Some(mut e) = engine() else { return };
    let mut d = doc(PixelFormat::F16);
    let id = d.active_layer.unwrap();
    let mut m = TileGrid::mask();
    m.map_rect(&IRect::new(0, 0, 150, 200), |x, _, p| p[0] = x as f32 / 150.0);
    d.layer_mut(id).unwrap().mask = Some(LayerMask { grid: Arc::new(m), enabled: true, default_visible: true });
    let pts: Vec<[f32; 2]> = (0..40).map(|i| [20.0 + i as f32 * 7.0, 30.0 + (i as f32 * 0.4).sin() * 60.0 + i as f32 * 2.0]).collect();
    let soft = BrushSettings { size: 70.0, hardness: 0.3, ..Default::default() };
    let content = PaintTarget { layer: id, mask: false, saved: None };
    let mask = PaintTarget { layer: id, mask: true, saved: None };
    let paint = StrokeMode::Paint { target: BlendTarget::Color([0.8, 0.31, 0.07, 0.9]), opacity: 0.85, lock_alpha: false };
    let cases = [
        ("paint", content, paint),
        ("erase", content, StrokeMode::Paint { target: BlendTarget::Erase, opacity: 0.6, lock_alpha: false }),
        ("smudge", content, StrokeMode::Retouch(RetouchMode::Smudge { strength: 0.7 })),
        ("mask", mask, StrokeMode::Paint { target: BlendTarget::Value(0.0), opacity: 0.7, lock_alpha: false }),
    ];
    for (name, t, mode) in cases {
        let sp = spec(&d, t, mode, None);
        let ds = dabs(soft.clone(), &pts);
        let a = gpu(&mut e, &d, &sp, &ds);
        let b = gpu_f16(&mut e, &d, &sp, &ds);
        let (ga, gb) = (grid(&a, t), grid(&b, t));
        assert_eq!(tile_bits(&ga), tile_bits(&gb), "{name}: f16 landing differs from the f32 read-back");
        assert!(tile_bits(&ga) != tile_bits(&grid(&d, t)), "{name}: stroke changed something");
    }
}

#[test]
fn f16_packing_rounds_like_the_half_crate() {
    let Some(mut e) = engine() else { return };
    // Edge cases first: zeros, subnormal boundaries, ties around every rounding position,
    // max/overflow, infinities, NaNs; then random bit patterns.
    let mut vals: Vec<f32> = vec![0.0, -0.0, 1.0, -1.0, 65504.0, 65519.0, 65520.0, 1e9, -1e9, f32::INFINITY, f32::NEG_INFINITY, f32::MIN_POSITIVE, 5.96e-8, 2.98e-8, 2.99e-8, 6.1e-5, 6.09e-5];
    for h in [0x0001u16, 0x03FF, 0x0400, 0x3C00, 0x3C01, 0x7BFF, 0x0002, 0x0003] {
        let f = half::f16::from_bits(h).to_f32();
        let next = half::f16::from_bits(h + 1).to_f32();
        let mid = (f + next) / 2.0;
        vals.extend([f, mid, f32::from_bits(mid.to_bits() + 1), f32::from_bits(mid.to_bits() - 1), -mid]);
    }
    let mut x: u32 = 0x1234_5678;
    while vals.len() < 256 * 256 * 4 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        // Mostly the f16 range, some anywhere.
        let v = if x.is_multiple_of(8) { f32::from_bits(x) } else { f32::from_bits((x & 0x87FF_FFFF) | 0x3000_0000) };
        vals.push(v);
    }
    let tex = |format: wgpu::TextureFormat, data: &[u8], bpp: u32| {
        let t = e.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        e.ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &t, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256 * bpp), rows_per_image: Some(256) },
            wgpu::Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
        );
        let view = t.create_view(&Default::default());
        prisme_gpu::GpuImage { tex: Arc::new(prisme_gpu::image::PooledTex { tex: t, view, w: 256, h: 256, format }), rect: IRect::new(0, 0, 256, 256), straight: true }
    };
    let rgba = tex(wgpu::TextureFormat::Rgba32Float, bytemuck::cast_slice(&vals), 16);
    let r = tex(wgpu::TextureFormat::R32Float, bytemuck::cast_slice(&vals[..256 * 256]), 4);
    let c = prisme_core::tile::TileCoord::new(0, 0);
    for (img, ch, src) in [(rgba, 4u8, &vals[..]), (r, 1, &vals[..256 * 256])] {
        let p = e.pack_tiles(&[(c, img)], ch);
        p.wait(&e.ctx.device);
        let bytes = p.take(&e.ctx.device).unwrap().remove(0).to_bytes();
        let got: Vec<u16> = bytemuck::pod_collect_to_vec(&bytes);
        let mut bad = 0;
        for (i, (&v, &g)) in src.iter().zip(&got).enumerate() {
            let want = half::f16::from_f32(v);
            let ok = if want.is_nan() { half::f16::from_bits(g).is_nan() } else { g == want.to_bits() };
            if !ok {
                if bad < 5 {
                    eprintln!("ch {ch} #{i}: {v:e} ({:#010x}) -> gpu {g:#06x}, half {:#06x}", v.to_bits(), want.to_bits());
                }
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "{ch} channels: {bad} values rounded differently");
    }
}
