//! GPU brush strokes.
//!
//! Dabs are placed on the CPU (`prisme_brush::Stroke`: spacing, dynamics, jitter) and
//! rasterized here. A stroke keeps level-0 textures over a tile-aligned rect that grows with
//! it: the target's snapshot (`base`, f32), the coverage buffer (paint: opacity caps the
//! stroke, flow builds up) and the `live` result. Evaluation splices `live` into the target's
//! upload ([`Engine::evaluate_with`]), so painting never rewrites CPU tiles per frame. When
//! the stroke ends the host reads `live` back asynchronously ([`Engine::stroke_readback`])
//! and writes it into the tiles as one history edit: the tile model stays authoritative.
//!
//! Paint/erase/mask-value strokes accumulate a batch of dabs per dispatch (same order and
//! maths as `StrokeCanvas`). Smudge, blur, clone and heal read pixels and depend on the
//! previous dab, so they run one dab per dispatch, sampling `live` (and `base` for the
//! clone source) instead of CPU tiles; the weighted means smudge/heal need are reduced on
//! the GPU too.
//!
//! A Liquify session is a stroke too ([`Engine::liquify_begin`]): `base` covers the whole
//! working area, the displacement field (painted on the CPU, `prisme_tools::liquify`) lives
//! in a node texture updated in place, and there is no `live` texture: evaluation warps
//! `base` through the field directly at the requested level (`k_liquify`), and the bake
//! renders it bicubic at full resolution for the readback.

use crate::engine::Engine;
use crate::image::{GpuImage, PooledTex};
use crate::kernels::Globals;
use crate::readback::Readback;
use crate::upload::UploadKey;
use prisme_brush::{BlendTarget, Dab, RetouchMode, StrokeMode, TipShape};
use prisme_core::tile::{TileGrid, TILE_PIXELS, TILE_SIZE};
use prisme_core::IRect;
use std::sync::Arc;

const TS: i32 = TILE_SIZE as i32;
/// Dabs per accumulation dispatch (keeps each dispatch rect tight on long, fast strokes).
const BATCH: usize = 64;
const G: Globals = Globals { level_scale: 1.0, mip: 0, doc_w: 0.0, doc_h: 0.0 };

fn align(r: IRect) -> IRect {
    IRect::new(r.x0.div_euclid(TS) * TS, r.y0.div_euclid(TS) * TS, (r.x1 + TS - 1).div_euclid(TS) * TS, (r.y1 + TS - 1).div_euclid(TS) * TS)
}

fn contains(outer: &IRect, inner: &IRect) -> bool {
    inner.is_empty() || outer.intersect(inner) == *inner
}

fn texture(device: &wgpu::Device, rect: IRect, format: wgpu::TextureFormat) -> GpuImage {
    let (w, h) = (rect.width().max(1) as u32, rect.height().max(1) as u32);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("stroke"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    GpuImage { tex: Arc::new(PooledTex { tex, view, w, h, format }), rect, straight: true }
}

/// Copy `r` (document px, inside both images) from `src` to `dst`.
fn copy(enc: &mut wgpu::CommandEncoder, src: &GpuImage, dst: &GpuImage, r: IRect) {
    let r = r.intersect(&src.rect).intersect(&dst.rect);
    if r.is_empty() {
        return;
    }
    fn at(img: &GpuImage, r: IRect) -> wgpu::TexelCopyTextureInfo<'_> {
        wgpu::TexelCopyTextureInfo {
            texture: &img.tex.tex,
            mip_level: 0,
            origin: wgpu::Origin3d { x: (r.x0 - img.rect.x0) as u32, y: (r.y0 - img.rect.y0) as u32, z: 0 },
            aspect: wgpu::TextureAspect::All,
        }
    }
    enc.copy_texture_to_texture(at(src, r), at(dst, r), wgpu::Extent3d { width: r.width() as u32, height: r.height() as u32, depth_or_array_layers: 1 });
}

fn format_for(channels: u8) -> wgpu::TextureFormat {
    if channels == 1 { wgpu::TextureFormat::R32Float } else { wgpu::TextureFormat::Rgba32Float }
}

/// A grid's pixels at f32 over a growing tile-aligned rect (missing tiles = `default`).
struct Snapshot {
    grid: Arc<TileGrid>,
    default: f32,
    img: Option<GpuImage>,
}

impl Snapshot {
    fn new(grid: Arc<TileGrid>, default: f32) -> Self {
        Self { grid, default, img: None }
    }

    fn rect(&self) -> IRect {
        self.img.as_ref().map(|i| i.rect).unwrap_or(IRect::EMPTY)
    }

    /// Grow to cover `want` (tile aligned, containing the current rect): copy the old texture,
    /// upload only the new tiles.
    fn grow(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, want: IRect) {
        let old = self.rect();
        if !old.is_empty() && old == want {
            return;
        }
        let ch = self.grid.channels as usize;
        let img = texture(device, want, format_for(self.grid.channels));
        if let Some(o) = &self.img {
            copy(enc, o, &img, old);
        }
        let mut fill: Option<Vec<f32>> = None;
        for c in TileGrid::coords_in(&want) {
            if contains(&old, &c.rect()) {
                continue;
            }
            let view;
            let data: &[f32] = match self.grid.get_tile(c) {
                Some(t) => {
                    view = t.f32_view();
                    &view
                }
                // New textures are zero-initialized.
                None if self.default == 0.0 => continue,
                None => fill.get_or_insert_with(|| vec![self.default; TILE_PIXELS * ch]),
            };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &img.tex.tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: (c.x * TS - want.x0) as u32, y: (c.y * TS - want.y0) as u32, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(data),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(TILE_SIZE * 4 * ch as u32), rows_per_image: Some(TILE_SIZE) },
                wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: 1 },
            );
        }
        self.img = Some(img);
    }
}

/// GPU state of one brush stroke (see the module docs).
pub struct GpuStroke {
    /// The upload this stroke replaces during evaluation (layer pixels, layer mask, saved mask).
    pub key: UploadKey,
    /// None for a Liquify session.
    mode: Option<StrokeMode>,
    liquify: Option<Liquify>,
    channels: u8,
    base: Snapshot,
    sel: Option<Snapshot>,
    /// Clone/heal source: the base grid around the offset dabs (own rect).
    src: Option<Snapshot>,
    live: Option<GpuImage>,
    cov: Option<GpuImage>,
    /// Smudge: previous `k_retouch_mean` output (carried colour).
    carried: Option<GpuImage>,
    /// Union of dab bounds (document px).
    pub dirty: IRect,
    /// Bumped on every change (evaluation cache key).
    pub generation: u64,
    /// The stroke outgrew the texture size limit: the preview stops growing and the host must
    /// land the stroke with the CPU engine instead of a readback.
    pub overflow: bool,
}

impl GpuStroke {
    /// Level-0 rect covered by the stroke textures.
    pub fn rect(&self) -> IRect {
        self.base.rect()
    }

    pub fn channels(&self) -> u8 {
        self.channels
    }

    pub(crate) fn live(&self) -> Option<&GpuImage> {
        self.live.as_ref()
    }

    fn format(&self) -> wgpu::TextureFormat {
        format_for(self.channels)
    }

    /// Make the destination textures cover `need`. False when that would exceed the limits.
    fn ensure(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, need: IRect) -> bool {
        if self.overflow {
            return false;
        }
        let old = self.rect();
        if !old.is_empty() && contains(&old, &need) {
            return true;
        }
        let want = align(old.union(&need));
        let max = device.limits().max_texture_dimension_2d as i32;
        if want.width() > max || want.height() > max {
            log::warn!("stroke rect {want:?} exceeds texture limits; finishing it on the CPU");
            self.overflow = true;
            return false;
        }
        self.base.grow(device, queue, enc, want);
        if let Some(s) = &mut self.sel {
            s.grow(device, queue, enc, want);
        }
        let base = self.base.img.clone().unwrap();
        // New live area starts as the base; keep what's already painted.
        let live = texture(device, want, self.format());
        copy(enc, &base, &live, want);
        if let Some(l) = &self.live {
            copy(enc, l, &live, old);
        }
        self.live = Some(live);
        if matches!(self.mode, Some(StrokeMode::Paint { .. })) {
            let cov = texture(device, want, wgpu::TextureFormat::R32Float);
            if let Some(c) = &self.cov {
                copy(enc, c, &cov, old);
            }
            self.cov = Some(cov);
        }
        true
    }

    fn ensure_src(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, need: IRect) -> bool {
        let Some(s) = &mut self.src else { return true };
        let old = s.rect();
        if !old.is_empty() && contains(&old, &need) {
            return true;
        }
        let want = align(old.union(&need));
        let max = device.limits().max_texture_dimension_2d as i32;
        if want.width() > max || want.height() > max {
            self.overflow = true;
            return false;
        }
        s.grow(device, queue, enc, want);
        true
    }
}

fn dab_vecs(d: &Dab) -> [[f32; 4]; 4] {
    let b = d.bounds();
    let tip = match d.tip {
        TipShape::Round => 0.0,
        TipShape::Square => 1.0,
        TipShape::Chalk => 2.0,
        TipShape::Charcoal => 3.0,
        TipShape::Splatter => 4.0,
    };
    [
        [d.center[0], d.center[1], d.radius, d.angle],
        [d.roundness, d.hardness, d.alpha, tip],
        [d.pixel as i32 as f32, d.grain, d.seed as f32, 0.0],
        [b[0] as f32, b[1] as f32, b[2] as f32, b[3] as f32],
    ]
}

fn dab_rect(d: &Dab) -> IRect {
    let b = d.bounds();
    IRect::new(b[0], b[1], b[2], b[3])
}

impl Engine {
    /// Start a stroke on `base` (the target's pixels at stroke start; missing tiles read as
    /// `base_default`). `key` names the upload the stroke replaces while it's displayed.
    pub fn stroke_begin(&mut self, key: UploadKey, base: Arc<TileGrid>, base_default: f32, selection: Option<Arc<TileGrid>>, mode: StrokeMode) -> GpuStroke {
        let channels = base.channels;
        let src = matches!(mode, StrokeMode::Retouch(RetouchMode::Clone { .. } | RetouchMode::Heal { .. })).then(|| Snapshot::new(base.clone(), base_default));
        GpuStroke {
            key,
            mode: Some(mode),
            liquify: None,
            channels,
            base: Snapshot::new(base, base_default),
            sel: selection.map(|s| Snapshot::new(s, 0.0)),
            src,
            live: None,
            cov: None,
            carried: None,
            dirty: IRect::EMPTY,
            generation: 0,
            overflow: false,
        }
    }

    /// Rasterize dabs into the stroke (one submission).
    pub fn stroke_dabs(&mut self, s: &mut GpuStroke, dabs: &[Dab]) {
        for d in dabs {
            s.dirty = s.dirty.union(&dab_rect(d));
        }
        let Some(mode) = s.mode.filter(|_| !dabs.is_empty() && !s.overflow) else { return };
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("stroke") });
        match mode {
            StrokeMode::Paint { target, opacity, lock_alpha } => {
                for chunk in dabs.chunks(BATCH) {
                    self.paint_batch(&mut enc, s, chunk, target, opacity, lock_alpha);
                }
            }
            StrokeMode::Retouch(m) => {
                for d in dabs {
                    self.retouch_dab(&mut enc, s, m, d);
                }
            }
        }
        self.ctx.queue.submit([enc.finish()]);
        s.generation += 1;
    }

    fn paint_batch(&mut self, enc: &mut wgpu::CommandEncoder, s: &mut GpuStroke, dabs: &[Dab], target: BlendTarget, opacity: f32, lock_alpha: bool) {
        let r = dabs.iter().fold(IRect::EMPTY, |a, d| a.union(&dab_rect(d)));
        if !s.ensure(&self.ctx.device, &self.ctx.queue, enc, r) {
            return;
        }
        let (cov, live) = (s.cov.clone().unwrap(), s.live.clone().unwrap());
        let base = s.base.img.clone().unwrap();
        let sel = s.sel.as_ref().and_then(|s| s.img.clone());
        let data: Vec<[f32; 4]> = dabs.iter().flat_map(dab_vecs).collect();
        let lut = self.kernels.lut_buffer(&data);
        let tmp = self.pool.image(r, wgpu::TextureFormat::R32Float);
        self.kernels.dispatch(enc, "k_stroke_dabs", &tmp, None, [Some(&cov), None, None, None], &[dabs.len() as f32], Some(&lut), &G);
        copy(enc, &tmp, &cov, r);
        let (mode, col, value) = match target {
            BlendTarget::Color(c) => (0.0, c, 0.0),
            BlendTarget::Erase => (1.0, [0.0; 4], 0.0),
            BlendTarget::Value(v) => (2.0, [0.0; 4], v),
        };
        let p = [col[0], col[1], col[2], col[3], mode, opacity, lock_alpha as i32 as f32, s.channels as f32, value];
        let out = self.pool.image(r, s.format());
        self.kernels.dispatch(enc, "k_stroke_paint", &out, None, [Some(&base), Some(&cov), sel.as_ref(), None], &p, None, &G);
        copy(enc, &out, &live, r);
    }

    fn retouch_dab(&mut self, enc: &mut wgpu::CommandEncoder, s: &mut GpuStroke, m: RetouchMode, d: &Dab) {
        let r = dab_rect(d);
        let rad = (d.radius * 0.15).clamp(1.0, 8.0) as i32;
        let (id, strength, off) = match m {
            RetouchMode::Smudge { strength } => (0.0, strength, [0, 0]),
            RetouchMode::Blur { strength } => (1.0, strength, [0, 0]),
            RetouchMode::Clone { offset } => (2.0, 0.0, [offset[0].round() as i32, offset[1].round() as i32]),
            RetouchMode::Heal { offset } => (3.0, 0.0, [offset[0].round() as i32, offset[1].round() as i32]),
        };
        let need = if id == 1.0 { r.expand(rad) } else { r };
        let (dev, q) = (self.ctx.device.clone(), self.ctx.queue.clone());
        if !s.ensure(&dev, &q, enc, need) || !s.ensure_src(&dev, &q, enc, r.translate(off[0], off[1])) {
            return;
        }
        let live = s.live.clone().unwrap();
        let src = s.src.as_ref().and_then(|s| s.img.clone());
        let sel = s.sel.as_ref().and_then(|s| s.img.clone());
        let mut p: Vec<f32> = dab_vecs(d).concat();
        p.extend([id, strength, off[0] as f32, off[1] as f32, rad as f32, s.carried.is_none() as i32 as f32, 0.0, 0.0]);
        let mean = if matches!(m, RetouchMode::Smudge { .. } | RetouchMode::Heal { .. }) {
            let out = self.pool.image(IRect::new(0, 0, 2, 1), wgpu::TextureFormat::Rgba32Float);
            self.kernels.dispatch(enc, "k_retouch_mean", &out, None, [Some(&live), src.as_ref(), sel.as_ref(), s.carried.as_ref()], &p, None, &G);
            Some(out)
        } else {
            None
        };
        let out = self.pool.image(r, s.format());
        self.kernels.dispatch(enc, "k_retouch_apply", &out, None, [Some(&live), src.as_ref(), sel.as_ref(), mean.as_ref()], &p, None, &G);
        copy(enc, &out, &live, r);
        if matches!(m, RetouchMode::Smudge { .. }) {
            s.carried = mean;
        }
    }

    /// Start reading the finished stroke back (async; poll with [`Readback::try_take`]).
    /// None when there's nothing to read or the stroke overflowed (land it on the CPU).
    pub fn stroke_readback(&self, s: &GpuStroke) -> Option<Readback> {
        let live = s.live.as_ref()?;
        let r = s.dirty.intersect(&live.rect);
        if s.overflow || r.is_empty() {
            return None;
        }
        Some(Readback::start(&self.ctx.device, &self.ctx.queue, live, r))
    }
}

/// Liquify session state (see the module docs).
struct Liquify {
    /// Field nodes `(dx, dy, freeze, 0)`; rect = node indices.
    field: GpuImage,
    /// Working area (document px).
    rect: IRect,
    step: i32,
    show_mask: bool,
}

impl GpuStroke {
    pub fn is_liquify(&self) -> bool {
        self.liquify.is_some()
    }

    /// `k_liquify` inputs (base, field, selection) and params for level scale `k`.
    pub(crate) fn liquify_args(&self, k: i32, bicubic: bool, default: f32) -> Option<([GpuImage; 2], Option<GpuImage>, Vec<f32>)> {
        let l = self.liquify.as_ref().filter(|_| !self.overflow)?;
        let base = self.base.img.clone()?;
        let sel = self.sel.as_ref().and_then(|s| s.img.clone());
        let r = l.rect;
        let show = l.show_mask && !bicubic;
        let p = vec![k as f32, l.step as f32, show as i32 as f32, bicubic as i32 as f32, r.x0 as f32, r.y0 as f32, r.x1 as f32, r.y1 as f32, default];
        Some(([base, l.field.clone()], sel, p))
    }

    /// Working area of a Liquify session.
    pub fn liquify_rect(&self) -> IRect {
        self.liquify.as_ref().map(|l| l.rect).unwrap_or(IRect::EMPTY)
    }
}

impl Engine {
    /// Start a Liquify session on `base` (layer pixels, missing tiles transparent) over the
    /// working area `rect`, with a `nodes` field of `step` px per node.
    pub fn liquify_begin(&mut self, key: UploadKey, base: Arc<TileGrid>, selection: Option<Arc<TileGrid>>, rect: IRect, step: i32, nodes: [i32; 2]) -> GpuStroke {
        let (device, queue) = (self.ctx.device.clone(), self.ctx.queue.clone());
        let channels = base.channels;
        let field = texture(&device, IRect::new(0, 0, nodes[0], nodes[1]), wgpu::TextureFormat::Rgba32Float);
        let mut s = GpuStroke {
            key,
            mode: None,
            liquify: Some(Liquify { field, rect, step, show_mask: false }),
            channels,
            base: Snapshot::new(base, 0.0),
            sel: selection.map(|s| Snapshot::new(s, 0.0)),
            src: None,
            live: None,
            cov: None,
            carried: None,
            dirty: IRect::EMPTY,
            generation: 0,
            overflow: false,
        };
        let want = align(rect);
        let max = device.limits().max_texture_dimension_2d as i32;
        if want.width() > max || want.height() > max || nodes[0] > max || nodes[1] > max {
            log::warn!("liquify area {want:?} exceeds texture limits; no GPU preview");
            s.overflow = true;
            return s;
        }
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("liquify") });
        s.base.grow(&device, &queue, &mut enc, want);
        if let Some(sel) = &mut s.sel {
            sel.grow(&device, &queue, &mut enc, want);
        }
        queue.submit([enc.finish()]);
        s
    }

    /// Upload changed field nodes (`nodes` rect, row-major values).
    pub fn liquify_update(&mut self, s: &mut GpuStroke, nodes: IRect, data: &[[f32; 4]]) {
        let Some(l) = &s.liquify else { return };
        if nodes.is_empty() || nodes.intersect(&l.field.rect) != nodes || data.len() != nodes.area() as usize {
            return;
        }
        self.ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &l.field.tex.tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: nodes.x0 as u32, y: nodes.y0 as u32, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(nodes.width() as u32 * 16), rows_per_image: Some(nodes.height() as u32) },
            wgpu::Extent3d { width: nodes.width() as u32, height: nodes.height() as u32, depth_or_array_layers: 1 },
        );
        s.generation += 1;
    }

    /// Tint frozen areas in the preview. True if it changed.
    pub fn liquify_show_mask(&mut self, s: &mut GpuStroke, on: bool) -> bool {
        match &mut s.liquify {
            Some(l) if l.show_mask != on => {
                l.show_mask = on;
                s.generation += 1;
                true
            }
            _ => false,
        }
    }

    /// Render the warped layer over `rect` (document px; bicubic for the bake, bilinear
    /// like the preview otherwise) and start reading it back. None on overflow (bake on the
    /// CPU instead).
    pub fn liquify_readback(&mut self, s: &GpuStroke, rect: IRect, bicubic: bool) -> Option<Readback> {
        let rect = rect.intersect(&s.liquify_rect());
        let ([base, field], sel, p) = s.liquify_args(1, bicubic, 0.0)?;
        if rect.is_empty() {
            return None;
        }
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("liquify-bake") });
        let out = self.pool.image(rect, wgpu::TextureFormat::Rgba32Float);
        self.kernels.dispatch(&mut enc, "k_liquify", &out, None, [None, Some(&base), Some(&field), sel.as_ref()], &p, None, &G);
        self.ctx.queue.submit([enc.finish()]);
        Some(Readback::start(&self.ctx.device, &self.ctx.queue, &out, rect))
    }
}
