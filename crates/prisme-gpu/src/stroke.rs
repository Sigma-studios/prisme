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
    mode: StrokeMode,
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
        if matches!(self.mode, StrokeMode::Paint { .. }) {
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
            mode,
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
        if dabs.is_empty() || s.overflow {
            return;
        }
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("stroke") });
        match s.mode {
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
