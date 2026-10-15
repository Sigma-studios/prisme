//! GPU brush strokes.
//!
//! Dabs are placed on the CPU (`prisme_brush::Stroke`: spacing, dynamics, jitter) and
//! rasterized here. A stroke keeps level-0 textures per document tile, created as dabs reach
//! them: the target's snapshot (`base`, f32), the coverage buffer (paint: opacity caps the
//! stroke, flow builds up), the selection and the `live` result. VRAM follows the area
//! painted, not the stroke's bounding box, so a long sweep with a big brush zoomed out on an
//! infinite canvas stays cheap. Evaluation splices `live` into the target's upload tile by
//! tile ([`Engine::evaluate_with`]), so painting never rewrites CPU tiles per frame. When the
//! stroke ends the host reads the `live` tiles back asynchronously
//! ([`Engine::stroke_readback`]) and writes them into the document as one history edit: the
//! tile model stays authoritative.
//!
//! The kernels work on dense images: each dispatch gathers the tiles it reads into pool
//! textures over its rect and scatters the result back. Paint/erase/mask-value strokes
//! accumulate a batch of dabs per dispatch (same order and maths as `StrokeCanvas`), split
//! into blocks so a huge batch never needs a huge temporary. Smudge, blur, clone and heal
//! read pixels and depend on the previous dab, so they run one dab per dispatch, sampling
//! `live` (and `base` for the clone source) instead of CPU tiles; the weighted means
//! smudge/heal need are reduced on the GPU too.
//!
//! The tiles stay under a VRAM budget; a stroke that would exceed it stops growing on the GPU
//! (`overflow`) and the host lands it with the CPU engine instead.
//!
//! A Liquify session is a stroke too ([`Engine::liquify_begin`]): a dense `base` covers the
//! whole working area, the displacement field (painted on the CPU, `prisme_tools::liquify`)
//! lives in a node texture updated in place, and there is no `live` texture: evaluation warps
//! `base` through the field directly at the requested level (`k_liquify`), and the bake
//! renders it bicubic at full resolution for the readback.
//!
//! A Free Transform preview is one too ([`Engine::xform_begin`]): a dense snapshot of the
//! pixels being transformed (and of the layer mask when it moves with them), drawn through
//! the current homography over the layer's regular upload (`k_xform`). Moving a handle only
//! changes the map, so nothing is resampled on the CPU or re-uploaded per frame. It has no
//! readback: the tool resamples the layer on the CPU once when the transform is applied.

use crate::engine::Engine;
use crate::image::{GpuImage, PooledTex, TexturePool};
use crate::kernels::Globals;
use crate::readback::Readback;
use crate::upload::UploadKey;
use prisme_brush::{BlendTarget, Dab, RetouchMode, StrokeMode, TipShape};
use prisme_core::tile::{TileCoord, TileGrid, TILE_PIXELS, TILE_SIZE};
use prisme_core::geom::Homography;
use prisme_core::IRect;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

const TS: i32 = TILE_SIZE as i32;
/// Dabs per accumulation dispatch (keeps each dispatch rect tight on long, fast strokes).
const BATCH: usize = 64;
/// Paint dispatches run over blocks of this size (document px, tile aligned) that hold
/// stroke tiles, so temporaries stay small whatever the batch's bounding box.
const BLOCK: i32 = 4 * TS;
/// VRAM the stroke tiles may hold. Past it the stroke overflows to the CPU.
const BUDGET: u64 = 2 << 30;
const G: Globals = Globals { level_scale: 1.0, mip: 0, doc_w: 0.0, doc_h: 0.0, infinite: false };

fn align(r: IRect) -> IRect {
    IRect::new(r.x0.div_euclid(TS) * TS, r.y0.div_euclid(TS) * TS, (r.x1 + TS - 1).div_euclid(TS) * TS, (r.y1 + TS - 1).div_euclid(TS) * TS)
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

/// Copy `r` (document px, clipped to both images) from `src` to `dst`.
pub(crate) fn copy(enc: &mut wgpu::CommandEncoder, src: &GpuImage, dst: &GpuImage, r: IRect) {
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

fn bytes_per_px(format: wgpu::TextureFormat) -> u64 {
    format.block_copy_size(None).unwrap_or(16) as u64
}

/// Upload tile `c` of `grid` at f32 into `img` (missing tiles = `default`).
fn upload_tile(queue: &wgpu::Queue, img: &GpuImage, origin: [u32; 2], grid: &TileGrid, default: f32, c: TileCoord) {
    let ch = grid.channels as usize;
    let view;
    let fill;
    let data: &[f32] = match grid.get_tile(c) {
        Some(t) => {
            view = t.f32_view();
            &view
        }
        // New textures are zero-initialized.
        None if default == 0.0 => return,
        None => {
            fill = vec![default; TILE_PIXELS * ch];
            &fill
        }
    };
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &img.tex.tex,
            mip_level: 0,
            origin: wgpu::Origin3d { x: origin[0], y: origin[1], z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(data),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(TILE_SIZE * 4 * ch as u32), rows_per_image: Some(TILE_SIZE) },
        wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: 1 },
    );
}

/// Level-0 textures of one kind, one per document tile.
struct TileSet {
    format: wgpu::TextureFormat,
    tiles: HashMap<TileCoord, GpuImage>,
}

impl TileSet {
    fn new(format: wgpu::TextureFormat) -> Self {
        Self { format, tiles: HashMap::new() }
    }

    fn bytes(&self) -> u64 {
        self.tiles.len() as u64 * TILE_PIXELS as u64 * bytes_per_px(self.format)
    }

    /// A pool image over `r` holding the tiles present (other pixels are undefined).
    fn gather(&self, enc: &mut wgpu::CommandEncoder, pool: &mut TexturePool, r: IRect) -> GpuImage {
        let img = pool.image(r, self.format);
        for c in TileGrid::coords_in(&r) {
            if let Some(t) = self.tiles.get(&c) {
                copy(enc, t, &img, r);
            }
        }
        img
    }

    /// Write `img` over `r` back into the tiles present.
    fn scatter(&self, enc: &mut wgpu::CommandEncoder, img: &GpuImage, r: IRect) {
        for c in TileGrid::coords_in(&r) {
            if let Some(t) = self.tiles.get(&c) {
                copy(enc, img, t, r);
            }
        }
    }
}

/// A grid's pixels at f32, uploaded tile by tile as the stroke reaches them.
struct Source {
    grid: Arc<TileGrid>,
    default: f32,
    set: TileSet,
}

impl Source {
    fn new(grid: Arc<TileGrid>, default: f32) -> Self {
        let set = TileSet::new(format_for(grid.channels));
        Self { grid, default, set }
    }

    fn tile_bytes(&self) -> u64 {
        TILE_PIXELS as u64 * bytes_per_px(self.set.format)
    }

    fn load(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, c: TileCoord) {
        if self.set.tiles.contains_key(&c) {
            return;
        }
        let img = texture(device, c.rect(), self.set.format);
        upload_tile(queue, &img, [0, 0], &self.grid, self.default, c);
        self.set.tiles.insert(c, img);
        prisme_core::perf::count("stroke base tile uploads", 1);
    }
}

/// A grid's pixels at f32 over one dense tile-aligned rect (Liquify's working area).
fn snapshot(device: &wgpu::Device, queue: &wgpu::Queue, grid: &TileGrid, rect: IRect, default: f32) -> GpuImage {
    let img = texture(device, rect, format_for(grid.channels));
    for c in TileGrid::coords_in(&rect) {
        upload_tile(queue, &img, [(c.x * TS - rect.x0) as u32, (c.y * TS - rect.y0) as u32], grid, default, c);
    }
    img
}

/// Tile state of a dab stroke.
struct Brush {
    mode: StrokeMode,
    /// The target stores f16: the stroke can land through [`Engine::stroke_pack`].
    half: bool,
    base: Source,
    sel: Option<Source>,
    live: TileSet,
    /// Paint only.
    cov: Option<TileSet>,
    /// Smudge: previous `k_retouch_mean` output (carried colour).
    carried: Option<GpuImage>,
}

impl Brush {
    fn bytes(&self) -> u64 {
        self.base.set.bytes() + self.sel.as_ref().map_or(0, |s| s.set.bytes()) + self.live.bytes() + self.cov.as_ref().map_or(0, |c| c.bytes())
    }

    /// Bytes each new stroke tile adds besides its `base` tile.
    fn layer_tile_bytes(&self) -> u64 {
        let px = bytes_per_px(self.live.format) + self.cov.as_ref().map_or(0, |c| bytes_per_px(c.format)) + self.sel.as_ref().map_or(0, |s| bytes_per_px(s.set.format));
        TILE_PIXELS as u64 * px
    }

    /// Create the stroke tiles `layer` (base, selection, live = base, zero coverage) and the
    /// base tiles `src` (clone source). False when that would exceed the budget.
    fn ensure(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, layer: &HashSet<TileCoord>, src: &HashSet<TileCoord>) -> bool {
        let new_layer: Vec<TileCoord> = layer.iter().filter(|c| !self.live.tiles.contains_key(c)).copied().collect();
        let new_base = layer.union(src).filter(|c| !self.base.set.tiles.contains_key(c)).count() as u64;
        let total = self.bytes() + new_base * self.base.tile_bytes() + new_layer.len() as u64 * self.layer_tile_bytes();
        if total > BUDGET {
            log::warn!("stroke needs {} MiB of tiles; finishing it on the CPU", total >> 20);
            return false;
        }
        let _span = prisme_core::perf::span("stroke tiles");
        for &c in layer.union(src) {
            self.base.load(device, queue, c);
        }
        prisme_core::perf::count("stroke tiles created", new_layer.len() as u64);
        for c in new_layer {
            if let Some(s) = &mut self.sel {
                s.load(device, queue, c);
            }
            let live = texture(device, c.rect(), self.live.format);
            copy(enc, &self.base.set.tiles[&c], &live, c.rect());
            self.live.tiles.insert(c, live);
            if let Some(cov) = &mut self.cov {
                cov.tiles.insert(c, texture(device, c.rect(), cov.format));
            }
        }
        true
    }
}

enum Kind {
    Brush(Brush),
    Liquify(Liquify),
    Xform(Xform),
}

/// GPU state of one brush stroke (see the module docs).
pub struct GpuStroke {
    /// The upload this stroke replaces during evaluation (layer pixels, layer mask, saved mask).
    pub key: UploadKey,
    kind: Kind,
    channels: u8,
    /// Union of dab bounds (document px).
    pub dirty: IRect,
    /// Bumped on every change (evaluation cache key).
    pub generation: u64,
    /// The stroke outgrew the VRAM budget or the texture size limit: the preview stops
    /// growing and the host must land the stroke with the CPU engine instead of a readback.
    pub overflow: bool,
    /// Unique per stroke (the engine's composed stroke view is cached under it).
    pub(crate) id: u64,
    /// Generation that last changed each live tile, so a view composed at generation `g`
    /// only re-renders the tiles changed after it.
    pub(crate) tile_gen: HashMap<TileCoord, u64>,
}

impl GpuStroke {
    fn new(key: UploadKey, kind: Kind, channels: u8, overflow: bool) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        GpuStroke { key, kind, channels, dirty: IRect::EMPTY, generation: 0, overflow, id, tile_gen: HashMap::new() }
    }

    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// VRAM held by the stroke's tiles (dab strokes).
    pub fn tile_bytes(&self) -> u64 {
        match &self.kind {
            Kind::Brush(b) => b.bytes(),
            Kind::Liquify(_) | Kind::Xform(_) => 0,
        }
    }

    /// The `live` tiles of a dab stroke.
    /// Stroke tiles created so far.
    pub fn live_tile_count(&self) -> usize {
        self.live_tiles().count()
    }

    pub(crate) fn live_tiles(&self) -> impl Iterator<Item = (TileCoord, &GpuImage)> {
        let tiles = match &self.kind {
            Kind::Brush(b) => Some(b.live.tiles.iter().map(|(c, t)| (*c, t))),
            Kind::Liquify(_) | Kind::Xform(_) => None,
        };
        tiles.into_iter().flatten()
    }

    fn format(&self) -> wgpu::TextureFormat {
        format_for(self.channels)
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

/// Blur radius of a retouch dab (px).
fn blur_radius(d: &Dab) -> i32 {
    (d.radius * 0.15).clamp(1.0, 8.0) as i32
}

/// Clone/heal source offset (whole px).
fn src_offset(m: RetouchMode) -> Option<[i32; 2]> {
    match m {
        RetouchMode::Clone { offset } | RetouchMode::Heal { offset } => Some([offset[0].round() as i32, offset[1].round() as i32]),
        _ => None,
    }
}

impl Engine {
    /// Start a stroke on `base` (the target's pixels at stroke start; missing tiles read as
    /// `base_default`). `key` names the upload the stroke replaces while it's displayed.
    pub fn stroke_begin(&mut self, key: UploadKey, base: Arc<TileGrid>, base_default: f32, selection: Option<Arc<TileGrid>>, mode: StrokeMode) -> GpuStroke {
        let channels = base.channels;
        let paint = matches!(mode, StrokeMode::Paint { .. });
        let brush = Brush {
            mode,
            half: base.format == prisme_core::tile::PixelFormat::F16,
            base: Source::new(base, base_default),
            sel: selection.map(|s| Source::new(s, 0.0)),
            live: TileSet::new(format_for(channels)),
            cov: paint.then(|| TileSet::new(wgpu::TextureFormat::R32Float)),
            carried: None,
        };
        GpuStroke::new(key, Kind::Brush(brush), channels, false)
    }

    /// Rasterize dabs into the stroke (one submission).
    pub fn stroke_dabs(&mut self, s: &mut GpuStroke, dabs: &[Dab]) {
        let _span = prisme_core::perf::span("stroke dabs");
        prisme_core::perf::count("dabs", dabs.len() as u64);
        for d in dabs {
            s.dirty = s.dirty.union(&dab_rect(d));
        }
        let channels = s.channels;
        let format = s.format();
        let Kind::Brush(b) = &mut s.kind else { return };
        if dabs.is_empty() || s.overflow {
            return;
        }
        // The tiles every dab touches (blur also reads around the dab, clone/heal the source).
        let (mut layer, mut src) = (HashSet::new(), HashSet::new());
        let mut widest = 0i64;
        for d in dabs {
            let r = dab_rect(d);
            match b.mode {
                StrokeMode::Paint { .. } => layer.extend(TileGrid::coords_in(&r)),
                StrokeMode::Retouch(m) => {
                    let need = if matches!(m, RetouchMode::Blur { .. }) { r.expand(blur_radius(d)) } else { r };
                    layer.extend(TileGrid::coords_in(&need));
                    if let Some(o) = src_offset(m) {
                        src.extend(TileGrid::coords_in(&r.translate(o[0], o[1])));
                    }
                    widest = widest.max(need.area());
                }
            }
        }
        // Retouch dabs run whole (the means span the dab): their temporaries must fit too.
        let max = self.ctx.device.limits().max_texture_dimension_2d as i64;
        let side = dabs.iter().map(|d| dab_rect(d).width().max(dab_rect(d).height()) as i64 + 16).max().unwrap_or(0);
        let temps = widest as u64 * (3 * bytes_per_px(format) + 4);
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("stroke") });
        if temps > BUDGET / 4 || side > max || !b.ensure(&self.ctx.device, &self.ctx.queue, &mut enc, &layer, &src) {
            s.overflow = true;
            prisme_core::perf::count("stroke overflowed (lands on CPU)", 1);
            return;
        }
        let scope = match &mut self.timer {
            Some(t) => t.begin(&self.ctx.device, &self.ctx.queue, &mut enc, "stroke dabs"),
            None => crate::timer::Scope::NONE,
        };
        match b.mode {
            StrokeMode::Paint { target, opacity, lock_alpha } => {
                for chunk in dabs.chunks(BATCH) {
                    self.paint_batch(&mut enc, b, channels, chunk, target, opacity, lock_alpha);
                }
            }
            StrokeMode::Retouch(m) => {
                for d in dabs {
                    self.retouch_dab(&mut enc, b, m, d);
                }
            }
        }
        if let Some(t) = &mut self.timer {
            t.end(&mut enc, scope);
        }
        self.ctx.queue.submit([enc.finish()]);
        s.generation += 1;
        for c in layer {
            s.tile_gen.insert(c, s.generation);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn paint_batch(&mut self, enc: &mut wgpu::CommandEncoder, b: &Brush, channels: u8, dabs: &[Dab], target: BlendTarget, opacity: f32, lock_alpha: bool) {
        let r = dabs.iter().fold(IRect::EMPTY, |a, d| a.union(&dab_rect(d)));
        let data: Vec<[f32; 4]> = dabs.iter().flat_map(dab_vecs).collect();
        let lut = self.kernels.lut_buffer(&data);
        let (mode, col, value) = match target {
            BlendTarget::Color(c) => (0.0, c, 0.0),
            BlendTarget::Erase => (1.0, [0.0; 4], 0.0),
            BlendTarget::Value(v) => (2.0, [0.0; 4], v),
        };
        let p = [col[0], col[1], col[2], col[3], mode, opacity, lock_alpha as i32 as f32, channels as f32, value];
        let Some(cov) = &b.cov else { return };
        let (bx0, by0) = (r.x0.div_euclid(BLOCK), r.y0.div_euclid(BLOCK));
        let (bx1, by1) = ((r.x1 - 1).div_euclid(BLOCK), (r.y1 - 1).div_euclid(BLOCK));
        for by in by0..=by1 {
            for bx in bx0..=bx1 {
                let br = IRect::new(bx * BLOCK, by * BLOCK, (bx + 1) * BLOCK, (by + 1) * BLOCK).intersect(&r);
                if !TileGrid::coords_in(&br).any(|c| b.live.tiles.contains_key(&c)) {
                    continue;
                }
                let prev = cov.gather(enc, &mut self.pool, br);
                let acc = self.pool.image(br, wgpu::TextureFormat::R32Float);
                self.kernels.dispatch(enc, "k_stroke_dabs", &acc, None, [Some(&prev), None, None, None], &[dabs.len() as f32], Some(&lut), &G);
                cov.scatter(enc, &acc, br);
                let base = b.base.set.gather(enc, &mut self.pool, br);
                let sel = b.sel.as_ref().map(|s| s.set.gather(enc, &mut self.pool, br));
                let out = self.pool.image(br, b.live.format);
                self.kernels.dispatch(enc, "k_stroke_paint", &out, None, [Some(&base), Some(&acc), sel.as_ref(), None], &p, None, &G);
                b.live.scatter(enc, &out, br);
            }
        }
    }

    fn retouch_dab(&mut self, enc: &mut wgpu::CommandEncoder, b: &mut Brush, m: RetouchMode, d: &Dab) {
        let r = dab_rect(d);
        let rad = blur_radius(d);
        let (id, strength) = match m {
            RetouchMode::Smudge { strength } => (0.0, strength),
            RetouchMode::Blur { strength } => (1.0, strength),
            RetouchMode::Clone { .. } => (2.0, 0.0),
            RetouchMode::Heal { .. } => (3.0, 0.0),
        };
        let off = src_offset(m).unwrap_or([0, 0]);
        let need = if id == 1.0 { r.expand(rad) } else { r };
        let live = b.live.gather(enc, &mut self.pool, need);
        let src = src_offset(m).map(|_| b.base.set.gather(enc, &mut self.pool, r.translate(off[0], off[1])));
        let sel = b.sel.as_ref().map(|s| s.set.gather(enc, &mut self.pool, r));
        let mut p: Vec<f32> = dab_vecs(d).concat();
        p.extend([id, strength, off[0] as f32, off[1] as f32, rad as f32, b.carried.is_none() as i32 as f32, 0.0, 0.0]);
        let mean = if matches!(m, RetouchMode::Smudge { .. } | RetouchMode::Heal { .. }) {
            let out = self.pool.image(IRect::new(0, 0, 2, 1), wgpu::TextureFormat::Rgba32Float);
            self.kernels.dispatch(enc, "k_retouch_mean", &out, None, [Some(&live), src.as_ref(), sel.as_ref(), b.carried.as_ref()], &p, None, &G);
            Some(out)
        } else {
            None
        };
        let out = self.pool.image(r, b.live.format);
        self.kernels.dispatch(enc, "k_retouch_apply", &out, None, [Some(&live), src.as_ref(), sel.as_ref(), mean.as_ref()], &p, None, &G);
        b.live.scatter(enc, &out, r);
        if matches!(m, RetouchMode::Smudge { .. }) {
            b.carried = mean;
        }
    }

    /// Start reading the finished stroke back, one readback per touched tile (async; poll
    /// with [`Readback::ready`]). None when there's nothing to read or the stroke overflowed
    /// (land it on the CPU).
    /// Land a stroke on an f16 target: its whole live tiles (base outside the stroke) converted
    /// to f16 on the GPU and read back as tiles, bit-identical to converting the f32 read-back
    /// on the CPU. None on overflow, for f32 targets, or with nothing painted.
    pub fn stroke_pack(&mut self, s: &GpuStroke) -> Option<crate::pack::PackedTiles> {
        let Kind::Brush(b) = &s.kind else { return None };
        if s.overflow || !b.half {
            return None;
        }
        let mut tiles: Vec<(TileCoord, GpuImage)> = s.live_tiles().filter(|(c, _)| !c.rect().intersect(&s.dirty).is_empty()).map(|(c, t)| (c, t.clone())).collect();
        if tiles.is_empty() {
            return None;
        }
        tiles.sort_unstable_by_key(|(c, _)| (c.y, c.x));
        Some(self.pack_tiles(&tiles, s.channels))
    }

    /// Convert f32 tile textures (`channels` 1 or 4) to f16 tiles on the GPU and start reading
    /// them back (see [`crate::pack`]).
    pub fn pack_tiles(&mut self, tiles: &[(TileCoord, GpuImage)], channels: u8) -> crate::pack::PackedTiles {
        let _span = prisme_core::perf::span("pack tiles");
        let device = self.ctx.device.clone();
        let mut enc = device.create_command_encoder(&Default::default());
        let scope = self.gpu_begin(&mut enc, "pack tiles");
        self.ctx.queue.submit([enc.finish()]);
        let p = self.pack.get_or_insert_with(|| crate::pack::PackPipelines::new(&device));
        let packed = crate::pack::PackedTiles::start(&self.ctx.device, &self.ctx.queue, p, tiles, channels);
        let mut enc = device.create_command_encoder(&Default::default());
        self.gpu_end(&mut enc, scope);
        self.ctx.queue.submit([enc.finish()]);
        packed
    }

    pub fn stroke_readback(&self, s: &GpuStroke) -> Option<Vec<Readback>> {
        if s.overflow {
            return None;
        }
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("stroke-readback") });
        let rbs: Vec<Readback> = s
            .live_tiles()
            .filter(|(c, _)| !c.rect().intersect(&s.dirty).is_empty())
            .map(|(c, t)| Readback::record(&self.ctx.device, &mut enc, t, c.rect().intersect(&s.dirty)))
            .collect();
        if rbs.is_empty() {
            return None;
        }
        self.ctx.queue.submit([enc.finish()]);
        for rb in &rbs {
            rb.map();
        }
        Some(rbs)
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
    /// Layer pixels and selection over the working area (None on overflow).
    base: Option<GpuImage>,
    sel: Option<GpuImage>,
}

impl GpuStroke {
    pub fn is_liquify(&self) -> bool {
        matches!(self.kind, Kind::Liquify(_))
    }

    /// `k_liquify` inputs (base, field, selection) and params for level scale `k`.
    pub(crate) fn liquify_args(&self, k: i32, bicubic: bool, default: f32) -> Option<([GpuImage; 2], Option<GpuImage>, Vec<f32>)> {
        let Kind::Liquify(l) = &self.kind else { return None };
        let base = l.base.clone().filter(|_| !self.overflow)?;
        let sel = l.sel.clone();
        let r = l.rect;
        let show = l.show_mask && !bicubic;
        let p = vec![k as f32, l.step as f32, show as i32 as f32, bicubic as i32 as f32, r.x0 as f32, r.y0 as f32, r.x1 as f32, r.y1 as f32, default];
        Some(([base, l.field.clone()], sel, p))
    }

    /// Working area of a Liquify session.
    pub fn liquify_rect(&self) -> IRect {
        match &self.kind {
            Kind::Liquify(l) => l.rect,
            Kind::Brush(_) | Kind::Xform(_) => IRect::EMPTY,
        }
    }
}

impl Engine {
    /// Start a Liquify session on `base` (layer pixels, missing tiles transparent) over the
    /// working area `rect`, with a `nodes` field of `step` px per node.
    pub fn liquify_begin(&mut self, key: UploadKey, base: Arc<TileGrid>, selection: Option<Arc<TileGrid>>, rect: IRect, step: i32, nodes: [i32; 2]) -> GpuStroke {
        let (device, queue) = (self.ctx.device.clone(), self.ctx.queue.clone());
        let channels = base.channels;
        let field = texture(&device, IRect::new(0, 0, nodes[0], nodes[1]), wgpu::TextureFormat::Rgba32Float);
        let want = align(rect);
        let max = device.limits().max_texture_dimension_2d as i32;
        let bytes = want.area() as u64 * (bytes_per_px(format_for(channels)) + if selection.is_some() { 4 } else { 0 });
        let overflow = want.width() > max || want.height() > max || nodes[0] > max || nodes[1] > max || bytes > BUDGET;
        let (base, sel) = if overflow {
            log::warn!("liquify area {want:?} exceeds GPU limits; no GPU preview");
            (None, None)
        } else {
            (Some(snapshot(&device, &queue, &base, want, 0.0)), selection.map(|s| snapshot(&device, &queue, &s, want, 0.0)))
        };
        let liquify = Liquify { field, rect, step, show_mask: false, base, sel };
        GpuStroke::new(key, Kind::Liquify(liquify), channels, overflow)
    }

    /// Upload changed field nodes (`nodes` rect, row-major values).
    pub fn liquify_update(&mut self, s: &mut GpuStroke, nodes: IRect, data: &[[f32; 4]]) {
        let Kind::Liquify(l) = &s.kind else { return };
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
        match &mut s.kind {
            Kind::Liquify(l) if l.show_mask != on => {
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

/// Free Transform preview state (see the module docs).
struct Xform {
    /// Pixels being transformed, and the moving layer mask with its missing-tile value
    /// (level 0, straight; None on overflow).
    base: Option<GpuImage>,
    mask: Option<(GpuImage, f32)>,
    /// Document px -> source px.
    inv: [f32; 9],
    /// Source rect, and its transformed bounds (document px).
    rect: IRect,
    bounds: IRect,
}

impl GpuStroke {
    pub fn is_xform(&self) -> bool {
        matches!(self.kind, Kind::Xform(_))
    }

    /// `k_xform` source and params when `key` is one of the preview's uploads (the layer's
    /// pixels or its moving mask), at level scale `k`.
    pub(crate) fn xform_args(&self, key: UploadKey, k: i32) -> Option<(GpuImage, Vec<f32>)> {
        let Kind::Xform(x) = &self.kind else { return None };
        let UploadKey::Content(id) = self.key else { return None };
        let (img, mask, default) = match key {
            _ if key == self.key => (x.base.clone()?, false, 0.0),
            UploadKey::LayerMask(m) if m == id => {
                let (img, d) = x.mask.clone()?;
                (img, true, d)
            }
            _ => return None,
        };
        let (m, b) = (x.inv, x.bounds);
        let p = vec![k as f32, mask as i32 as f32, default, 0.0, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], 0.0, 0.0, 0.0, b.x0 as f32, b.y0 as f32, b.x1 as f32, b.y1 as f32];
        Some((img, p))
    }
}

impl Engine {
    /// Start a Free Transform preview of `source` (layer `key`'s pixels over `rect`), with the
    /// layer mask when it moves along.
    pub fn xform_begin(&mut self, key: UploadKey, source: Arc<TileGrid>, rect: IRect, mask: Option<(Arc<TileGrid>, f32)>) -> GpuStroke {
        let (device, queue) = (self.ctx.device.clone(), self.ctx.queue.clone());
        let want = align(rect);
        let max = device.limits().max_texture_dimension_2d as i32;
        let bytes = want.area() as u64 * (bytes_per_px(format_for(source.channels)) + if mask.is_some() { 4 } else { 0 });
        let overflow = want.width() > max || want.height() > max || bytes > BUDGET;
        let (base, mask) = if overflow {
            log::warn!("transform source {want:?} exceeds GPU limits; no GPU preview");
            (None, None)
        } else {
            (Some(snapshot(&device, &queue, &source, want, 0.0)), mask.map(|(m, d)| (snapshot(&device, &queue, &m, want, d), d)))
        };
        let xform = Xform { base, mask, inv: Homography::IDENTITY.m, rect, bounds: rect };
        GpuStroke::new(key, Kind::Xform(xform), source.channels, overflow)
    }

    /// Move the preview: `h` maps source px to document px.
    pub fn xform_update(&mut self, s: &mut GpuStroke, h: &Homography) {
        let Kind::Xform(x) = &mut s.kind else { return };
        let Some(inv) = h.inverse() else { return };
        let r = x.rect;
        let q = [[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1]].map(|p| h.apply([p[0] as f32, p[1] as f32]));
        if q.iter().any(|p| !p[0].is_finite() || !p[1].is_finite()) {
            return;
        }
        x.inv = inv.m;
        x.bounds = quad_bounds(&q).expand(1);
        s.generation += 1;
    }
}

fn quad_bounds(q: &[[f32; 2]; 4]) -> IRect {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in q {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    IRect::from_f32(x0, y0, x1, y1)
}
