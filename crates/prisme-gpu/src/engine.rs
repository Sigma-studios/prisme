//! Viewport-driven node graph evaluation.
//!
//! A request asks for the image of a node (usually the graph output) over a level
//! rect at a mip level. Evaluation pulls inputs recursively, expanding the rect by each
//! node's padding (ROI). Every node result is cached under a content hash that covers
//! its kind, parameters, inputs and referenced layers/masks, so editing one node only
//! re-runs that node and its downstream chain.

use crate::context::GpuContext;
use crate::image::{GpuImage, TexturePool};
use crate::kernels::{Globals, Kernels, OutFormat};
use crate::stroke::GpuStroke;
use crate::upload::{UploadCache, UploadKey};
use prisme_core::graph::{Node, ParamValue};
use prisme_core::layer::{FillContent, Layer, LayerKind};
use prisme_core::tile::TILE_SIZE;
use prisme_core::{DocUid, Document, Gradient, IRect, LayerId, NodeId, TileGrid};
use prisme_graph::{eval_curve, registry, SocketType, CURVE_CHANNELS, LEVEL_CHANNELS};
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::Arc;
use prisme_core::perf;

/// Spare textures the pool always keeps (bytes).
const BASE_POOL_FREE: u64 = 256 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EvalFormat {
    /// Half-float intermediates: interactive preview.
    F16,
    /// Full float intermediates: export / lossless paths.
    F32,
}

impl EvalFormat {
    fn out(self) -> OutFormat {
        match self {
            EvalFormat::F16 => OutFormat::Rgba16F,
            EvalFormat::F32 => OutFormat::Rgba32F,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderSource {
    /// The graph output (default view).
    Graph,
    /// The raw layer stack, ignoring nodes.
    LayerStack,
    /// Preview a specific node's first output.
    Node(NodeId),
}

pub struct EvalRequest<'a> {
    pub doc: &'a Document,
    /// Level-space rect (document pixels / 2^mip).
    pub rect: IRect,
    pub mip: u32,
    pub format: EvalFormat,
    pub source: RenderSource,
}

/// Renders non-raster layers (text, vector) into level-space grids.
pub trait LayerRasterizer: Send + Sync {
    /// Rasterize `layer` at `mip` (grid coordinates are level pixels). Straight alpha.
    fn rasterize(&self, layer: &Layer, doc: &Document, mip: u32) -> Option<TileGrid>;
}

struct CachedResult {
    image: GpuImage,
    last_used: u64,
}

pub struct Engine {
    pub ctx: GpuContext,
    pub kernels: Kernels,
    pub(crate) pool: TexturePool,
    uploads: UploadCache,
    /// Rasterized text/vector layers per (document, layer, level): content hash, grid, last
    /// frame used.
    rasters: HashMap<(DocUid, LayerId, u32), (u64, Arc<TileGrid>, u64)>,
    results: HashMap<u64, CachedResult>,
    luts: HashMap<u64, Arc<wgpu::Buffer>>,
    pub rasterizer: Option<Arc<dyn LayerRasterizer>>,
    frame: u64,
    /// Statistics of the last evaluation.
    pub last_stats: EvalStats,
    /// Free texture bytes the pool keeps, and the frame that set it (see `evaluate_with`).
    pool_keep: (u64, u64),
    /// f16 landing pipelines (created on first use).
    pub(crate) pack: Option<crate::pack::PackPipelines>,
    /// GPU timestamps for the Performance panel (None without `TIMESTAMP_QUERY`).
    pub timer: Option<crate::timer::GpuTimer>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EvalStats {
    pub dispatches: u32,
    pub cache_hits: u32,
    pub nodes_run: u32,
}

fn hasher() -> ahash::AHasher {
    ahash::RandomState::with_seeds(0x5052_4953, 0x4d45, 0x9e37_79b9, 0x7f4a_7c15).build_hasher()
}

fn hash_bytes<T: serde::Serialize>(v: &T) -> u64 {
    let mut h = hasher();
    if let Ok(b) = postcard::to_allocvec(v) {
        b.hash(&mut h);
    }
    h.finish()
}

fn grid_hash(g: &TileGrid, h: &mut ahash::AHasher) {
    let mut ids: Vec<(i32, i32, u64)> = g.tiles.iter().map(|(c, t)| (c.x, c.y, t.id())).collect();
    ids.sort_unstable();
    ids.hash(h);
}

struct Frame<'a> {
    doc: &'a Document,
    mip: u32,
    format: EvalFormat,
    g: Globals,
    enc: wgpu::CommandEncoder,
    layer_hashes: HashMap<LayerId, u64>,
    node_hashes: HashMap<NodeId, u64>,
    stats: EvalStats,
    /// In-progress (or not yet landed) brush stroke replacing one upload.
    stroke: Option<&'a GpuStroke>,
}

impl Engine {
    pub fn new(ctx: GpuContext) -> Self {
        let kernels = Kernels::new(&ctx.device);
        let pool = TexturePool::new(ctx.device.clone());
        let uploads = UploadCache::new(ctx.device.clone(), ctx.queue.clone());
        let timer = crate::timer::GpuTimer::new(&ctx.device, &ctx.queue);
        Engine {
            ctx,
            kernels,
            pool,
            uploads,
            rasters: HashMap::new(),
            results: HashMap::new(),
            luts: HashMap::new(),
            rasterizer: None,
            frame: 0,
            last_stats: EvalStats::default(),
            pool_keep: (0, 0),
            pack: None,
            timer,
        }
    }

    /// Drop all cached results (e.g. after a device-level change).
    pub fn clear_caches(&mut self) {
        self.results.clear();
        self.rasters.clear();
        self.luts.clear();
    }

    /// Free everything held for a closed document: its layer views, tile keys and rasterized
    /// layers, the tiles no other document uses (their pages repacked so the memory goes
    /// back), and cached results and spare textures sized for its views.
    pub fn release_document(&mut self, doc: DocUid) {
        self.uploads.release(doc);
        self.rasters.retain(|k, _| k.0 != doc);
        self.results.clear();
        self.pool_keep = (0, self.frame);
        self.pool.trim(BASE_POOL_FREE);
    }

    /// Resident tiles: (uploaded from the CPU, built on the GPU, VRAM bytes) so far.
    pub fn tile_stats(&self) -> (u64, u64, u64) {
        let t = &self.uploads.store;
        (t.uploads, t.builds, t.bytes())
    }

    /// Idle-time work: make the tiles of the visible raster layers (and their masks) over
    /// `regions` (mip, level rect) resident before a view needs them, so the first pan or
    /// zoom-out into them doesn't upload or build anything. Stops when `out_of_time`; true
    /// when there's nothing left to do.
    pub fn prefetch(&mut self, doc: &Document, regions: &[(u32, IRect)], out_of_time: &dyn Fn() -> bool) -> bool {
        let _span = perf::span("prefetch");
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("prefetch") });
        let mut done = true;
        'all: for &(mip, rect) in regions {
            for l in doc.all_layers() {
                if !l.visible {
                    continue;
                }
                if let LayerKind::Raster(g) = &l.kind
                    && !self.uploads.prefetch(&mut self.kernels, &mut enc, doc.uid, UploadKey::Content(l.id.0), g, mip, rect, false, 0.0, out_of_time)
                {
                    done = false;
                    break 'all;
                }
                if let Some(m) = l.mask.as_ref().filter(|m| m.enabled) {
                    let d = if m.default_visible { 1.0 } else { 0.0 };
                    if !self.uploads.prefetch(&mut self.kernels, &mut enc, doc.uid, UploadKey::LayerMask(l.id.0), &m.grid, mip, rect, false, d, out_of_time) {
                        done = false;
                        break 'all;
                    }
                }
            }
        }
        self.ctx.queue.submit([enc.finish()]);
        done
    }

    /// Hand landed f16 tiles to the tile store: `ids[i]` is the id the document gave
    /// `p.coords[i]`'s tile (0: not kept). The store copies their bits from the GPU buffers they
    /// were read back from, so views don't upload them again. RGBA only (masks are stored as
    /// f32 on the GPU and upload as usual).
    pub fn adopt_tiles(&mut self, doc: DocUid, p: &crate::pack::PackedTiles, ids: &[u64]) {
        if p.channels() != 4 {
            return;
        }
        let _span = perf::span("adopt tiles");
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("adopt") });
        for (i, &id) in ids.iter().enumerate() {
            if let (true, Some(src)) = (id != 0, p.bits(i)) {
                self.uploads.store.adopt(&mut enc, doc, crate::tiles::Class::Rgba16, id, src);
            }
        }
        self.ctx.queue.submit([enc.finish()]);
    }

    /// Drop resident tiles no view used for `max_age` evaluations and give the memory back
    /// (also done periodically on its own).
    pub fn trim_tiles(&mut self, max_age: u64) {
        self.uploads.store.trim(max_age);
    }

    /// VRAM the resident tiles may hold (default 1 GiB).
    pub fn set_tile_budget(&mut self, bytes: u64) {
        self.uploads.store.budget = bytes;
    }

    /// Start a GPU-timed scope in `enc` (see [`crate::timer`]).
    pub(crate) fn gpu_begin(&mut self, enc: &mut wgpu::CommandEncoder, label: &'static str) -> crate::timer::Scope {
        match &mut self.timer {
            Some(t) => t.begin(&self.ctx.device, &self.ctx.queue, enc, label),
            None => crate::timer::Scope::NONE,
        }
    }

    pub(crate) fn gpu_end(&mut self, enc: &mut wgpu::CommandEncoder, scope: crate::timer::Scope) {
        if let Some(t) = &mut self.timer {
            t.end(enc, scope);
        }
    }

    /// Per UI frame: deliver GPU timings that came back.
    pub fn poll_timer(&mut self) {
        if let Some(t) = &mut self.timer {
            t.poll(&self.ctx.device, &self.ctx.queue);
        }
    }

    pub fn evaluate(&mut self, req: &EvalRequest) -> Option<GpuImage> {
        self.evaluate_with(req, None)
    }

    /// Evaluate with a GPU brush stroke spliced into its target (layer pixels or mask).
    pub fn evaluate_with(&mut self, req: &EvalRequest, stroke: Option<&GpuStroke>) -> Option<GpuImage> {
        if req.rect.is_empty() {
            return None;
        }
        let _span = perf::span("evaluate");
        self.uploads.begin_frame();
        self.frame += 1;
        let level_scale = 1.0 / (1u32 << req.mip) as f32;
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("eval") });
        let gpu = self.gpu_begin(&mut enc, "evaluate");
        let mut f = Frame {
            doc: req.doc,
            mip: req.mip,
            format: req.format,
            g: Globals { level_scale, mip: req.mip, doc_w: req.doc.info.width as f32, doc_h: req.doc.info.height as f32, infinite: req.doc.info.is_infinite() },
            enc,
            layer_hashes: HashMap::new(),
            node_hashes: HashMap::new(),
            stats: EvalStats::default(),
            stroke,
        };
        let ov = stroke.map(|s| (s.key, s.generation));
        fn hash_layers(v: &[Layer], out: &mut HashMap<LayerId, u64>, ov: Option<(UploadKey, u64)>) -> u64 {
            let mut all = hasher();
            for l in v {
                let mut h = hasher();
                l.id.hash(&mut h);
                l.visible.hash(&mut h);
                l.opacity.to_bits().hash(&mut h);
                (l.blend as u32).hash(&mut h);
                if let Some(m) = &l.mask {
                    m.enabled.hash(&mut h);
                    m.default_visible.hash(&mut h);
                    grid_hash(&m.grid, &mut h);
                }
                match &l.kind {
                    LayerKind::Raster(g) => grid_hash(g, &mut h),
                    LayerKind::Group(g) => hash_layers(&g.children, out, ov).hash(&mut h),
                    LayerKind::Text(t) => hash_bytes(t).hash(&mut h),
                    LayerKind::Vector(v) => hash_bytes(v).hash(&mut h),
                    LayerKind::Fill(fl) => hash_bytes(fl).hash(&mut h),
                }
                if let Some((UploadKey::Content(id) | UploadKey::LayerMask(id), g)) = ov
                    && id == l.id.0 {
                        (0x5757_u32, g).hash(&mut h);
                    }
                let v = h.finish();
                out.insert(l.id, v);
                v.hash(&mut all);
            }
            all.finish()
        }
        {
            let _s = perf::span("hash layers");
            hash_layers(&req.doc.layers, &mut f.layer_hashes, ov);
        }
        let record = perf::span("record");

        let result = match req.source {
            RenderSource::Graph => self.eval_node(&mut f, req.doc.graph.output, 0, req.rect, SocketType::Image),
            RenderSource::Node(id) => self.eval_node(&mut f, id, 0, req.rect, SocketType::Image),
            RenderSource::LayerStack => {
                let mut h = hasher();
                for l in req.doc.all_layers() {
                    f.layer_hashes.get(&l.id).hash(&mut h);
                }
                (req.mip, req.format, 0x57ac_u32).hash(&mut h);
                let key = h.finish();
                match self.results.get_mut(&key).filter(|c| c.image.rect.intersect(&req.rect) == req.rect) {
                    Some(c) => {
                        c.last_used = self.frame;
                        Some(c.image.clone())
                    }
                    None => {
                        let img = self.composite(&mut f, &req.doc.layers, req.rect, &HashSet::new());
                        self.results.insert(key, CachedResult { image: img.clone(), last_used: self.frame });
                        Some(img)
                    }
                }
            }
        };
        let result = match result {
            Some(r) => r,
            None => {
                let out = self.new_image(&f, req.rect);
                self.run(&mut f, "k_fill", &out, [None; 4], &[0.0; 4], None);
                out
            }
        };
        drop(record);
        self.last_stats = f.stats;
        perf::count("dispatches", f.stats.dispatches as u64);
        perf::count("nodes run", f.stats.nodes_run as u64);
        perf::count("node cache hits", f.stats.cache_hits as u64);
        let mut enc = f.enc;
        self.gpu_end(&mut enc, gpu);
        {
            let _s = perf::span("submit");
            self.ctx.queue.submit([enc.finish()]);
        }
        let _s = perf::span("cache upkeep");
        self.uploads.end_frame();
        let fr = self.frame;
        // Recent results, newest first, up to twice this one's size: while zooming every
        // result has a new rect, and holding six full-view images blocks their textures'
        // reuse (each step then allocates new ones).
        let cap = 2 * result.byte_size();
        let mut recent: Vec<(u64, u64)> = self.results.iter().filter(|(_, r)| fr - r.last_used < 6).map(|(k, r)| (r.last_used, *k)).collect();
        recent.sort_unstable_by(|a, b| b.cmp(a));
        let mut bytes = 0;
        let keep: HashSet<u64> = recent
            .into_iter()
            .take_while(|(_, k)| {
                bytes += self.results[k].image.byte_size();
                bytes <= cap
            })
            .map(|(_, k)| k)
            .collect();
        self.results.retain(|k, r| keep.contains(k) || r.last_used == fr);
        // Keep enough free textures for the next evaluation's intermediates: a fixed budget
        // smaller than one full-view image (4K, infinite canvas) re-allocates and clears every
        // texture on every pan or zoom step. Thumbnails interleave with the canvas, so the
        // largest recent need sets it.
        let px_bytes = if req.format == EvalFormat::F32 { 16 } else { 8 };
        let need = req.rect.area() as u64 * px_bytes * 4;
        if need >= self.pool_keep.0 || fr - self.pool_keep.1 > 300 {
            self.pool_keep = (need, fr);
        }
        self.pool.trim(BASE_POOL_FREE.max(self.pool_keep.0));
        self.rasters.retain(|_, r| fr - r.2 < 600);
        Some(result)
    }

    /// Evaluate a node over [`Document::extent`] at a small size and convert to sRGB8
    /// (navigator, node editor thumbnails).
    pub fn thumbnail(&mut self, doc: &Document, node: NodeId, max_side: u32) -> Option<GpuImage> {
        let _span = perf::span("thumbnail");
        let ext = doc.extent();
        let side = ext.width().max(ext.height()).max(1) as u32;
        let mut mip = 0;
        while (side >> mip) > max_side * 2 && mip < 12 {
            mip += 1;
        }
        let s = 1 << mip;
        let rect = IRect::new(ext.x0.div_euclid(s), ext.y0.div_euclid(s), (ext.x1 + s - 1).div_euclid(s), (ext.y1 + s - 1).div_euclid(s));
        let img = self.evaluate(&EvalRequest { doc, rect, mip, format: EvalFormat::F16, source: RenderSource::Node(node) })?;
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        let out = self.pool.image(rect, wgpu::TextureFormat::Rgba8Unorm);
        let g = Globals { level_scale: 1.0 / s as f32, mip, doc_w: doc.info.width as f32, doc_h: doc.info.height as f32, infinite: doc.info.is_infinite() };
        self.kernels.dispatch(&mut enc, "k_thumb", &out, None, [Some(&img), None, None, None], &[], None, &g);
        self.ctx.queue.submit([enc.finish()]);
        Some(out)
    }

    // ------------------------------------------------------------------ helpers

    fn new_image(&mut self, f: &Frame, rect: IRect) -> GpuImage {
        self.pool.image(rect, f.format.out().texture_format())
    }

    fn run(&mut self, f: &mut Frame, entry: &'static str, out: &GpuImage, inputs: [Option<&GpuImage>; 4], params: &[f32], lut: Option<&wgpu::Buffer>) {
        f.stats.dispatches += 1;
        self.kernels.dispatch(&mut f.enc, entry, out, None, inputs, params, lut, &f.g);
    }

    fn lut(&mut self, key: u64, make: impl FnOnce() -> Vec<[f32; 4]>) -> Arc<wgpu::Buffer> {
        if let Some(b) = self.luts.get(&key) {
            return b.clone();
        }
        let b = Arc::new(self.kernels.lut_buffer(&make()));
        if self.luts.len() > 256 {
            self.luts.clear();
        }
        self.luts.insert(key, b.clone());
        b
    }

    fn gradient_lut(&mut self, g: &Gradient) -> Arc<wgpu::Buffer> {
        let key = hash_bytes(g);
        self.lut(key, || g.to_lut(256))
    }

    fn node_hash(&self, f: &mut Frame, id: NodeId) -> u64 {
        if let Some(h) = f.node_hashes.get(&id) {
            return *h;
        }
        let graph = &f.doc.graph;
        let Some(node) = graph.nodes.get(&id) else { return 0 };
        let mut h = hasher();
        node.kind.hash(&mut h);
        hash_bytes(&node.params).hash(&mut h);
        node.bypass.hash(&mut h);
        f.mip.hash(&mut h);
        f.format.hash(&mut h);
        f.doc.info.width.hash(&mut h);
        f.doc.info.height.hash(&mut h);
        f.doc.info.canvas.hash(&mut h);
        let mut ins: Vec<_> = graph.links.iter().filter(|l| l.to == id).copied().collect();
        ins.sort_by_key(|l| l.to_in);
        for l in ins {
            l.to_in.hash(&mut h);
            l.from_out.hash(&mut h);
            self.node_hash(f, l.from).hash(&mut h);
        }
        match node.kind.as_str() {
            prisme_core::graph::KIND_LAYER_STACK => {
                for l in f.doc.all_layers() {
                    f.layer_hashes.get(&l.id).hash(&mut h);
                }
            }
            "layer" => {
                if let Some(ParamValue::Layer(Some(lid))) = node.params.get("layer") {
                    f.layer_hashes.get(lid).hash(&mut h);
                }
            }
            "selection" => {
                if let Some(s) = &f.doc.selection {
                    grid_hash(s, &mut h);
                }
            }
            "mask" => {
                if let Some(ParamValue::Mask(Some(mid))) = node.params.get("mask")
                    && let Some(m) = f.doc.saved_mask(*mid) {
                        grid_hash(&m.mask, &mut h);
                        if let Some(s) = f.stroke.filter(|s| s.key == UploadKey::SavedMask(mid.0)) {
                            s.generation.hash(&mut h);
                        }
                    }
            }
            _ => {}
        }
        let v = h.finish();
        f.node_hashes.insert(id, v);
        v
    }

    /// Evaluate output `out_idx` of `id` over `rect`, coerced to `want`.
    fn eval_node(&mut self, f: &mut Frame, id: NodeId, out_idx: u16, rect: IRect, want: SocketType) -> Option<GpuImage> {
        let node = f.doc.graph.nodes.get(&id)?.clone();
        let desc = registry().get(&node.kind)?;
        let produced = desc.outputs.get(out_idx as usize).map(|s| s.ty).unwrap_or(SocketType::Image);
        let key = self.node_hash(f, id) ^ (out_idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let img = if let Some(c) = self.results.get_mut(&key).filter(|c| c.image.rect.intersect(&rect) == rect) {
            c.last_used = self.frame;
            f.stats.cache_hits += 1;
            Some(c.image.clone())
        } else {
            f.stats.nodes_run += 1;
            let r = if node.bypass && !desc.inputs.is_empty() {
                self.input(f, &node, 0, rect, produced)
            } else {
                self.eval_kind(f, &node, rect)
            };
            if let Some(img) = &r {
                self.results.insert(key, CachedResult { image: img.clone(), last_used: self.frame });
            }
            r
        }?;
        if produced == want {
            return Some(img);
        }
        // Coerce between masks and images.
        let out = self.new_image(f, rect);
        match want {
            SocketType::Mask => self.run(f, "k_to_mask", &out, [Some(&img), None, None, None], &[], None),
            SocketType::Image => self.run(f, "k_mask_to_image", &out, [Some(&img), None, None, None], &[0.0], None),
        }
        Some(out)
    }

    fn input(&mut self, f: &mut Frame, node: &Node, idx: u16, rect: IRect, want: SocketType) -> Option<GpuImage> {
        let (from, out) = f.doc.graph.input(node.id, idx)?;
        self.eval_node(f, from, out, rect, want)
    }

    fn pad(&self, f: &Frame, node: &Node) -> i32 {
        let desc = registry().get(&node.kind).unwrap();
        ((desc.pad)(node) * f.g.level_scale).ceil() as i32 + 1
    }

    /// Run a standard adjustment: image on input 0, optional mask on input 1 (bound to slot 3).
    fn adjust(&mut self, f: &mut Frame, node: &Node, rect: IRect, entry: &'static str, params: &[f32], lut: Option<&wgpu::Buffer>) -> Option<GpuImage> {
        let src = self.input(f, node, 0, rect, SocketType::Image)?;
        let mask = self.input(f, node, 1, rect, SocketType::Mask);
        let out = self.new_image(f, rect);
        self.run(f, entry, &out, [Some(&src), None, None, mask.as_ref()], params, lut);
        Some(out)
    }

    /// Gaussian blur of `src` producing an image over `rect` (src must cover rect + 3σ).
    fn gaussian(&mut self, f: &mut Frame, src: &GpuImage, rect: IRect, sigma: f32, edges: f32) -> GpuImage {
        if sigma < 0.25 {
            let out = self.new_image(f, rect);
            self.run(f, "k_copy", &out, [Some(src), None, None, None], &[], None);
            return out;
        }
        let r = (sigma * 3.0).ceil();
        let mid_rect = IRect::new(rect.x0, rect.y0 - r as i32, rect.x1, rect.y1 + r as i32);
        let mid = self.new_image(f, mid_rect);
        self.run(f, "k_blur_h", &mid, [Some(src), None, None, None], &[sigma, r, edges], None);
        let out = self.new_image(f, rect);
        self.run(f, "k_blur_v", &out, [Some(&mid), None, None, None], &[sigma, r, edges], None);
        out
    }

    fn masked_mix(&mut self, f: &mut Frame, node: &Node, rect: IRect, orig: &GpuImage, processed: GpuImage) -> GpuImage {
        match self.input(f, node, 1, rect, SocketType::Mask) {
            Some(mask) => {
                let out = self.new_image(f, rect);
                self.run(f, "k_mix", &out, [Some(orig), Some(&processed), None, Some(&mask)], &[1.0], None);
                out
            }
            None => processed,
        }
    }

    fn eval_kind(&mut self, f: &mut Frame, node: &Node, rect: IRect) -> Option<GpuImage> {
        let n = node;
        match n.kind.as_str() {
            prisme_core::graph::KIND_OUTPUT => self.input(f, n, 0, rect, SocketType::Image),
            prisme_core::graph::KIND_LAYER_STACK => {
                let exclude: HashSet<LayerId> = match n.params.get("exclude") {
                    Some(ParamValue::LayerSet(v)) => v.iter().copied().collect(),
                    _ => HashSet::new(),
                };
                let layers = f.doc.layers.clone();
                Some(self.composite(f, &layers, rect, &exclude))
            }
            "layer" => {
                let Some(ParamValue::Layer(Some(lid))) = n.params.get("layer") else { return None };
                let layer = f.doc.layer(*lid)?.clone();
                let apply_mask = n.b("apply_mask", true);
                let opacity = if n.b("apply_opacity", false) { layer.opacity } else { 1.0 };
                let content = self.layer_content(f, &layer, rect)?;
                let (mask, mdef) = if apply_mask { self.layer_mask(f, &layer, rect) } else { (None, 1.0) };
                let out = self.new_image(f, rect);
                let params = [0.0, opacity, if content.straight { 1.0 } else { 0.0 }, mdef, mask.is_some() as i32 as f32, 0.0, 0.0, 0.0];
                self.run(f, "k_layer", &out, [None, Some(&content), mask.as_ref(), None], &params, None);
                Some(out)
            }
            "mask" | "selection" => {
                let (grid, key, default) = if n.kind == "selection" {
                    match &f.doc.selection {
                        Some(s) => (s.clone(), UploadKey::Selection, 0.0),
                        None => {
                            // No selection == everything selected.
                            let out = self.new_image(f, rect);
                            self.run(f, "k_fill", &out, [None; 4], &[1.0, 1.0, 1.0, 1.0], None);
                            return Some(out);
                        }
                    }
                } else {
                    let Some(ParamValue::Mask(Some(mid))) = n.params.get("mask") else { return None };
                    let m = f.doc.saved_mask(*mid)?;
                    (m.mask.clone(), UploadKey::SavedMask(mid.0), 0.0)
                };
                let up = self.upload(f, key, &grid, rect, default);
                let out = self.new_image(f, rect);
                self.run(f, "k_mask_copy", &out, [up.as_ref(), None, None, None], &[default], None);
                Some(out)
            }
            "color" => {
                let c = n.color("color", [0.5, 0.5, 0.5, 1.0]);
                let out = self.new_image(f, rect);
                self.run(f, "k_fill", &out, [None; 4], &[c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]], None);
                Some(out)
            }
            "gradient" => {
                let g = match n.params.get("gradient") {
                    Some(ParamValue::Gradient(g)) => g.clone(),
                    _ => Gradient::default(),
                };
                let lut = self.gradient_lut(&g);
                let s = n.vec2("start", [0.0, 0.0]);
                let e = n.vec2("end", [512.0, 0.0]);
                let out = self.new_image(f, rect);
                let params = [s[0], s[1], e[0], e[1], n.choice("shape", 0) as f32, 0.0, 0.0, 1.0];
                self.run(f, "k_gradient", &out, [None; 4], &params, Some(&lut));
                Some(out)
            }
            "levels" => {
                let mut p = vec![];
                for ch in LEVEL_CHANNELS {
                    for k in ["in_black", "in_white", "gamma", "out_black", "out_white"] {
                        let d = match k {
                            "in_white" | "gamma" | "out_white" => 1.0,
                            _ => 0.0,
                        };
                        p.push(n.f(&format!("{ch}.{k}"), d));
                    }
                }
                self.adjust(f, n, rect, "k_levels", &p, None)
            }
            "curves" => {
                let curves: Vec<Vec<[f32; 2]>> = CURVE_CHANNELS
                    .iter()
                    .map(|ch| match n.params.get(&format!("curve.{ch}")) {
                        Some(ParamValue::Curve(c)) => c.clone(),
                        _ => vec![[0.0, 0.0], [1.0, 1.0]],
                    })
                    .collect();
                let l_active = curves[4] != vec![[0.0, 0.0], [1.0, 1.0]];
                let lut = self.lut(hash_bytes(&curves), || {
                    let mut v = Vec::with_capacity(2048);
                    for i in 0..1024 {
                        let x = i as f32 / 1023.0;
                        v.push([eval_curve(&curves[0], x), eval_curve(&curves[1], x), eval_curve(&curves[2], x), eval_curve(&curves[3], x)]);
                    }
                    for i in 0..1024 {
                        let x = i as f32 / 1023.0;
                        v.push([eval_curve(&curves[4], x), 0.0, 0.0, 0.0]);
                    }
                    v
                });
                self.adjust(f, n, rect, "k_curves", &[l_active as i32 as f32], Some(&lut))
            }
            "exposure" => self.adjust(f, n, rect, "k_exposure", &[n.f("exposure", 0.0), n.f("offset", 0.0), n.f("gamma", 1.0)], None),
            "brightness_contrast" => self.adjust(
                f,
                n,
                rect,
                "k_brightness_contrast",
                &[n.f("brightness", 0.0), n.f("contrast", 0.0), n.f("pivot", 0.5)],
                None,
            ),
            "shadows_highlights" => {
                let pad = self.pad(f, n);
                let big = rect.expand(pad);
                let src = self.input(f, n, 0, big, SocketType::Image)?;
                let l = self.new_image(f, big);
                self.run(f, "k_oklab_l", &l, [Some(&src), None, None, None], &[], None);
                let sigma = n.f("radius", 30.0) * f.g.level_scale;
                let blurred = self.gaussian(f, &l, rect, sigma, n.choice("edges", 0) as f32);
                let mask = self.input(f, n, 1, rect, SocketType::Mask);
                let out = self.new_image(f, rect);
                self.run(
                    f,
                    "k_sh_combine",
                    &out,
                    [Some(&src), Some(&blurred), None, mask.as_ref()],
                    &[n.f("shadows", 0.3), n.f("highlights", 0.0)],
                    None,
                );
                Some(out)
            }
            "color_wheels" => {
                let w = |k| {
                    let v = n.vec3(k, [0.0; 3]);
                    [v[0], v[1], v[2], 0.0]
                };
                let mut p: Vec<f32> = [w("lift"), w("gamma"), w("gain"), w("offset")].concat();
                p.push(n.f("saturation", 1.0));
                self.adjust(f, n, rect, "k_color_wheels", &p, None)
            }
            "hsl" => {
                let c = n.color("colorize_color", [0.8, 0.5, 0.3, 1.0]);
                let p = [
                    n.f("hue", 0.0),
                    n.f("saturation", 0.0),
                    n.f("lightness", 0.0),
                    n.b("colorize", false) as i32 as f32,
                    c[0],
                    c[1],
                    c[2],
                    c[3],
                ];
                self.adjust(f, n, rect, "k_hsl", &p, None)
            }
            "vibrance" => self.adjust(f, n, rect, "k_vibrance", &[n.f("vibrance", 0.3), n.f("saturation", 0.0)], None),
            "white_balance" => self.adjust(f, n, rect, "k_white_balance", &[n.f("temperature", 0.0), n.f("tint", 0.0)], None),
            "color_balance" => {
                let w = |k| {
                    let v = n.vec3(k, [0.0; 3]);
                    [v[0], v[1], v[2], 0.0]
                };
                let mut p: Vec<f32> = [w("shadows"), w("midtones"), w("highlights")].concat();
                p.push(n.b("preserve_luminosity", true) as i32 as f32);
                self.adjust(f, n, rect, "k_color_balance", &p, None)
            }
            "channel_mixer" => {
                let p = [
                    n.f("rr", 1.0),
                    n.f("rg", 0.0),
                    n.f("rb", 0.0),
                    0.0,
                    n.f("gr", 0.0),
                    n.f("gg", 1.0),
                    n.f("gb", 0.0),
                    0.0,
                    n.f("br", 0.0),
                    n.f("bg", 0.0),
                    n.f("bb", 1.0),
                    0.0,
                    n.b("monochrome", false) as i32 as f32,
                ];
                self.adjust(f, n, rect, "k_channel_mixer", &p, None)
            }
            "selective_color" => {
                const CENTRES: [f32; 6] = [0.51, 1.70, 2.48, 3.38, 4.57, 5.65];
                let c = CENTRES[(n.choice("range", 0) as usize).min(5)];
                let p = [c, n.f("width", 0.35), n.f("hue", 0.0), n.f("saturation", 0.0), n.f("lightness", 0.0)];
                self.adjust(f, n, rect, "k_selective_color", &p, None)
            }
            "black_white" => {
                let t = n.color("tint_color", [0.9, 0.75, 0.55, 1.0]);
                let p = [
                    n.f("reds", 0.4),
                    n.f("yellows", 0.6),
                    n.f("greens", 0.4),
                    n.f("cyans", 0.6),
                    n.f("blues", 0.2),
                    n.f("magentas", 0.8),
                    n.b("tint", false) as i32 as f32,
                    0.0,
                    t[0],
                    t[1],
                    t[2],
                    t[3],
                ];
                self.adjust(f, n, rect, "k_black_white", &p, None)
            }
            "gradient_map" => {
                let g = match n.params.get("gradient") {
                    Some(ParamValue::Gradient(g)) => g.clone(),
                    _ => Gradient::default(),
                };
                let lut = self.gradient_lut(&g);
                self.adjust(f, n, rect, "k_gradient_map", &[], Some(&lut))
            }
            "lut" => {
                let text = match n.params.get("cube") {
                    Some(ParamValue::Text(t)) => t.clone(),
                    _ => String::new(),
                };
                let Some((size, data)) = prisme_graph::parse_cube(&text) else {
                    return self.input(f, n, 0, rect, SocketType::Image);
                };
                let lut = self.lut(hash_bytes(&text), || data.iter().map(|c| [c[0], c[1], c[2], 1.0]).collect());
                self.adjust(f, n, rect, "k_lut3d", &[size as f32, n.f("strength", 1.0)], Some(&lut))
            }
            "invert" => self.adjust(f, n, rect, "k_invert", &[], None),
            "threshold" => self.adjust(f, n, rect, "k_threshold", &[n.f("level", 0.5)], None),
            "posterize" => self.adjust(f, n, rect, "k_posterize", &[n.f("levels", 6.0)], None),
            "gaussian_blur" | "box_blur" => {
                let pad = self.pad(f, n);
                let src = self.input(f, n, 0, rect.expand(pad), SocketType::Image)?;
                let radius = n.f("radius", 4.0) * f.g.level_scale;
                let blurred = if n.kind == "gaussian_blur" {
                    self.gaussian(f, &src, rect, radius, n.choice("edges", 0) as f32)
                } else {
                    let r = radius.round().max(0.0);
                    let mid = self.new_image(f, IRect::new(rect.x0, rect.y0 - r as i32, rect.x1, rect.y1 + r as i32));
                    self.run(f, "k_box_h", &mid, [Some(&src), None, None, None], &[r, n.choice("edges", 0) as f32], None);
                    let out = self.new_image(f, rect);
                    self.run(f, "k_box_v", &out, [Some(&mid), None, None, None], &[r, n.choice("edges", 0) as f32], None);
                    out
                };
                Some(self.masked_mix(f, n, rect, &src, blurred))
            }
            "motion_blur" => {
                let pad = self.pad(f, n);
                let src = self.input(f, n, 0, rect.expand(pad), SocketType::Image)?;
                let mask = self.input(f, n, 1, rect, SocketType::Mask);
                let a = n.f("angle", 0.0).to_radians();
                let out = self.new_image(f, rect);
                let p = [a.cos(), a.sin(), n.f("distance", 20.0) * f.g.level_scale, n.choice("edges", 0) as f32];
                self.run(f, "k_motion", &out, [Some(&src), None, None, mask.as_ref()], &p, None);
                Some(out)
            }
            "unsharp_mask" | "high_pass" => {
                let pad = self.pad(f, n);
                let src = self.input(f, n, 0, rect.expand(pad), SocketType::Image)?;
                let sigma = n.f("radius", if n.kind == "high_pass" { 10.0 } else { 2.0 }) * f.g.level_scale;
                let blurred = self.gaussian(f, &src, rect, sigma, n.choice("edges", 0) as f32);
                let mask = self.input(f, n, 1, rect, SocketType::Mask);
                let out = self.new_image(f, rect);
                if n.kind == "unsharp_mask" {
                    let p = [n.f("amount", 1.0), n.f("threshold", 0.0)];
                    self.run(f, "k_unsharp", &out, [Some(&src), Some(&blurred), None, mask.as_ref()], &p, None);
                } else {
                    self.run(f, "k_high_pass", &out, [Some(&src), Some(&blurred), None, mask.as_ref()], &[], None);
                }
                Some(out)
            }
            "noise" => self.adjust(
                f,
                n,
                rect,
                "k_noise",
                &[n.f("amount", 0.1), n.b("mono", true) as i32 as f32, n.f("seed", 1.0)],
                None,
            ),
            "vignette" => self.adjust(
                f,
                n,
                rect,
                "k_vignette",
                &[n.f("amount", -0.5), n.f("midpoint", 0.5), n.f("roundness", 1.0), n.f("feather", 0.5)],
                None,
            ),
            "blend" => {
                let base = self.input(f, n, 0, rect, SocketType::Image);
                let top = self.input(f, n, 1, rect, SocketType::Image);
                let mask = self.input(f, n, 2, rect, SocketType::Mask);
                let out = self.new_image(f, rect);
                let p = [n.choice("mode", 0) as f32, n.f("opacity", 1.0), 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
                self.run(f, "k_blend", &out, [base.as_ref(), top.as_ref(), None, mask.as_ref()], &p, None);
                Some(out)
            }
            "mix" => {
                let a = self.input(f, n, 0, rect, SocketType::Image);
                let b = self.input(f, n, 1, rect, SocketType::Image);
                let mask = self.input(f, n, 2, rect, SocketType::Mask);
                let out = self.new_image(f, rect);
                self.run(f, "k_mix", &out, [a.as_ref(), b.as_ref(), None, mask.as_ref()], &[n.f("factor", 1.0)], None);
                Some(out)
            }
            "transform" => {
                let t = n.vec2("translate", [0.0, 0.0]);
                let rot = n.f("rotate", 0.0).to_radians();
                let sc = n.f("scale", 1.0).max(1e-3);
                let s = f.g.level_scale;
                let c = [f.g.doc_w * 0.5 * s, f.g.doc_h * 0.5 * s];
                use prisme_core::geom::Affine2;
                let fwd = Affine2::translate(c[0] + t[0] * s, c[1] + t[1] * s)
                    .mul(&Affine2::rotate(rot))
                    .mul(&Affine2::scale(sc, sc))
                    .mul(&Affine2::translate(-c[0], -c[1]));
                let inv = fwd.inverse()?;
                // Input region = inverse image of the output rect.
                let corners = [
                    [rect.x0 as f32, rect.y0 as f32],
                    [rect.x1 as f32, rect.y0 as f32],
                    [rect.x0 as f32, rect.y1 as f32],
                    [rect.x1 as f32, rect.y1 as f32],
                ]
                .map(|p| inv.apply(p));
                let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
                for p in corners {
                    x0 = x0.min(p[0]);
                    y0 = y0.min(p[1]);
                    x1 = x1.max(p[0]);
                    y1 = y1.max(p[1]);
                }
                let in_rect = IRect::from_f32(x0, y0, x1, y1).expand(2);
                // Fixed: nothing exists past the page. Infinite: rotation/scale centre is the
                // nominal frame's (stable while painting, unlike the content bounds).
                // Strong zoom-out on an infinite canvas: bound the input by the content.
                let in_rect = if f.g.infinite && in_rect.width().max(in_rect.height()) > 8192 {
                    let e = f.doc.content_bounds(false, None).union(&f.doc.bounds());
                    in_rect.intersect(&IRect::from_f32(e.x0 as f32 * s, e.y0 as f32 * s, e.x1 as f32 * s, e.y1 as f32 * s).expand(2))
                } else {
                    in_rect
                };
                let in_rect = if f.g.infinite { in_rect } else { in_rect.intersect(&IRect::new(0, 0, (f.g.doc_w * s).ceil() as i32, (f.g.doc_h * s).ceil() as i32).expand(2)) };
                let out = self.new_image(f, rect);
                let src = if in_rect.is_empty() { None } else { self.input(f, n, 0, in_rect, SocketType::Image) };
                let m = inv.m;
                self.run(f, "k_transform", &out, [src.as_ref(), None, None, None], &[m[0], m[1], m[2], 0.0, m[3], m[4], m[5], 0.0], None);
                Some(out)
            }
            "crop" => {
                let src = self.input(f, n, 0, rect, SocketType::Image)?;
                let a = n.vec2("min", [0.0, 0.0]);
                let b = n.vec2("max", [512.0, 512.0]);
                let out = self.new_image(f, rect);
                self.run(f, "k_crop", &out, [Some(&src), None, None, None], &[a[0], a[1], b[0], b[1]], None);
                Some(out)
            }
            "channel" => {
                let src = self.input(f, n, 0, rect, SocketType::Image)?;
                let out = self.new_image(f, rect);
                self.run(f, "k_channel", &out, [Some(&src), None, None, None], &[n.choice("channel", 4) as f32], None);
                Some(out)
            }
            "combine" => {
                let r = self.input(f, n, 0, rect, SocketType::Mask);
                let g = self.input(f, n, 1, rect, SocketType::Mask);
                let b = self.input(f, n, 2, rect, SocketType::Mask);
                let a = self.input(f, n, 3, rect, SocketType::Mask);
                let out = self.new_image(f, rect);
                self.run(f, "k_combine", &out, [r.as_ref(), g.as_ref(), b.as_ref(), a.as_ref()], &[], None);
                Some(out)
            }
            "mask_invert" | "mask_levels" => {
                let m = self.input(f, n, 0, rect, SocketType::Mask);
                let out = self.new_image(f, rect);
                if n.kind == "mask_invert" {
                    self.run(f, "k_mask_invert", &out, [m.as_ref(), None, None, None], &[], None);
                } else {
                    self.run(f, "k_mask_levels", &out, [m.as_ref(), None, None, None], &[n.f("low", 0.0), n.f("high", 1.0)], None);
                }
                Some(out)
            }
            "math" => {
                let src = self.input(f, n, 0, rect, SocketType::Image)?;
                let out = self.new_image(f, rect);
                self.run(f, "k_math", &out, [Some(&src), None, None, None], &[n.choice("op", 1) as f32, n.f("value", 1.0)], None);
                Some(out)
            }
            "set_alpha" => self.adjust(f, n, rect, "k_set_alpha", &[], None),
            other => {
                log::warn!("no kernel for node kind {other}");
                self.input(f, n, 0, rect, SocketType::Image)
            }
        }
    }

    // ------------------------------------------------------------------ layers

    fn upload(&mut self, f: &mut Frame, key: UploadKey, grid: &TileGrid, rect: IRect, default: f32) -> Option<GpuImage> {
        let f32 = f.format == EvalFormat::F32;
        let img = self.uploads.get(&self.ctx.device, &mut self.pool, &mut self.kernels, &mut f.enc, f.doc.uid, key, grid, f.mip, rect, f32, default);
        if let Some((base, p)) = f.stroke.and_then(|s| s.xform_args(key, 1 << f.mip)) {
            let mut out = self.new_image(f, rect);
            self.run(f, "k_xform", &out, [img.as_ref(), Some(&base), None, None], &p, None);
            out.straight = true;
            return Some(out);
        }
        match f.stroke {
            Some(s) if s.key == key && s.is_liquify() => self.liquify_view(f, s, img, rect, default),
            Some(s) if s.key == key => self.stroke_view(f, s, img, rect, default),
            _ => img,
        }
    }

    /// The upload warped by a Liquify session inside its working area.
    fn liquify_view(&mut self, f: &mut Frame, s: &GpuStroke, regular: Option<GpuImage>, rect: IRect, default: f32) -> Option<GpuImage> {
        let k = 1i32 << f.mip;
        let Some(([base, field], sel, p)) = s.liquify_args(k, false, default) else { return regular };
        let lr = s.liquify_rect();
        let level = IRect::new(lr.x0.div_euclid(k), lr.y0.div_euclid(k), (lr.x1 + k - 1).div_euclid(k), (lr.y1 + k - 1).div_euclid(k));
        if level.intersect(&rect).is_empty() {
            return regular;
        }
        let mut out = self.new_image(f, rect);
        self.run(f, "k_liquify", &out, [regular.as_ref(), Some(&base), Some(&field), sel.as_ref()], &p, None);
        out.straight = true;
        Some(out)
    }

    /// The upload with the live stroke tiles spliced in (block-averaged at mip > 0). Each
    /// tile's footprint at this level is rendered on its own and copied in. Past mip 8 a level
    /// pixel spans several tiles: the regular upload shows until the stroke lands.
    fn stroke_view(&mut self, f: &mut Frame, s: &GpuStroke, regular: Option<GpuImage>, rect: IRect, default: f32) -> Option<GpuImage> {
        let k = 1i32 << f.mip;
        if k > TILE_SIZE as i32 {
            return regular;
        }
        let hits: Vec<(IRect, GpuImage)> = s
            .live_tiles()
            .filter_map(|(c, t)| {
                let r = c.rect();
                let level = IRect::new(r.x0 / k, r.y0 / k, r.x1 / k, r.y1 / k).intersect(&rect);
                (!level.is_empty()).then(|| (level, t.clone()))
            })
            .collect();
        if hits.is_empty() {
            return regular;
        }
        let _span = perf::span("stroke view");
        perf::count("stroke view tiles", hits.len() as u64);
        let p = [k as f32, s.channels() as f32, default];
        let mut out = self.new_image(f, rect);
        self.run(f, "k_stroke_view", &out, [regular.as_ref(), None, None, None], &p, None);
        for (level, tile) in hits {
            let part = self.new_image(f, level);
            self.run(f, "k_stroke_view", &part, [regular.as_ref(), Some(&tile), None, None], &p, None);
            crate::stroke::copy(&mut f.enc, &part, &out, level);
        }
        out.straight = true;
        Some(out)
    }

    fn layer_mask(&mut self, f: &mut Frame, layer: &Layer, rect: IRect) -> (Option<GpuImage>, f32) {
        match &layer.mask {
            Some(m) if m.enabled => {
                let def = if m.default_visible { 1.0 } else { 0.0 };
                (self.upload(f, UploadKey::LayerMask(layer.id.0), &m.grid.clone(), rect, def), def)
            }
            _ => (None, 1.0),
        }
    }

    /// The layer's own pixels over `rect` (no opacity/mask applied).
    fn layer_content(&mut self, f: &mut Frame, layer: &Layer, rect: IRect) -> Option<GpuImage> {
        match &layer.kind {
            LayerKind::Raster(g) => self.upload(f, UploadKey::Content(layer.id.0), &g.clone(), rect, 0.0),
            LayerKind::Group(g) => {
                let children = g.children.clone();
                Some(self.composite(f, &children, rect, &HashSet::new()))
            }
            LayerKind::Fill(fill) => {
                let out = self.new_image(f, rect);
                match &fill.content {
                    FillContent::Solid(c) => {
                        self.run(f, "k_fill", &out, [None; 4], &[c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]], None)
                    }
                    FillContent::Gradient(g, geo) => {
                        let lut = self.gradient_lut(g);
                        let params = [
                            geo.start[0],
                            geo.start[1],
                            geo.end[0],
                            geo.end[1],
                            geo.kind as u32 as f32,
                            geo.repeat as u32 as f32,
                            geo.reverse as i32 as f32,
                            geo.dither as i32 as f32,
                        ];
                        self.run(f, "k_gradient", &out, [None; 4], &params, Some(&lut));
                    }
                }
                Some(out)
            }
            LayerKind::Text(_) | LayerKind::Vector(_) => {
                let hash = *f.layer_hashes.get(&layer.id)?;
                let key = (f.doc.uid, layer.id, f.mip);
                let grid = match self.rasters.get_mut(&key) {
                    Some((h, g, used)) if *h == hash => {
                        *used = self.frame;
                        g.clone()
                    }
                    _ => {
                        let r = self.rasterizer.as_ref()?.rasterize(layer, f.doc, f.mip)?;
                        let g = Arc::new(r);
                        self.rasters.insert(key, (hash, g.clone(), self.frame));
                        g
                    }
                };
                // The rasterized grid is already in level space: upload it as mip 0.
                let f32 = f.format == EvalFormat::F32;
                self.uploads.get(
                    &self.ctx.device,
                    &mut self.pool,
                    &mut self.kernels,
                    &mut f.enc,
                    f.doc.uid,
                    UploadKey::Rasterized(layer.id.0, f.mip),
                    &grid,
                    0,
                    rect,
                    f32,
                    0.0,
                )
            }
        }
    }

    /// Composite a list of layers (bottom to top) over transparency.
    fn composite(&mut self, f: &mut Frame, layers: &[Layer], rect: IRect, exclude: &HashSet<LayerId>) -> GpuImage {
        let mut acc = self.new_image(f, rect);
        self.run(f, "k_fill", &acc, [None; 4], &[0.0; 4], None);
        for layer in layers {
            if !layer.visible || exclude.contains(&layer.id) || layer.opacity <= 0.0 {
                continue;
            }
            let Some(content) = self.layer_content(f, layer, rect) else { continue };
            let (mask, mdef) = self.layer_mask(f, layer, rect);
            let out = self.new_image(f, rect);
            let params = [
                layer.blend as u32 as f32,
                layer.opacity,
                if content.straight { 1.0 } else { 0.0 },
                mdef,
                mask.is_some() as i32 as f32,
                0.0,
                0.0,
                0.0,
            ];
            self.run(f, "k_blend", &out, [Some(&acc), Some(&content), mask.as_ref(), None], &params, None);
            acc = out;
        }
        acc
    }

    pub fn pool_bytes(&self) -> u64 {
        self.pool.total_bytes()
    }
}
