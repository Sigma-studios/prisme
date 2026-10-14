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
use crate::levels::MipCache;
use crate::upload::{UploadCache, UploadKey};
use prisme_core::graph::{Node, ParamValue};
use prisme_core::layer::{FillContent, Layer, LayerKind};
use prisme_core::{Document, Gradient, IRect, LayerId, NodeId, TileGrid};
use prisme_graph::{eval_curve, registry, SocketType, CURVE_CHANNELS, LEVEL_CHANNELS};
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::Arc;

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
    pool: TexturePool,
    uploads: UploadCache,
    mips: MipCache,
    rasters: HashMap<(LayerId, u32), (u64, Arc<TileGrid>)>,
    results: HashMap<u64, CachedResult>,
    luts: HashMap<u64, Arc<wgpu::Buffer>>,
    pub rasterizer: Option<Arc<dyn LayerRasterizer>>,
    frame: u64,
    /// Statistics of the last evaluation.
    pub last_stats: EvalStats,
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
}

impl Engine {
    pub fn new(ctx: GpuContext) -> Self {
        let kernels = Kernels::new(&ctx.device);
        let pool = TexturePool::new(ctx.device.clone());
        Engine {
            ctx,
            kernels,
            pool,
            uploads: UploadCache::default(),
            mips: MipCache::default(),
            rasters: HashMap::new(),
            results: HashMap::new(),
            luts: HashMap::new(),
            rasterizer: None,
            frame: 0,
            last_stats: EvalStats::default(),
        }
    }

    /// Drop all cached results (e.g. after a device-level change).
    pub fn clear_caches(&mut self) {
        self.results.clear();
        self.rasters.clear();
        self.luts.clear();
    }

    pub fn evaluate(&mut self, req: &EvalRequest) -> Option<GpuImage> {
        if req.rect.is_empty() {
            return None;
        }
        self.uploads.begin_frame();
        self.frame += 1;
        let level_scale = 1.0 / (1u32 << req.mip) as f32;
        let enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("eval") });
        let mut f = Frame {
            doc: req.doc,
            mip: req.mip,
            format: req.format,
            g: Globals { level_scale, mip: req.mip, doc_w: req.doc.info.width as f32, doc_h: req.doc.info.height as f32 },
            enc,
            layer_hashes: HashMap::new(),
            node_hashes: HashMap::new(),
            stats: EvalStats::default(),
        };
        fn hash_layers(v: &[Layer], out: &mut HashMap<LayerId, u64>) -> u64 {
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
                    LayerKind::Group(g) => hash_layers(&g.children, out).hash(&mut h),
                    LayerKind::Text(t) => hash_bytes(t).hash(&mut h),
                    LayerKind::Vector(v) => hash_bytes(v).hash(&mut h),
                    LayerKind::Fill(fl) => hash_bytes(fl).hash(&mut h),
                }
                let v = h.finish();
                out.insert(l.id, v);
                v.hash(&mut all);
            }
            all.finish()
        }
        hash_layers(&req.doc.layers, &mut f.layer_hashes);

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
        self.last_stats = f.stats;
        self.ctx.queue.submit([f.enc.finish()]);
        self.uploads.end_frame();
        self.mips.end_frame(4096);
        let fr = self.frame;
        self.results.retain(|_, r| fr - r.last_used < 6);
        self.pool.trim(256 << 20);
        Some(result)
    }

    /// Evaluate a node at a small size and convert to sRGB8 (node editor thumbnails).
    pub fn thumbnail(&mut self, doc: &Document, node: NodeId, max_side: u32) -> Option<GpuImage> {
        let side = doc.info.width.max(doc.info.height).max(1);
        let mut mip = 0;
        while (side >> mip) > max_side * 2 && mip < 12 {
            mip += 1;
        }
        let s = 1 << mip;
        let rect = IRect::new(0, 0, (doc.info.width as i32 + s - 1) / s, (doc.info.height as i32 + s - 1) / s);
        let img = self.evaluate(&EvalRequest { doc, rect, mip, format: EvalFormat::F16, source: RenderSource::Node(node) })?;
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        let out = self.pool.image(rect, wgpu::TextureFormat::Rgba8Unorm);
        let g = Globals { level_scale: 1.0 / s as f32, mip, doc_w: doc.info.width as f32, doc_h: doc.info.height as f32 };
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
                let doc_level = IRect::new(0, 0, (f.g.doc_w * s).ceil() as i32, (f.g.doc_h * s).ceil() as i32).expand(2);
                let in_rect = IRect::from_f32(x0, y0, x1, y1).expand(2).intersect(&doc_level);
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
        self.uploads.get(&self.ctx.device, &self.ctx.queue, &mut self.mips, key, grid, f.mip, rect, f32, default)
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
                let key = (layer.id, f.mip);
                let grid = match self.rasters.get(&key) {
                    Some((h, g)) if *h == hash => g.clone(),
                    _ => {
                        let r = self.rasterizer.as_ref()?.rasterize(layer, f.doc, f.mip)?;
                        let g = Arc::new(r);
                        self.rasters.insert(key, (hash, g.clone()));
                        g
                    }
                };
                // The rasterized grid is already in level space: upload it as mip 0.
                let f32 = f.format == EvalFormat::F32;
                self.uploads.get(
                    &self.ctx.device,
                    &self.ctx.queue,
                    &mut self.mips,
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
