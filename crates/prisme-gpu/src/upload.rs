//! Per-layer view textures, filled from the resident tile store.
//!
//! Each cached texture covers a tile-aligned level rect of one grid (a layer's pixels, a
//! mask, the selection…) at one mip level. Its tiles are GPU copies of resident tiles
//! ([`crate::tiles`]): on reuse only tiles whose content key changed are copied again, and
//! a texture that has to grow or change level is refilled the same way, so views never
//! re-send pixels from the CPU and never downsample there. The textures themselves come from
//! the engine's pool, so changing level or growing doesn't allocate either.
//!
//! Everything here is keyed by document as well: layer and mask ids restart in every document,
//! and [`UploadCache::release`] drops a closed document's textures and tiles.

use crate::image::{GpuImage, TexturePool};
use crate::kernels::Globals;
use crate::kernels::Kernels;
use crate::tiles::{Class, Pyramid, TileStore};
use prisme_core::tile::{TileCoord, TileGrid, TILE_SIZE};
use prisme_core::{DocUid, IRect};
use std::collections::HashMap;

const TS: i32 = TILE_SIZE as i32;
/// VRAM the resident tiles may hold before the least recently used are evicted.
const STORE_BUDGET: u64 = 1 << 30;
/// Tiles no view used for this many evaluations are dropped, checked every `TRIM_EVERY`.
const TILE_MAX_AGE: u64 = 2048;
const TRIM_EVERY: u64 = 256;
const G: Globals = Globals { level_scale: 1.0, mip: 0, doc_w: 0.0, doc_h: 0.0, infinite: false };

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UploadKey {
    Content(u64),
    LayerMask(u64),
    Selection,
    SavedMask(u64),
    Rasterized(u64, u32),
}

struct Entry {
    image: GpuImage,
    mip: u32,
    class: Class,
    default_value: f32,
    /// Content key of each tile the texture holds (0 = empty / default).
    tiles: HashMap<TileCoord, u64>,
    last_used: u64,
}

type Key = (DocUid, UploadKey);

pub struct UploadCache {
    entries: HashMap<Key, Entry>,
    pyramids: HashMap<Key, Pyramid>,
    pub store: TileStore,
    frame: u64,
}

fn align(r: IRect) -> IRect {
    IRect::new(r.x0.div_euclid(TS) * TS, r.y0.div_euclid(TS) * TS, (r.x1 + TS - 1).div_euclid(TS) * TS, (r.y1 + TS - 1).div_euclid(TS) * TS)
}

impl UploadCache {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self { entries: HashMap::new(), pyramids: HashMap::new(), store: TileStore::new(device, queue, STORE_BUDGET), frame: 0 }
    }

    pub fn begin_frame(&mut self) {
        self.frame += 1;
        self.store.begin_frame();
    }

    /// Evict textures unused for several evaluations (thumbnails and the canvas interleave),
    /// the tile keys of grids not shown for a while, and now and then the tiles nothing has
    /// shown for a long while (deleted layers, areas long left), compacting when that frees
    /// most of the pages. Called after the evaluation's commands are submitted.
    pub fn end_frame(&mut self) {
        let f = self.frame;
        self.entries.retain(|_, e| f - e.last_used < 6);
        self.pyramids.retain(|_, p| f - p.last_used < 600);
        if f.is_multiple_of(TRIM_EVERY) {
            self.store.trim(TILE_MAX_AGE);
        }
    }

    /// Drop everything held for `doc` (a closed document) and give the memory back.
    pub fn release(&mut self, doc: DocUid) {
        self.entries.retain(|k, _| k.0 != doc);
        self.pyramids.retain(|k, _| k.0 != doc);
        self.store.release(doc);
        self.store.compact();
    }

    /// Make the grid's tiles at `mip` over `rect` (level px) resident ahead of time, recording
    /// builds into `enc`, until `out_of_time` (checked between tiles; at least one tile is
    /// done per call). Stops early once the store is three quarters full, so it never evicts
    /// what views use. True when there's nothing left to do.
    #[allow(clippy::too_many_arguments)]
    pub fn prefetch(
        &mut self,
        kernels: &mut Kernels,
        enc: &mut wgpu::CommandEncoder,
        doc: DocUid,
        key: UploadKey,
        grid: &TileGrid,
        mip: u32,
        rect: IRect,
        f32: bool,
        default_value: f32,
        out_of_time: &dyn Fn() -> bool,
    ) -> bool {
        let class = Class::of(grid.channels, f32);
        let key = (doc, key);
        let pyr = self.pyramids.entry(key).or_default();
        pyr.last_used = self.frame;
        pyr.sync(grid, mip);
        let r = align(rect.intersect(&pyr.bounds(mip)));
        let pyr = &self.pyramids[&key];
        let mut worked = false;
        for c in TileGrid::coords_in(&r) {
            if pyr.key(mip, c) == 0 || self.store.resident(doc, class, pyr, mip, c, default_value) {
                continue;
            }
            if self.store.bytes() > self.store.budget / 4 * 3 || (worked && out_of_time()) {
                return self.store.bytes() > self.store.budget / 4 * 3;
            }
            self.store.get(kernels, enc, doc, class, grid, pyr, mip, c, default_value);
            worked = true;
        }
        true
    }

    /// Get a texture with the grid's pixels at `mip` over `need` (level rect): exact
    /// everywhere in `need` it covers (outside its rect, the grid has nothing). Beyond `need`
    /// it may hold stale pixels. Returns None when the grid has nothing in `need`.
    /// Tile copies (and builds of missing levels) are recorded into `enc`.
    #[allow(clippy::too_many_arguments)]
    pub fn get(
        &mut self,
        device: &wgpu::Device,
        pool: &mut TexturePool,
        kernels: &mut Kernels,
        enc: &mut wgpu::CommandEncoder,
        doc: DocUid,
        key: UploadKey,
        grid: &TileGrid,
        mip: u32,
        need: IRect,
        f32: bool,
        default_value: f32,
    ) -> Option<GpuImage> {
        let class = Class::of(grid.channels, f32);
        let key = (doc, key);
        let pyr = self.pyramids.entry(key).or_default();
        pyr.last_used = self.frame;
        pyr.sync(grid, mip);
        // Level-space bounds of the grid content.
        let bounds = pyr.bounds(mip);
        // For masks with a non-zero default we still need a texture even if empty.
        let want = if default_value != 0.0 { need } else { need.intersect(&bounds) };
        if want.is_empty() {
            if let Some(e) = self.entries.get_mut(&key) {
                e.last_used = self.frame;
            }
            return None;
        }
        let want = align(want);
        let reusable = self.entries.get(&key).is_some_and(|e| {
            e.mip == mip && e.class == class && e.default_value == default_value && e.image.rect.intersect(&want) == want
        });
        if !reusable {
            // Allocate with a quarter of the size as margin on each side (clipped to content)
            // so pans and zoom-out steps keep reusing it.
            let m = |v: i32| (v / 4).div_euclid(TS) * TS + TS;
            let mut rect = IRect::new(want.x0 - m(want.width()), want.y0 - m(want.height()), want.x1 + m(want.width()), want.y1 + m(want.height()));
            if default_value == 0.0 {
                rect = rect.intersect(&align(bounds));
            }
            let max = device.limits().max_texture_dimension_2d as i32;
            if rect.width() > max || rect.height() > max {
                rect = want;
            }
            if rect.width() > max || rect.height() > max {
                log::warn!("upload rect {rect:?} exceeds texture limits");
                return None;
            }
            // From the pool (reused across levels and frames, never cleared): fill it with the
            // grid's default, then only tiles with content are copied in.
            let mut image = pool.image(rect, class.format());
            image.straight = true;
            let d = default_value;
            kernels.dispatch(enc, "k_fill", &image, None, [None; 4], &[d, d, d, d], None, &G);
            let tiles = TileGrid::coords_in(&rect).map(|c| (c, 0)).collect();
            self.entries.insert(key, Entry { image, mip, class, default_value, tiles, last_used: self.frame });
        }
        let pyr = &self.pyramids[&key];
        let e = self.entries.get_mut(&key).unwrap();
        e.last_used = self.frame;
        let rect = e.image.rect;
        // Every tile a caller may read (all of `need` the texture covers, content or not: an
        // earlier request, or another grid's, may have left pixels there); the margin beyond
        // keeps what it has until a request reaches it.
        for c in TileGrid::coords_in(&align(need).intersect(&rect)) {
            let id = pyr.key(mip, c);
            if e.tiles.get(&c) == Some(&id) {
                continue;
            }
            let slot = match self.store.get(kernels, enc, doc, class, grid, pyr, mip, c, default_value) {
                Some(s) => s,
                None => self.store.constant(class, default_value),
            };
            self.store.copy(enc, class, slot, &e.image.tex.tex, [(c.x * TS - rect.x0) as u32, (c.y * TS - rect.y0) as u32]);
            e.tiles.insert(c, id);
        }
        Some(e.image.clone())
    }
}
