//! Resident GPU tiles: every tile the engine has shown, at every mip level it was shown at.
//!
//! Tiles live in texture-array pages (one 256² layer each) keyed by content: a level-0 tile by
//! its id (bumped on every edit), a level-`m` tile by a hash of its four children's keys
//! ([`Pyramid`]). Level 0 is uploaded from the CPU once per tile version; coarser levels are
//! built on the GPU from their children (`k_downsample`). Views pull tiles from here with
//! GPU copies ([`crate::upload`]), so panning, zooming and changing mip level never re-send
//! pixels, and nothing is downsampled on the CPU. Least recently used tiles are evicted past
//! a VRAM budget (a level-0 tile comes back from the CPU, a coarser one is rebuilt).

use crate::image::{GpuImage, PooledTex};
use crate::kernels::{Globals, Kernels};
use prisme_core::tile::{TileBuf, TileCoord, TileGrid, TILE_PIXELS, TILE_SIZE};
use prisme_core::IRect;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::Arc;

const TS: i32 = TILE_SIZE as i32;
/// Layers of a page's first texture (later ones double, up to the device's array limit).
const FIRST_PAGE_LAYERS: u32 = 16;
const MAX_PAGE_LAYERS: u32 = 256;
const G: Globals = Globals { level_scale: 1.0, mip: 0, doc_w: 0.0, doc_h: 0.0, infinite: false };

/// Storage class of a tile on the GPU (matches the views it's copied into).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    /// Single channel, f32 (masks, selections).
    Mask,
    Rgba16,
    Rgba32,
}

impl Class {
    pub fn of(channels: u8, f32: bool) -> Class {
        match (channels, f32) {
            (1, _) => Class::Mask,
            (_, false) => Class::Rgba16,
            (_, true) => Class::Rgba32,
        }
    }

    pub fn format(self) -> wgpu::TextureFormat {
        match self {
            Class::Mask => wgpu::TextureFormat::R32Float,
            Class::Rgba16 => wgpu::TextureFormat::Rgba16Float,
            Class::Rgba32 => wgpu::TextureFormat::Rgba32Float,
        }
    }

    fn tile_bytes(self) -> u64 {
        TILE_PIXELS as u64 * self.format().block_copy_size(None).unwrap_or(16) as u64
    }

    /// A tile's pixels in this class's texel layout (borrowed when the formats match).
    fn bytes(self, t: &prisme_core::tile::Tile) -> Cow<'_, [u8]> {
        match (t.buf(), self) {
            (TileBuf::F16(v), Class::Rgba16) => Cow::Borrowed(bytemuck::cast_slice(v.as_slice())),
            (TileBuf::F32(v), Class::Rgba16) => Cow::Owned(bytemuck::cast_slice(&v.iter().map(|&x| half::f16::from_f32(x)).collect::<Vec<_>>()).to_vec()),
            (TileBuf::F32(v), _) => Cow::Borrowed(bytemuck::cast_slice(v.as_slice())),
            (TileBuf::F16(v), _) => Cow::Owned(bytemuck::cast_slice(&v.iter().map(|x| x.to_f32()).collect::<Vec<_>>()).to_vec()),
        }
    }
}

fn hasher() -> ahash::AHasher {
    ahash::RandomState::with_seeds(0x7469_6c65, 0x7079_7261, 0x6d69_6473, 0x5052_4953).build_hasher()
}

/// Content keys of a grid's tiles at every level built so far, kept in sync incrementally:
/// [`Pyramid::sync`] diffs the grid's tile ids and only rehashes the ancestors of changed
/// tiles. Key 0 = no tile (all descendants missing).
#[derive(Default)]
pub struct Pyramid {
    leaves: HashMap<TileCoord, u64>,
    /// `levels[m - 1]`: level `m`.
    levels: Vec<HashMap<TileCoord, u64>>,
    /// Content bounds per level (level px), cached until the next change.
    bounds: Vec<Option<IRect>>,
    pub last_used: u64,
}

fn parent(c: TileCoord) -> TileCoord {
    TileCoord::new(c.x.div_euclid(2), c.y.div_euclid(2))
}

fn children(c: TileCoord) -> [TileCoord; 4] {
    [TileCoord::new(c.x * 2, c.y * 2), TileCoord::new(c.x * 2 + 1, c.y * 2), TileCoord::new(c.x * 2, c.y * 2 + 1), TileCoord::new(c.x * 2 + 1, c.y * 2 + 1)]
}

impl Pyramid {
    pub fn key(&self, mip: u32, c: TileCoord) -> u64 {
        let m = if mip == 0 { &self.leaves } else { &self.levels[mip as usize - 1] };
        m.get(&c).copied().unwrap_or(0)
    }

    fn child_keys(&self, mip: u32, c: TileCoord) -> [u64; 4] {
        children(c).map(|k| self.key(mip - 1, k))
    }

    fn level_key(mip: u32, kids: [u64; 4]) -> u64 {
        if kids == [0; 4] {
            return 0;
        }
        let mut h = hasher();
        (mip, kids).hash(&mut h);
        h.finish() | 1
    }

    fn set(&mut self, mip: u32, c: TileCoord, key: u64) {
        let m = &mut self.levels[mip as usize - 1];
        if key == 0 {
            m.remove(&c);
        } else {
            m.insert(c, key);
        }
    }

    /// Bring the keys in line with `grid`, building levels up to `mip`.
    pub fn sync(&mut self, grid: &TileGrid, mip: u32) {
        let mut dirty: HashSet<TileCoord> = HashSet::new();
        for (c, t) in &grid.tiles {
            if self.leaves.get(c) != Some(&t.id()) {
                self.leaves.insert(*c, t.id());
                dirty.insert(*c);
            }
        }
        if self.leaves.len() != grid.tiles.len() {
            self.leaves.retain(|c, _| {
                let keep = grid.tiles.contains_key(c);
                if !keep {
                    dirty.insert(*c);
                }
                keep
            });
        }
        if !dirty.is_empty() {
            self.bounds.clear();
        }
        // Existing levels: rehash the changed tiles' ancestors.
        for m in 1..=self.levels.len() as u32 {
            if dirty.is_empty() {
                break;
            }
            dirty = dirty.into_iter().map(parent).collect();
            for &c in &dirty {
                let k = Self::level_key(m, self.child_keys(m, c));
                self.set(m, c, k);
            }
        }
        // New levels: built from the one below.
        while (self.levels.len() as u32) < mip {
            let m = self.levels.len() as u32 + 1;
            let below: Vec<TileCoord> = if m == 1 { self.leaves.keys().copied().collect() } else { self.levels[m as usize - 2].keys().copied().collect() };
            let parents: HashSet<TileCoord> = below.into_iter().map(parent).collect();
            self.levels.push(HashMap::with_capacity(parents.len()));
            for c in parents {
                let k = Self::level_key(m, self.child_keys(m, c));
                self.set(m, c, k);
            }
            self.bounds.clear();
        }
    }

    /// Bounds of the tiles present at `mip` (level px).
    pub fn bounds(&mut self, mip: u32) -> IRect {
        let i = mip as usize;
        if self.bounds.len() <= i {
            self.bounds.resize(i + 1, None);
        }
        *self.bounds[i].get_or_insert_with(|| {
            let m = if mip == 0 { &self.leaves } else { &self.levels[i - 1] };
            m.keys().fold(IRect::EMPTY, |acc, c| acc.union(&c.rect()))
        })
    }
}

/// One texture array of tile slots.
struct Page {
    tex: wgpu::Texture,
    /// One single-layer view per slot (kernel input/output).
    slots: Vec<GpuImage>,
    free: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    page: u32,
    layer: u32,
}

struct Resident {
    slot: Slot,
    last_used: u64,
}

/// Store key: class, level, content key, and (levels > 0) the default missing children read as.
type Key = (Class, u32, u64, u32);

pub struct TileStore {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pages: HashMap<Class, Vec<Page>>,
    resident: HashMap<Key, Resident>,
    /// Constant tiles (missing tiles of masks that reveal by default), by class and value bits.
    constants: HashMap<(Class, u32), Slot>,
    frame: u64,
    bytes: u64,
    pub budget: u64,
    max_layers: u32,
    /// Tiles uploaded / built on the GPU since creation (tests, profiling).
    pub uploads: u64,
    pub builds: u64,
}

impl TileStore {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, budget: u64) -> Self {
        let max_layers = device.limits().max_texture_array_layers.min(MAX_PAGE_LAYERS);
        Self { device, queue, pages: HashMap::new(), resident: HashMap::new(), constants: HashMap::new(), frame: 0, bytes: 0, budget, max_layers, uploads: 0, builds: 0 }
    }

    pub fn begin_frame(&mut self) {
        self.frame += 1;
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    fn image(&self, class: Class, s: Slot) -> &GpuImage {
        &self.pages[&class][s.page as usize].slots[s.layer as usize]
    }

    fn new_page(&mut self, class: Class) {
        let pages = self.pages.entry(class).or_default();
        let layers = (FIRST_PAGE_LAYERS << pages.len().min(8)).min(self.max_layers).max(1);
        let format = class.format();
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tile-page"),
            size: wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: layers },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let rect = IRect::new(0, 0, TS, TS);
        let slots = (0..layers)
            .map(|i| {
                let view = tex.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                GpuImage { tex: Arc::new(PooledTex { tex: tex.clone(), view, w: TILE_SIZE, h: TILE_SIZE, format }), rect, straight: true }
            })
            .collect();
        self.bytes += layers as u64 * class.tile_bytes();
        pages.push(Page { tex, slots, free: (0..layers).rev().collect() });
    }

    /// Drop the least recently used quarter of the tiles not used this frame.
    fn evict(&mut self) -> bool {
        let mut old: Vec<(u64, Key)> = self.resident.iter().filter(|(_, r)| r.last_used < self.frame).map(|(k, r)| (r.last_used, *k)).collect();
        if old.is_empty() {
            return false;
        }
        old.sort_unstable_by_key(|(t, _)| *t);
        let n = old.len().div_ceil(4);
        for (_, k) in old.into_iter().take(n) {
            let r = self.resident.remove(&k).unwrap();
            self.pages.get_mut(&k.0).unwrap()[r.slot.page as usize].free.push(r.slot.layer);
        }
        true
    }

    fn alloc(&mut self, class: Class) -> Slot {
        loop {
            if let Some(pages) = self.pages.get_mut(&class) {
                for (i, p) in pages.iter_mut().enumerate() {
                    if let Some(layer) = p.free.pop() {
                        return Slot { page: i as u32, layer };
                    }
                }
            }
            let next = (FIRST_PAGE_LAYERS << self.pages.get(&class).map_or(0, |p| p.len()).min(8)).min(self.max_layers) as u64 * class.tile_bytes();
            if self.bytes + next > self.budget && self.evict() {
                continue;
            }
            self.new_page(class);
        }
    }

    fn touch(&mut self, k: &Key) -> Option<Slot> {
        let frame = self.frame;
        self.resident.get_mut(k).map(|r| {
            r.last_used = frame;
            r.slot
        })
    }

    /// The slot holding tile `c` of `grid` at level `mip` (keys from `pyr`, synced up to
    /// `mip`), uploading it or building it from its children as needed. None: no tile there.
    #[allow(clippy::too_many_arguments)]
    pub fn get(&mut self, kernels: &mut Kernels, enc: &mut wgpu::CommandEncoder, class: Class, grid: &TileGrid, pyr: &Pyramid, mip: u32, c: TileCoord, default: f32) -> Option<Slot> {
        let id = pyr.key(mip, c);
        if id == 0 {
            return None;
        }
        let key: Key = (class, mip, id, if mip == 0 { 0 } else { default.to_bits() });
        if let Some(s) = self.touch(&key) {
            return Some(s);
        }
        if mip == 0 {
            let t = grid.get_tile(c)?;
            let slot = self.alloc(class);
            self.write(class, slot, &class.bytes(t));
            self.uploads += 1;
            self.resident.insert(key, Resident { slot, last_used: self.frame });
            return Some(slot);
        }
        let kids = children(c).map(|k| self.get(kernels, enc, class, grid, pyr, mip - 1, k, default));
        let slot = self.alloc(class);
        let inputs = kids.map(|k| k.map(|s| self.image(class, s).clone()));
        let channels = if class == Class::Mask { 1.0 } else { 4.0 };
        kernels.dispatch(enc, "k_downsample", self.image(class, slot), None, [inputs[0].as_ref(), inputs[1].as_ref(), inputs[2].as_ref(), inputs[3].as_ref()], &[channels, default], None, &G);
        self.builds += 1;
        self.resident.insert(key, Resident { slot, last_used: self.frame });
        Some(slot)
    }

    /// A tile filled with `value` (missing tiles of masks that reveal by default).
    pub fn constant(&mut self, class: Class, value: f32) -> Slot {
        if let Some(s) = self.constants.get(&(class, value.to_bits())) {
            return *s;
        }
        // Constants are allocated outside `resident`, so they're never evicted.
        let slot = self.alloc(class);
        let data: Vec<u8> = match class {
            Class::Rgba16 => bytemuck::cast_slice(&vec![half::f16::from_f32(value); TILE_PIXELS * 4]).to_vec(),
            Class::Rgba32 => bytemuck::cast_slice(&vec![value; TILE_PIXELS * 4]).to_vec(),
            Class::Mask => bytemuck::cast_slice(&vec![value; TILE_PIXELS]).to_vec(),
        };
        self.write(class, slot, &data);
        self.constants.insert((class, value.to_bits()), slot);
        slot
    }

    fn write(&self, class: Class, s: Slot, data: &[u8]) {
        let tex = &self.pages[&class][s.page as usize].tex;
        let bpp = class.format().block_copy_size(None).unwrap_or(16);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: s.layer }, aspect: wgpu::TextureAspect::All },
            data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(TILE_SIZE * bpp), rows_per_image: Some(TILE_SIZE) },
            wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: 1 },
        );
    }

    /// Record a copy of slot `s` into `dst` at texel `origin`.
    pub fn copy(&self, enc: &mut wgpu::CommandEncoder, class: Class, s: Slot, dst: &wgpu::Texture, origin: [u32; 2]) {
        let src = &self.pages[&class][s.page as usize].tex;
        enc.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo { texture: src, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: s.layer }, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyTextureInfo { texture: dst, mip_level: 0, origin: wgpu::Origin3d { x: origin[0], y: origin[1], z: 0 }, aspect: wgpu::TextureAspect::All },
            wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: 1 },
        );
    }
}
