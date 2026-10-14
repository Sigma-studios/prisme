//! Incremental upload of tile grids into per-layer GPU textures.
//!
//! Each cached texture covers a tile-aligned level rect. On reuse only tiles whose
//! content id changed are rewritten, so painting uploads just the dirty tiles.

use crate::image::{GpuImage, PooledTex};
use crate::levels::MipCache;
use half::f16;
use prisme_core::tile::{PixelFormat, Tile, TileBuf, TileCoord, TileGrid, TILE_PIXELS, TILE_SIZE};
use prisme_core::IRect;
use std::collections::HashMap;
use std::sync::Arc;

const TS: i32 = TILE_SIZE as i32;

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
    f32: bool,
    default_value: f32,
    tiles: HashMap<TileCoord, u64>,
    last_used: u64,
}

#[derive(Default)]
pub struct UploadCache {
    entries: HashMap<UploadKey, Entry>,
    frame: u64,
    zero_f16: Vec<u8>,
    zero_f32: Vec<u8>,
}


fn texture_format(channels: u8, f32: bool) -> wgpu::TextureFormat {
    match (channels, f32) {
        (1, false) => wgpu::TextureFormat::R16Float,
        (1, true) => wgpu::TextureFormat::R32Float,
        (_, false) => wgpu::TextureFormat::Rgba16Float,
        (_, true) => wgpu::TextureFormat::Rgba32Float,
    }
}

fn tile_bytes(t: &Tile, f32: bool) -> std::borrow::Cow<'_, [u8]> {
    match (t.buf(), f32) {
        (TileBuf::F16(v), false) => std::borrow::Cow::Borrowed(bytemuck::cast_slice(v.as_slice())),
        (TileBuf::F32(v), true) => std::borrow::Cow::Borrowed(bytemuck::cast_slice(v.as_slice())),
        (TileBuf::F32(v), false) => {
            let h: Vec<f16> = v.iter().map(|&x| f16::from_f32(x)).collect();
            std::borrow::Cow::Owned(bytemuck::cast_slice(&h).to_vec())
        }
        (TileBuf::F16(v), true) => {
            let f: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();
            std::borrow::Cow::Owned(bytemuck::cast_slice(&f).to_vec())
        }
    }
}

impl UploadCache {
    pub fn begin_frame(&mut self) {
        self.frame += 1;
    }

    /// Evict entries unused for several evaluations (thumbnails and the canvas interleave).
    pub fn end_frame(&mut self) {
        let f = self.frame;
        self.entries.retain(|_, e| f - e.last_used < 6);
    }

    fn filled(&mut self, channels: u8, f32: bool, value: f32) -> Vec<u8> {
        let n = TILE_PIXELS * channels as usize;
        if value == 0.0 {
            let z = if f32 { &mut self.zero_f32 } else { &mut self.zero_f16 };
            let bytes = n * if f32 { 4 } else { 2 };
            if z.len() < bytes {
                z.resize(bytes, 0);
            }
            return z[..bytes].to_vec();
        }
        if f32 {
            bytemuck::cast_slice(&vec![value; n]).to_vec()
        } else {
            bytemuck::cast_slice(&vec![f16::from_f32(value); n]).to_vec()
        }
    }

    /// Get a texture with the grid's pixels at `mip` covering at least `need` (level rect),
    /// clipped to the grid's content. Returns None when the grid has nothing there.
    #[allow(clippy::too_many_arguments)]
    pub fn get(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mips: &mut MipCache,
        key: UploadKey,
        grid: &TileGrid,
        mip: u32,
        need: IRect,
        f32: bool,
        default_value: f32,
    ) -> Option<GpuImage> {
        // Level-space bounds of the grid content.
        let bounds = if mip == 0 {
            grid.tile_bounds()
        } else {
            match crate::levels::level_tile_bounds(grid, mip) {
                Some((a, b)) => IRect::new(a.x * TS, a.y * TS, (b.x + 1) * TS, (b.y + 1) * TS),
                None => IRect::EMPTY,
            }
        };
        // For masks with a non-zero default we still need a texture even if empty.
        let want = if default_value != 0.0 { need } else { need.intersect(&bounds) };
        if want.is_empty() {
            if let Some(e) = self.entries.get_mut(&key) {
                e.last_used = self.frame;
            }
            return None;
        }
        let align = |r: IRect| {
            IRect::new(
                r.x0.div_euclid(TS) * TS,
                r.y0.div_euclid(TS) * TS,
                (r.x1 + TS - 1).div_euclid(TS) * TS,
                (r.y1 + TS - 1).div_euclid(TS) * TS,
            )
        };
        let want = align(want);
        let reusable = self.entries.get(&key).is_some_and(|e| {
            e.mip == mip && e.f32 == f32 && e.default_value == default_value && e.image.rect.intersect(&want) == want
        });
        if !reusable {
            // Allocate with one tile of margin (clipped to content) to absorb small pans.
            let mut rect = want.expand(TS);
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
            let format = texture_format(grid.channels, f32);
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("layer"),
                size: wgpu::Extent3d { width: rect.width() as u32, height: rect.height() as u32, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            let pooled =
                Arc::new(PooledTex { tex, view, w: rect.width() as u32, h: rect.height() as u32, format });
            self.entries.insert(
                key,
                Entry {
                    image: GpuImage { tex: pooled, rect, straight: true },
                    mip,
                    f32,
                    default_value,
                    tiles: HashMap::new(),
                    last_used: self.frame,
                },
            );
        }
        let rect = self.entries[&key].image.rect;
        let coords: Vec<TileCoord> = TileGrid::coords_in(&rect).collect();
        mips.prefetch(grid, &coords, mip);
        let bpp = grid.channels as u32 * if f32 { 4 } else { 2 };
        let mut writes: Vec<(TileCoord, Option<Arc<Tile>>)> = vec![];
        {
            let e = self.entries.get_mut(&key).unwrap();
            e.last_used = self.frame;
            for c in coords {
                let t = mips.tile(grid, c, mip);
                let id = t.as_ref().map(|t| t.id()).unwrap_or(0);
                if e.tiles.get(&c) != Some(&id) {
                    e.tiles.insert(c, id);
                    writes.push((c, t));
                }
            }
        }
        let tex = self.entries[&key].image.tex.clone();
        for (c, t) in writes {
            let origin = wgpu::Origin3d { x: (c.x * TS - rect.x0) as u32, y: (c.y * TS - rect.y0) as u32, z: 0 };
            let data: std::borrow::Cow<[u8]> = match &t {
                Some(t) => tile_bytes(t, f32),
                None => std::borrow::Cow::Owned(self.filled(grid.channels, f32, default_value)),
            };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &tex.tex, mip_level: 0, origin, aspect: wgpu::TextureAspect::All },
                &data,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(TILE_SIZE * bpp), rows_per_image: Some(TILE_SIZE) },
                wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: 1 },
            );
        }
        let _ = PixelFormat::F16;
        Some(self.entries[&key].image.clone())
    }
}
