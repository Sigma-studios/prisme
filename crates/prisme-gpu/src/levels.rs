//! CPU mip pyramid of tile grids, computed lazily and cached by child tile ids.

use prisme_core::tile::{Tile, TileCoord, TileGrid};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Default)]
pub struct MipCache {
    /// (mip, child ids TL TR BL BR) -> downsampled tile (None = all children empty).
    cache: HashMap<(u32, [u64; 4]), Arc<Tile>>,
    /// Entries used since the last `end_frame` (for eviction).
    used: std::collections::HashSet<(u32, [u64; 4])>,
}

impl MipCache {
    /// Tile at `coord` of mip level `mip` (coordinates in that level's tile grid).
    pub fn tile(&mut self, grid: &TileGrid, coord: TileCoord, mip: u32) -> Option<Arc<Tile>> {
        if mip == 0 {
            return grid.get_tile(coord).cloned();
        }
        let children = [
            TileCoord::new(coord.x * 2, coord.y * 2),
            TileCoord::new(coord.x * 2 + 1, coord.y * 2),
            TileCoord::new(coord.x * 2, coord.y * 2 + 1),
            TileCoord::new(coord.x * 2 + 1, coord.y * 2 + 1),
        ];
        let kids: Vec<Option<Arc<Tile>>> = children.iter().map(|c| self.tile(grid, *c, mip - 1)).collect();
        if kids.iter().all(|k| k.is_none()) {
            return None;
        }
        let ids = [0, 1, 2, 3].map(|i| kids[i].as_ref().map(|t| t.id()).unwrap_or(0));
        let key = (mip, ids);
        self.used.insert(key);
        if let Some(t) = self.cache.get(&key) {
            return Some(t.clone());
        }
        let refs = [0, 1, 2, 3].map(|i| kids[i].as_deref());
        let t = Arc::new(Tile::downsample(refs, grid.format, grid.channels));
        self.cache.insert(key, t.clone());
        Some(t)
    }

    /// Precompute (in parallel on native) all tiles of a level covering `coords`.
    pub fn prefetch(&mut self, grid: &TileGrid, coords: &[TileCoord], mip: u32) {
        if mip == 0 || coords.len() < 4 {
            return;
        }
        // Compute the level below in parallel first, then this level sequentially (cheap).
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            if mip == 1 {
                // Gather jobs whose results aren't cached yet.
                let jobs: Vec<(TileCoord, [Option<Arc<Tile>>; 4])> = coords
                    .iter()
                    .filter_map(|c| {
                        let kids = [
                            grid.get_tile(TileCoord::new(c.x * 2, c.y * 2)).cloned(),
                            grid.get_tile(TileCoord::new(c.x * 2 + 1, c.y * 2)).cloned(),
                            grid.get_tile(TileCoord::new(c.x * 2, c.y * 2 + 1)).cloned(),
                            grid.get_tile(TileCoord::new(c.x * 2 + 1, c.y * 2 + 1)).cloned(),
                        ];
                        if kids.iter().all(|k| k.is_none()) {
                            return None;
                        }
                        let ids = [0, 1, 2, 3].map(|i| kids[i].as_ref().map(|t| t.id()).unwrap_or(0));
                        (!self.cache.contains_key(&(1, ids))).then_some((*c, kids))
                    })
                    .collect();
                let results: Vec<([u64; 4], Arc<Tile>)> = jobs
                    .par_iter()
                    .map(|(_, kids)| {
                        let ids = [0, 1, 2, 3].map(|i| kids[i].as_ref().map(|t| t.id()).unwrap_or(0));
                        let refs = [0, 1, 2, 3].map(|i| kids[i].as_deref());
                        (ids, Arc::new(Tile::downsample(refs, grid.format, grid.channels)))
                    })
                    .collect();
                for (ids, t) in results {
                    self.cache.insert((1, ids), t);
                }
            } else {
                let below: Vec<TileCoord> = coords
                    .iter()
                    .flat_map(|c| {
                        [
                            TileCoord::new(c.x * 2, c.y * 2),
                            TileCoord::new(c.x * 2 + 1, c.y * 2),
                            TileCoord::new(c.x * 2, c.y * 2 + 1),
                            TileCoord::new(c.x * 2 + 1, c.y * 2 + 1),
                        ]
                    })
                    .collect();
                self.prefetch(grid, &below, mip - 1);
            }
        }
        let _ = (grid, coords);
    }

    /// Evict entries not used since the last call when the cache grows large.
    pub fn end_frame(&mut self, max_entries: usize) {
        if self.cache.len() > max_entries {
            let used = std::mem::take(&mut self.used);
            self.cache.retain(|k, _| used.contains(k));
        } else {
            self.used.clear();
        }
    }
}

/// Bounds of a grid's tiles at a mip level (in level tile coordinates).
pub fn level_tile_bounds(grid: &TileGrid, mip: u32) -> Option<(TileCoord, TileCoord)> {
    if grid.tiles.is_empty() {
        return None;
    }
    let s = 1i32 << mip;
    let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for c in grid.tiles.keys() {
        x0 = x0.min(c.x.div_euclid(s));
        y0 = y0.min(c.y.div_euclid(s));
        x1 = x1.max(c.x.div_euclid(s));
        y1 = y1.max(c.y.div_euclid(s));
    }
    Some((TileCoord::new(x0, y0), TileCoord::new(x1, y1)))
}
