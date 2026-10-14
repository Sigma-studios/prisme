//! Selection outlines ("marching ants") via marching squares on the mask.

use prisme_core::tile::TileGrid;
use prisme_core::IRect;
use std::collections::HashMap;

/// Hash of the grid's tile ids (cheap change detection).
pub fn grid_key(g: &TileGrid) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut ids: Vec<(i32, i32, u64)> = g.tiles.iter().map(|(c, t)| (c.x, c.y, t.id())).collect();
    ids.sort_unstable();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ids.hash(&mut h);
    h.finish()
}

/// Extract iso-contours at 0.5 as polylines in document coordinates.
/// `step` subsamples large masks (1 = full resolution).
pub fn contours(mask: &TileGrid, bounds: IRect) -> Vec<Vec<[f32; 2]>> {
    let b = mask.content_bounds().intersect(&bounds.expand(1)).expand(1);
    if b.is_empty() {
        return vec![];
    }
    let step = ((b.area() as f64 / 4_000_000.0).sqrt().ceil() as i32).max(1);
    let w = (b.width() / step + 1) as usize;
    let h = (b.height() / step + 1) as usize;
    let v: Vec<bool> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as i32 * step + b.x0, (i / w) as i32 * step + b.y0);
            mask.get_pixel(x, y)[0] >= 0.5
        })
        .collect();
    let at = |x: usize, y: usize| v[y * w + x];
    // Edge midpoints as integer keys (2x grid).
    let mut segs: Vec<((i32, i32), (i32, i32))> = vec![];
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            let c = (at(x, y) as u8) | (at(x + 1, y) as u8) << 1 | (at(x + 1, y + 1) as u8) << 2 | (at(x, y + 1) as u8) << 3;
            let (xi, yi) = (2 * x as i32, 2 * y as i32);
            let top = (xi + 1, yi);
            let right = (xi + 2, yi + 1);
            let bottom = (xi + 1, yi + 2);
            let left = (xi, yi + 1);
            let mut add = |a, b| segs.push((a, b));
            match c {
                0 | 15 => {}
                1 | 14 => add(left, top),
                2 | 13 => add(top, right),
                3 | 12 => add(left, right),
                4 | 11 => add(right, bottom),
                5 => {
                    add(left, top);
                    add(right, bottom);
                }
                6 | 9 => add(top, bottom),
                7 | 8 => add(left, bottom),
                10 => {
                    add(top, right);
                    add(left, bottom);
                }
                _ => {}
            }
        }
    }
    // Chain segments into polylines.
    let mut adj: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for (i, (a, b)) in segs.iter().enumerate() {
        adj.entry(*a).or_default().push(i);
        adj.entry(*b).or_default().push(i);
    }
    let mut used = vec![false; segs.len()];
    let to_doc = |p: (i32, i32)| [b.x0 as f32 + p.0 as f32 * 0.5 * step as f32 + 0.5 * step as f32 - 0.5 * step as f32 + 0.5, b.y0 as f32 + p.1 as f32 * 0.5 * step as f32 + 0.5];
    let mut lines = vec![];
    for s in 0..segs.len() {
        if used[s] {
            continue;
        }
        used[s] = true;
        let (a, mut cur) = segs[s];
        let mut pts = vec![to_doc(a), to_doc(cur)];
        loop {
            let next = adj.get(&cur).and_then(|v| v.iter().copied().find(|&i| !used[i]));
            let Some(n) = next else { break };
            used[n] = true;
            let (p, q) = segs[n];
            cur = if p == cur { q } else { p };
            pts.push(to_doc(cur));
            if cur == a {
                break;
            }
        }
        if pts.len() >= 2 {
            lines.push(pts);
        }
        if lines.len() > 20_000 {
            break;
        }
    }
    lines
}
