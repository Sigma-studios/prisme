//! Geometric operations on tile grids: translation, projective resampling, merging
//! floating selections, whole-document resize/rotation.

use prisme_core::geom::{Affine2, Homography};
use prisme_core::layer::{FillContent, Layer, LayerKind};
use prisme_core::tile::{Tile, TileCoord, TileGrid, TILE_PIXELS, TILE_SIZE};
use prisme_core::{Document, IRect};
use std::sync::Arc;

const TS: i32 = TILE_SIZE as i32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resample {
    Nearest,
    Bilinear,
    Bicubic,
}

fn build_tiles(coords: Vec<TileCoord>, channels: u8, format: prisme_core::PixelFormat, f: impl Fn(TileCoord, &mut [f32]) -> bool + Sync + Send) -> TileGrid {
    let mut jobs: Vec<(TileCoord, Option<Tile>)> = coords.into_iter().map(|c| (c, None)).collect();
    crate::par_for_each(&mut jobs, |(c, out)| {
        let mut buf = vec![0.0f32; TILE_PIXELS * channels as usize];
        if f(*c, &mut buf) {
            let t = Tile::from_f32(format, channels, &buf);
            if !t.is_clear() {
                *out = Some(t);
            }
        }
    });
    let mut g = TileGrid::new(format, channels);
    for (c, t) in jobs {
        if let Some(t) = t {
            g.tiles.insert(c, Arc::new(t));
        }
    }
    g
}

/// Integer translation, computed tile by tile in parallel.
pub fn translate(src: &TileGrid, dx: i32, dy: i32) -> TileGrid {
    if dx.rem_euclid(TS) == 0 && dy.rem_euclid(TS) == 0 {
        return src.translated(dx, dy);
    }
    let b = src.tile_bounds().translate(dx, dy);
    let coords: Vec<TileCoord> = TileGrid::coords_in(&b).collect();
    let ch = src.channels as usize;
    build_tiles(coords, src.channels, src.format, |c, buf| {
        let r = c.rect().translate(-dx, -dy);
        let data = src.read_rect(&r);
        if data.iter().all(|&v| v == 0.0) {
            return false;
        }
        buf.copy_from_slice(&data[..TILE_PIXELS * ch]);
        true
    })
}

fn cubic(t: f32) -> [f32; 4] {
    // Catmull-Rom
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

fn sample(src: &TileGrid, x: f32, y: f32, mode: Resample) -> [f32; 4] {
    match mode {
        Resample::Nearest => src.get_pixel(x.floor() as i32, y.floor() as i32),
        Resample::Bilinear => src.sample_bilinear(x, y),
        Resample::Bicubic => {
            let fx = x - 0.5;
            let fy = y - 0.5;
            let ix = fx.floor() as i32;
            let iy = fy.floor() as i32;
            let wx = cubic(fx - ix as f32);
            let wy = cubic(fy - iy as f32);
            let mut acc = [0.0f32; 4];
            for (j, wyj) in wy.iter().enumerate() {
                for (i, wxi) in wx.iter().enumerate() {
                    let p = src.get_pixel(ix - 1 + i as i32, iy - 1 + j as i32);
                    let w = wxi * wyj;
                    if src.channels == 1 {
                        acc[0] += p[0] * w;
                    } else {
                        let a = p[3] * w;
                        acc[0] += p[0] * a;
                        acc[1] += p[1] * a;
                        acc[2] += p[2] * a;
                        acc[3] += a;
                    }
                }
            }
            if src.channels == 1 {
                return [acc[0].clamp(0.0, 1.0); 4];
            }
            if acc[3] > 1e-6 {
                [acc[0] / acc[3], acc[1] / acc[3], acc[2] / acc[3], acc[3].clamp(0.0, 1.0)]
            } else {
                [0.0; 4]
            }
        }
    }
}

/// Resample `src` through a projective map `h` (src -> dst) into `dst_bounds`.
pub fn warp(src: &TileGrid, h: &Homography, dst_bounds: IRect, mode: Resample) -> TileGrid {
    let Some(inv) = h.inverse() else { return TileGrid::new(src.format, src.channels) };
    let coords: Vec<TileCoord> = TileGrid::coords_in(&dst_bounds).collect();
    let ch = src.channels as usize;
    build_tiles(coords, src.channels, src.format, |c, buf| {
        let r = c.rect().intersect(&dst_bounds);
        let mut any = false;
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                let s = inv.apply([x as f32 + 0.5, y as f32 + 0.5]);
                if !s[0].is_finite() || !s[1].is_finite() {
                    continue;
                }
                let p = sample(src, s[0], s[1], mode);
                let i = ((y - c.y * TS) * TS + (x - c.x * TS)) as usize * ch;
                buf[i..i + ch].copy_from_slice(&p[..ch]);
                any |= p[if ch == 4 { 3 } else { 0 }] > 0.0;
            }
        }
        any
    })
}

/// Bounds of a quad.
pub fn quad_bounds(q: &[[f32; 2]; 4]) -> IRect {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in q {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    IRect::from_f32(x0, y0, x1, y1)
}

/// Split a layer grid by a selection: (floating = grid × sel, remainder = grid × (1 − sel)).
pub fn split_by_selection(grid: &TileGrid, sel: &TileGrid) -> (TileGrid, TileGrid) {
    let mut floating = TileGrid::new(grid.format, grid.channels);
    let mut rest = grid.clone();
    let ch = grid.channels as usize;
    for (c, t) in &grid.tiles {
        let Some(st) = sel.get_tile(*c) else { continue };
        let sv = st.f32_view();
        let tv = t.f32_view();
        let mut f = vec![0.0f32; TILE_PIXELS * ch];
        let mut r = tv.to_vec();
        for i in 0..TILE_PIXELS {
            let s = sv[i];
            if s <= 0.0 {
                continue;
            }
            if ch == 4 {
                f[i * 4..i * 4 + 3].copy_from_slice(&tv[i * 4..i * 4 + 3]);
                f[i * 4 + 3] = tv[i * 4 + 3] * s;
                r[i * 4 + 3] = tv[i * 4 + 3] * (1.0 - s);
            } else {
                f[i] = tv[i] * s;
                r[i] = tv[i] * (1.0 - s);
            }
        }
        floating.insert(*c, Tile::from_f32(grid.format, grid.channels, &f));
        rest.insert(*c, Tile::from_f32(grid.format, grid.channels, &r));
    }
    floating.remove_clear_tiles();
    rest.remove_clear_tiles();
    (floating, rest)
}

/// Composite `top` over `base` (straight alpha, normal blend).
pub fn merge_over(base: &TileGrid, top: &TileGrid) -> TileGrid {
    let mut out = base.clone();
    for (c, t) in &top.tiles {
        let tv = t.f32_view();
        match base.get_tile(*c) {
            None => {
                out.insert(*c, Tile::from_f32(base.format, base.channels, &tv));
            }
            Some(bt) => {
                let bv = bt.f32_view();
                let mut o = bv.to_vec();
                for i in 0..TILE_PIXELS {
                    let p = i * 4;
                    let a = tv[p + 3];
                    if a <= 0.0 {
                        continue;
                    }
                    let ab = bv[p + 3];
                    let ao = a + ab * (1.0 - a);
                    for k in 0..3 {
                        o[p + k] = (tv[p + k] * a + bv[p + k] * ab * (1.0 - a)) / ao.max(1e-7);
                    }
                    o[p + 3] = ao;
                }
                out.insert(*c, Tile::from_f32(base.format, base.channels, &o));
            }
        }
    }
    out
}

fn transform_layers(layers: &mut [Layer], h: &Homography, aff: &Affine2, scale: f32, bounds_of: &dyn Fn(&TileGrid) -> IRect, mode: Resample) {
    for l in layers {
        if let Some(m) = &mut l.mask {
            let b = bounds_of(&m.grid);
            m.grid = Arc::new(warp(&m.grid, h, b, mode));
        }
        match &mut l.kind {
            LayerKind::Raster(g) => {
                let b = bounds_of(g);
                *g = Arc::new(warp(g, h, b, mode));
            }
            LayerKind::Vector(v) => v.transform = aff.mul(&v.transform),
            LayerKind::Text(t) => {
                t.position = aff.apply(t.position);
                t.size *= scale;
                t.rotation += aff.m[3].atan2(aff.m[0]);
            }
            LayerKind::Fill(f) => {
                if let FillContent::Gradient(_, geo) = &mut f.content {
                    geo.start = aff.apply(geo.start);
                    geo.end = aff.apply(geo.end);
                }
            }
            LayerKind::Group(g) => transform_layers(&mut g.children, h, aff, scale, bounds_of, mode),
        }
    }
}

/// Resample the whole document to a new pixel size ("Image Size").
pub fn resize_document(doc: &mut Document, w: u32, h: u32, mode: Resample) {
    let sx = w as f32 / doc.info.width as f32;
    let sy = h as f32 / doc.info.height as f32;
    let aff = Affine2::scale(sx, sy);
    let hm = Homography::from_affine(&aff);
    let bounds_of = |g: &TileGrid| {
        let b = g.tile_bounds();
        IRect::from_f32(b.x0 as f32 * sx, b.y0 as f32 * sy, b.x1 as f32 * sx, b.y1 as f32 * sy)
    };
    transform_layers(&mut doc.layers, &hm, &aff, (sx * sy).sqrt(), &bounds_of, mode);
    if let Some(s) = &doc.selection {
        doc.selection = Some(Arc::new(warp(s, &hm, bounds_of(s), mode)));
    }
    for m in &mut doc.masks {
        m.mask = Arc::new(warp(&m.mask, &hm, bounds_of(&m.mask), mode));
    }
    doc.info.width = w;
    doc.info.height = h;
}

/// Rotate all layers by `angle` radians around the canvas centre (canvas size unchanged).
pub fn rotate_document(doc: &mut Document, angle: f32, mode: Resample) {
    let (cx, cy) = (doc.info.width as f32 * 0.5, doc.info.height as f32 * 0.5);
    let aff = Affine2::translate(cx, cy).mul(&Affine2::rotate(angle)).mul(&Affine2::translate(-cx, -cy));
    let hm = Homography::from_affine(&aff);
    let bounds_of = |g: &TileGrid| {
        let b = g.tile_bounds();
        let q = [
            aff.apply([b.x0 as f32, b.y0 as f32]),
            aff.apply([b.x1 as f32, b.y0 as f32]),
            aff.apply([b.x1 as f32, b.y1 as f32]),
            aff.apply([b.x0 as f32, b.y1 as f32]),
        ];
        quad_bounds(&q)
    };
    transform_layers(&mut doc.layers, &hm, &aff, 1.0, &bounds_of, mode);
    if let Some(s) = &doc.selection {
        doc.selection = Some(Arc::new(warp(s, &hm, bounds_of(s), mode)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prisme_core::PixelFormat;

    #[test]
    fn translate_matches_reference() {
        let mut g = TileGrid::rgba(PixelFormat::F32);
        g.set_pixel(10, 20, [1.0, 0.5, 0.25, 1.0]);
        g.set_pixel(300, 5, [0.1, 0.2, 0.3, 0.4]);
        let t = translate(&g, 37, -3);
        assert_eq!(t.get_pixel(47, 17), [1.0, 0.5, 0.25, 1.0]);
        assert_eq!(t.get_pixel(337, 2), [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(t.content_bounds(), IRect::new(47, 2, 338, 18));
    }

    #[test]
    fn identity_warp_preserves() {
        let mut g = TileGrid::rgba(PixelFormat::F32);
        g.map_rect(&IRect::new(5, 5, 50, 50), |x, y, p| p.copy_from_slice(&[x as f32 / 50.0, y as f32 / 50.0, 0.5, 1.0]));
        let w = warp(&g, &Homography::IDENTITY, IRect::new(0, 0, 64, 64), Resample::Bilinear);
        for (x, y) in [(5, 5), (20, 30), (49, 49)] {
            let a = g.get_pixel(x, y);
            let b = w.get_pixel(x, y);
            for k in 0..4 {
                assert!((a[k] - b[k]).abs() < 1e-5, "{a:?} {b:?}");
            }
        }
    }

    #[test]
    fn split_and_merge_roundtrip() {
        let mut g = TileGrid::rgba(PixelFormat::F32);
        g.map_rect(&IRect::new(0, 0, 40, 40), |_, _, p| p.copy_from_slice(&[0.3, 0.6, 0.9, 1.0]));
        let sel = prisme_core::mask::rect([10.0, 10.0, 30.0, 30.0], false);
        let (f, r) = split_by_selection(&g, &sel);
        assert_eq!(f.get_pixel(5, 5)[3], 0.0);
        assert_eq!(r.get_pixel(20, 20)[3], 0.0);
        let m = merge_over(&r, &f);
        let p = m.get_pixel(20, 20);
        assert!((p[3] - 1.0).abs() < 1e-6 && (p[1] - 0.6).abs() < 1e-6);
    }
}
