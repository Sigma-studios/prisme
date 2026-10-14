//! CPU flattening of raster/fill layers, used for "sample all layers" tools.
//! (Display and export use the GPU engine; this is an approximation with normal blending.)

use prisme_core::layer::{FillContent, Layer, LayerKind};
use prisme_core::tile::{TileCoord, TILE_PIXELS, TILE_SIZE};
use prisme_core::Document;

/// Composite visible layers for one tile. Returns straight-alpha RGBA f32.
pub fn flatten_tile(doc: &Document, c: TileCoord) -> Vec<f32> {
    let mut acc = vec![0.0f32; TILE_PIXELS * 4]; // premultiplied
    composite_into(&doc.layers, c, &mut acc, 1.0);
    for p in acc.chunks_exact_mut(4) {
        if p[3] > 1e-7 {
            p[0] /= p[3];
            p[1] /= p[3];
            p[2] /= p[3];
        }
    }
    acc
}

fn composite_into(layers: &[Layer], c: TileCoord, acc: &mut [f32], parent_opacity: f32) {
    let ts = TILE_SIZE as usize;
    for l in layers {
        if !l.visible || l.opacity <= 0.0 {
            continue;
        }
        let op = l.opacity * parent_opacity;
        let mask = l.mask.as_ref().filter(|m| m.enabled).map(|m| (m.grid.get_tile(c).map(|t| t.f32_view().into_owned()), m.default_visible));
        let mask_at = |i: usize| -> f32 {
            match &mask {
                None => 1.0,
                Some((Some(v), _)) => v[i],
                Some((None, d)) => *d as i32 as f32,
            }
        };
        match &l.kind {
            LayerKind::Raster(g) => {
                let Some(t) = g.get_tile(c) else { continue };
                let v = t.f32_view();
                for i in 0..ts * ts {
                    let a = v[i * 4 + 3] * op * mask_at(i);
                    if a <= 0.0 {
                        continue;
                    }
                    for k in 0..3 {
                        acc[i * 4 + k] = v[i * 4 + k] * a + acc[i * 4 + k] * (1.0 - a);
                    }
                    acc[i * 4 + 3] = a + acc[i * 4 + 3] * (1.0 - a);
                }
            }
            LayerKind::Fill(f) => {
                let r = c.rect();
                for i in 0..ts * ts {
                    let (x, y) = (r.x0 + (i % ts) as i32, r.y0 + (i / ts) as i32);
                    let col = match &f.content {
                        FillContent::Solid(col) => *col,
                        FillContent::Gradient(g, geo) => g.eval(geo.t_at([x as f32 + 0.5, y as f32 + 0.5])),
                    };
                    let a = col[3] * op * mask_at(i);
                    for k in 0..3 {
                        acc[i * 4 + k] = col[k] * a + acc[i * 4 + k] * (1.0 - a);
                    }
                    acc[i * 4 + 3] = a + acc[i * 4 + 3] * (1.0 - a);
                }
            }
            LayerKind::Group(g) => composite_into(&g.children, c, acc, op),
            LayerKind::Text(_) | LayerKind::Vector(_) => {}
        }
    }
}

pub fn flatten_pixel(doc: &Document, x: i32, y: i32) -> [f32; 4] {
    let c = TileCoord::of_pixel(x, y);
    let t = flatten_tile(doc, c);
    let r = c.rect();
    let i = ((y - r.y0) as usize * TILE_SIZE as usize + (x - r.x0) as usize) * 4;
    [t[i], t[i + 1], t[i + 2], t[i + 3]]
}
