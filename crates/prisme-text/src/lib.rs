//! Parametric text rendering for Prisme.
//!
//! * [`FontDb`] — bundled fallback fonts (always available, also on the web), system
//!   fonts (native, enumerated in the background) and user font files.
//! * [`TextRenderer`] — shapes (`rustybuzz`), lays out (explicit line breaks, alignment,
//!   line height, letter spacing) and rasterizes [`TextLayer`]s with their effects
//!   (gradient fill, outline, outer/inner glow, drop shadow), warps and 3D extrusion onto
//!   a sphere, at any mip level.
//!
//! Output grids use level-space pixel coordinates (document × 2^-mip), straight alpha,
//! scene-linear working-space RGBA in F16 tiles.
//!
//! Units: `size`, `letter_spacing` (extra advance after each character), outline width,
//! glow radius, shadow offset/blur, warp amplitudes/wavelengths and 3D depth/radius are
//! document pixels; `line_height` is a multiple of `size`.

mod fonts;
mod geom;
mod layout;
mod raster;
mod three_d;

pub use fonts::{BUNDLED_MONO, BUNDLED_SANS, FontDb};

use geom::{OutlineCollector, P2, Seg};
use layout::{Layout, LayoutKey, PlacedGlyph};
use prisme_core::geom::IRect;
use prisme_core::layer::{TextFill, TextLayer};
use prisme_core::{PixelFormat, TileGrid};
use raster::Effects;
use rustybuzz::ttf_parser;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use three_d::{Mesh, Paint};
use tiny_skia::{FillRule, Mask, Path, PathBuilder, Stroke, Transform};

/// Output chunk side (multiple of the tile size); each chunk is rendered independently.
const CHUNK: i32 = 512;
/// Level-space coordinate clamp to avoid pathological allocations.
const MAX_COORD: f32 = (1 << 20) as f32;
const LAYOUT_CACHE_CAP: usize = 256;
const OUTLINE_CACHE_CAP: usize = 20_000;
/// Synthetic bold stroke width as a fraction of the font size.
const SYNTH_BOLD: f32 = 0.035;
/// Synthetic italic shear.
const SYNTH_ITALIC: f32 = 0.2;

enum FlatPaint {
    Solid([f32; 4]),
    /// LUT + level-space affine parameter `t = a·x + b·y + c`.
    Gradient(Vec<[f32; 4]>, [f32; 3]),
}

enum Shape {
    Flat {
        fill: Option<Path>,
        bold: Option<Path>,
        outline: Vec<Path>,
        paint: FlatPaint,
    },
    Mesh(Box<Mesh>),
}

/// (face, glyph id, `wght` bits)
type OutlineKey = (fontdb::ID, u16, u32);
/// (coverage, premultiplied fill, shape ∪ outline coverage)
type ShapePlanes = (Vec<f32>, Vec<[f32; 4]>, Option<Vec<f32>>);

struct Prepared {
    shape: Shape,
    /// Level-space bounds of the shape (without effects).
    bbox: [f32; 4],
    fx: Effects,
}

/// Rasterizer for text layers. Caches shaped layouts and glyph outlines, so re-rendering
/// at another zoom level only re-flattens and re-rasterizes.
pub struct TextRenderer {
    fonts: Arc<FontDb>,
    layouts: Mutex<HashMap<LayoutKey, Arc<Layout>>>,
    outlines: Mutex<HashMap<OutlineKey, Arc<Vec<Seg>>>>,
}

fn union(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

fn valid(b: &[f32; 4]) -> bool {
    b.iter().all(|v| v.is_finite()) && b[2] >= b[0] && b[3] >= b[1]
}

fn lut_of(g: &prisme_core::Gradient) -> Vec<[f32; 4]> {
    g.to_lut(256)
}

impl TextRenderer {
    pub fn new(fonts: Arc<FontDb>) -> Self {
        Self {
            fonts,
            layouts: Mutex::new(HashMap::new()),
            outlines: Mutex::new(HashMap::new()),
        }
    }

    pub fn fonts(&self) -> &Arc<FontDb> {
        &self.fonts
    }

    fn layout(&self, l: &TextLayer) -> Arc<Layout> {
        let key = LayoutKey::of(l, self.fonts.generation());
        if let Some(v) = self.layouts.lock().unwrap().get(&key) {
            return v.clone();
        }
        let lay = Arc::new(layout::layout(&self.fonts, l));
        let mut c = self.layouts.lock().unwrap();
        if c.len() >= LAYOUT_CACHE_CAP {
            c.clear();
        }
        c.insert(key, lay.clone());
        lay
    }

    fn outline(&self, g: &PlacedGlyph) -> Option<Arc<Vec<Seg>>> {
        let key = (g.face, g.gid, g.wght.map(f32::to_bits).unwrap_or(0));
        if let Some(o) = self.outlines.lock().unwrap().get(&key) {
            return Some(o.clone());
        }
        let segs = self.fonts.with_face(g.face, |f| {
            if let Some(w) = g.wght {
                f.set_variations(&[rustybuzz::Variation {
                    tag: ttf_parser::Tag::from_bytes(b"wght"),
                    value: w,
                }]);
            }
            let mut c = OutlineCollector::default();
            f.outline_glyph(ttf_parser::GlyphId(g.gid), &mut c);
            c.0
        })?;
        let segs = Arc::new(segs);
        let mut c = self.outlines.lock().unwrap();
        if c.len() >= OUTLINE_CACHE_CAP {
            c.clear();
        }
        c.insert(key, segs.clone());
        Some(segs)
    }

    /// Per-glyph closed polylines in text space (warp applied) + synthetic-bold flag.
    fn glyph_contours(
        &self,
        l: &TextLayer,
        lay: &Layout,
        tol: f32,
        max_len: f32,
    ) -> Vec<(Vec<Vec<P2>>, bool)> {
        let block = [lay.width, lay.height];
        let warp = geom::warp_active(&l.warp).then_some(l.warp);
        let mut out = Vec::with_capacity(lay.glyphs.len());
        for g in &lay.glyphs {
            let Some(segs) = self.outline(g) else {
                continue;
            };
            if segs.is_empty() {
                continue;
            }
            let skew = if g.synth_italic { SYNTH_ITALIC } else { 0.0 };
            let map = |p: P2| -> P2 {
                let q = [g.x + (p[0] + skew * p[1]) * g.scale, g.y - p[1] * g.scale];
                match &warp {
                    Some(w) => geom::warp_point(w, block, q),
                    None => q,
                }
            };
            let mut contours = Vec::new();
            geom::flatten(&segs, map, tol, max_len, &mut contours);
            if !contours.is_empty() {
                out.push((contours, g.synth_bold));
            }
        }
        out
    }

    fn prepare(&self, l: &TextLayer, s: f32) -> Option<Prepared> {
        let lay = self.layout(l);
        if lay.glyphs.is_empty() {
            return None;
        }
        let tol = 0.2 / s;
        let warp_len = if geom::warp_active(&l.warp) {
            3.0 / s
        } else {
            f32::INFINITY
        };
        let td = &l.three_d;
        let max_len = if td.enabled && td.radius > 1e-3 {
            warp_len.min((td.radius * 0.04).max(tol * 4.0))
        } else {
            warp_len
        };
        let glyphs = self.glyph_contours(l, &lay, tol, max_len);
        if glyphs.is_empty() {
            return None;
        }
        // Ink bounds (text space) for the gradient span.
        let mut ink = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        for (cs, _) in &glyphs {
            for p in cs.iter().flatten() {
                ink = union(ink, [p[0], p[1], p[0], p[1]]);
            }
        }
        let text_t = |angle: f32| -> [f32; 3] {
            let (sa, ca) = angle.sin_cos();
            let (iw, ih) = (ink[2] - ink[0], ink[3] - ink[1]);
            let ext = (iw * ca).abs() + (ih * sa).abs();
            let ext = ext.max(1e-3);
            let c = [(ink[0] + ink[2]) * 0.5, (ink[1] + ink[3]) * 0.5];
            [ca / ext, sa / ext, 0.5 - (c[0] * ca + c[1] * sa) / ext]
        };
        let fx = Effects::new(&l.outline, &l.glow, &l.shadow, s);
        let (sr, cr) = l.rotation.sin_cos();

        if td.enabled {
            let (fill, fill_t) = match &l.fill {
                TextFill::Solid(c) => (Paint::Solid(*c), [0.0; 3]),
                TextFill::Gradient { gradient, angle } => {
                    (Paint::Lut(lut_of(gradient)), text_t(*angle))
                }
            };
            let side = match &td.side_gradient {
                Some(g) => Paint::Lut(lut_of(g)),
                None => Paint::Solid(td.side_color),
            };
            let contours: Vec<Vec<Vec<P2>>> = glyphs.into_iter().map(|(c, _)| c).collect();
            let mesh = Mesh::build(
                &contours,
                three_d::Params {
                    cfg: td,
                    block: [lay.width, lay.height],
                    position: l.position,
                    rotation: l.rotation,
                    scale: s,
                    fill,
                    fill_t,
                    side,
                    max_len,
                },
            );
            if mesh.is_empty() || !valid(&mesh.bbox) {
                return None;
            }
            let bbox = mesh.bbox;
            return Some(Prepared {
                shape: Shape::Mesh(Box::new(mesh)),
                bbox,
                fx,
            });
        }

        // 2D: text → level affine.
        let to_level = |p: P2| -> P2 {
            [
                s * (l.position[0] + cr * p[0] - sr * p[1]),
                s * (l.position[1] + sr * p[0] + cr * p[1]),
            ]
        };
        let mut pb = PathBuilder::new();
        let mut bb = PathBuilder::new();
        for (cs, bold) in &glyphs {
            let b = if *bold { &mut bb } else { &mut pb };
            for c in cs {
                let p0 = to_level(c[0]);
                b.move_to(p0[0], p0[1]);
                for p in &c[1..] {
                    let q = to_level(*p);
                    b.line_to(q[0], q[1]);
                }
                b.close();
            }
        }
        let fill = pb.finish();
        let bold_w = l.size * SYNTH_BOLD * s;
        let bold = bb.finish().and_then(|p| {
            let st = Stroke {
                width: bold_w,
                line_join: tiny_skia::LineJoin::Round,
                ..Default::default()
            };
            let stroked = p.stroke(&st, 1.0);
            // Keep both: fill of the outline and its stroke.
            let mut b = PathBuilder::new();
            b.push_path(&p);
            if let Some(sp) = stroked {
                b.push_path(&sp);
            }
            b.finish()
        });
        let mut bbox = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        for p in fill.iter().chain(bold.iter()) {
            let r = p.bounds();
            bbox = union(bbox, [r.left(), r.top(), r.right(), r.bottom()]);
        }
        if !valid(&bbox) {
            return None;
        }
        let mut outline = Vec::new();
        if let Some((ow, _)) = fx.outline {
            let st = Stroke {
                width: 2.0 * ow,
                line_join: tiny_skia::LineJoin::Round,
                line_cap: tiny_skia::LineCap::Round,
                ..Default::default()
            };
            outline.extend(
                fill.iter()
                    .chain(bold.iter())
                    .filter_map(|p| p.stroke(&st, 1.0)),
            );
        }
        let paint = match &l.fill {
            TextFill::Solid(c) => FlatPaint::Solid(*c),
            TextFill::Gradient { gradient, angle } => {
                let t = text_t(*angle);
                // t(q) = a·q + c with q = R⁻¹(L/s − pos)  ⇒  t(L) = (R a)·L/s + c − (R a)·pos
                let ra = [cr * t[0] - sr * t[1], sr * t[0] + cr * t[1]];
                FlatPaint::Gradient(
                    lut_of(gradient),
                    [
                        ra[0] / s,
                        ra[1] / s,
                        t[2] - ra[0] * l.position[0] - ra[1] * l.position[1],
                    ],
                )
            }
        };
        Some(Prepared {
            shape: Shape::Flat {
                fill,
                bold,
                outline,
                paint,
            },
            bbox,
            fx,
        })
    }

    /// Render the shape (coverage, premultiplied fill, outline coverage) in region `e`.
    fn render_shape(p: &Prepared, e: &IRect) -> Option<ShapePlanes> {
        let (w, h) = (e.width() as usize, e.height() as usize);
        match &p.shape {
            Shape::Flat {
                fill,
                bold,
                outline,
                paint,
            } => {
                let tr = Transform::from_translate(-e.x0 as f32, -e.y0 as f32);
                let mut m = Mask::new(w as u32, h as u32)?;
                for path in fill.iter().chain(bold.iter()) {
                    m.fill_path(path, FillRule::Winding, true, tr);
                }
                let cov: Vec<f32> = m.data().iter().map(|&v| v as f32 * (1.0 / 255.0)).collect();
                let mut col = vec![[0.0f32; 4]; w * h];
                match paint {
                    FlatPaint::Solid(c) => {
                        let a0 = c[3].clamp(0.0, 1.0);
                        for (o, &a) in col.iter_mut().zip(&cov) {
                            let a = a * a0;
                            *o = [c[0] * a, c[1] * a, c[2] * a, a];
                        }
                    }
                    FlatPaint::Gradient(lut, k) => {
                        let n = (lut.len() - 1) as f32;
                        for y in 0..h {
                            let py = (e.y0 + y as i32) as f32 + 0.5;
                            for x in 0..w {
                                let a = cov[y * w + x];
                                if a <= 0.0 {
                                    continue;
                                }
                                let px = (e.x0 + x as i32) as f32 + 0.5;
                                let t = (k[0] * px + k[1] * py + k[2]).clamp(0.0, 1.0);
                                let c = lut[(t * n).round() as usize];
                                let a = a * c[3].clamp(0.0, 1.0);
                                col[y * w + x] = [c[0] * a, c[1] * a, c[2] * a, a];
                            }
                        }
                    }
                }
                let oc = if outline.is_empty() {
                    None
                } else {
                    let mut om = Mask::new(w as u32, h as u32)?;
                    for path in outline {
                        om.fill_path(path, FillRule::Winding, true, tr);
                    }
                    Some(
                        om.data()
                            .iter()
                            .zip(&cov)
                            .map(|(&v, &a)| (v as f32 * (1.0 / 255.0)).max(a))
                            .collect(),
                    )
                };
                Some((cov, col, oc))
            }
            Shape::Mesh(mesh) => {
                let (cov, col) = mesh.render(e);
                let oc = p.fx.outline.map(|(ow, _)| raster::dilate(&cov, w, h, ow));
                Some((cov, col, oc))
            }
        }
    }

    /// Rasterize the full text layer (effects, warp and 3D included) at `mip`, in level
    /// coordinates. Returns `None` for empty text.
    pub fn rasterize(&self, layer: &TextLayer, mip: u32) -> Option<TileGrid> {
        let s = 0.5f32.powi(mip as i32);
        let p = self.prepare(layer, s)?;
        let ext = p.fx.extents();
        let b = [
            p.bbox[0] - ext[0],
            p.bbox[1] - ext[1],
            p.bbox[2] + ext[2],
            p.bbox[3] + ext[3],
        ];
        let total = IRect::from_f32(
            b[0].max(-MAX_COORD),
            b[1].max(-MAX_COORD),
            b[2].min(MAX_COORD),
            b[3].min(MAX_COORD),
        )
        .expand(1);
        if total.is_empty() {
            return None;
        }
        let margin = p.fx.margin();
        let shape_rect = IRect::from_f32(p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3]).expand(1);
        let chunks: Vec<IRect> = {
            let (cx0, cy0) = (total.x0.div_euclid(CHUNK), total.y0.div_euclid(CHUNK));
            let (cx1, cy1) = (
                (total.x1 - 1).div_euclid(CHUNK),
                (total.y1 - 1).div_euclid(CHUNK),
            );
            (cy0..=cy1)
                .flat_map(|cy| (cx0..=cx1).map(move |cx| (cx, cy)))
                .map(|(cx, cy)| {
                    IRect::new(cx * CHUNK, cy * CHUNK, (cx + 1) * CHUNK, (cy + 1) * CHUNK)
                        .intersect(&total)
                })
                .filter(|r| !r.is_empty() && !r.expand(margin).intersect(&shape_rect).is_empty())
                .collect()
        };
        let job = |c: &IRect| -> Option<(IRect, Vec<f32>)> {
            let e = c.expand(margin);
            let (cov, fill, oc) = Self::render_shape(&p, &e)?;
            let data = raster::composite(
                &p.fx,
                e.width() as usize,
                e.height() as usize,
                &cov,
                &fill,
                oc.as_deref(),
                (
                    margin as usize,
                    margin as usize,
                    c.width() as usize,
                    c.height() as usize,
                ),
            );
            Some((*c, data))
        };
        #[cfg(not(target_arch = "wasm32"))]
        let results: Vec<(IRect, Vec<f32>)> = {
            use rayon::prelude::*;
            chunks.par_iter().filter_map(job).collect()
        };
        #[cfg(target_arch = "wasm32")]
        let results: Vec<(IRect, Vec<f32>)> = chunks.iter().filter_map(job).collect();

        let mut grid = TileGrid::rgba(PixelFormat::F16);
        for (r, d) in results {
            grid.write_rect(&r, &d);
        }
        if grid.tiles.is_empty() {
            None
        } else {
            Some(grid)
        }
    }

    /// Document-space axis-aligned bounds `[x0, y0, x1, y1]` of the rendered result,
    /// effects included. Empty text yields a zero-size box at `position`.
    pub fn bounds(&self, layer: &TextLayer) -> [f32; 4] {
        match self.prepare(layer, 1.0) {
            Some(p) => {
                let e = p.fx.extents();
                [
                    p.bbox[0] - e[0],
                    p.bbox[1] - e[1],
                    p.bbox[2] + e[2],
                    p.bbox[3] + e[3],
                ]
            }
            None => [
                layer.position[0],
                layer.position[1],
                layer.position[0],
                layer.position[1],
            ],
        }
    }

    /// Text block size `[width, height]` in document pixels (unrotated layout box).
    pub fn block_size(&self, layer: &TextLayer) -> [f32; 2] {
        let l = self.layout(layer);
        [l.width, l.height]
    }

    /// Caret geometry for on-canvas editing: for each char index `0..=char_count`
    /// (the last entry is the end of the text), `[x, y_top, y_bottom]` in document space.
    /// Positions are those of the flat layout: rotation, warp and 3D are NOT applied
    /// (apply `rotation` around `position` yourself if needed). A `'\n'` char's caret is
    /// at the end of its line.
    pub fn caret_positions(&self, layer: &TextLayer) -> Vec<[f32; 3]> {
        let l = self.layout(layer);
        let (px, py) = (layer.position[0], layer.position[1]);
        if l.carets.is_empty() {
            let h = layer.size * layer.line_height;
            return vec![[px, py, py + h]; layer.text.chars().count() + 1];
        }
        l.carets
            .iter()
            .map(|c| [c[0] + px, c[1] + py, c[2] + py])
            .collect()
    }
}

#[cfg(test)]
mod tests;
