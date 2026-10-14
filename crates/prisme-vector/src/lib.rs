//! SVG layer rendering for Prisme.
//!
//! Vector layers keep their SVG source until render time. [`VectorRenderer`] parses the
//! SVG once (cached per `Arc<String>`), and rasterizes it with `resvg` at any mip level
//! into a sparse [`TileGrid`] in level-space pixel coordinates (document coordinate ×
//! 2^-mip). Output is straight-alpha, scene-linear RGBA in the working space.
//!
//! Large outputs are rendered in independent tile-aligned chunks (parallel with rayon on
//! native targets), so memory stays bounded whatever the zoom level.

use prisme_core::geom::{Affine2, IRect};
use prisme_core::layer::VectorLayer;
use prisme_core::{PixelFormat, TileGrid};
use resvg::tiny_skia;
use resvg::usvg;
use resvg::usvg::fontdb;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Side of the square chunks the output is rendered in (multiple of the tile size).
const CHUNK: i32 = 1024;
/// Hard cap on the rendered level-space area (pixels); beyond this the region is clipped
/// around the origin to avoid pathological allocations.
const MAX_LEVEL_DIM: i32 = 1 << 17;
/// Max number of parsed trees kept in the cache.
const CACHE_CAP: usize = 64;

struct CacheEntry {
    src: Weak<String>,
    tree: Arc<usvg::Tree>,
    has_text: bool,
}

/// Parsed-SVG cache + font database. Cheap to share (`Send + Sync`).
pub struct VectorRenderer {
    fonts: Mutex<Arc<fontdb::Database>>,
    system_loaded: Mutex<bool>,
    cache: Mutex<HashMap<usize, CacheEntry>>,
}

impl Default for VectorRenderer {
    fn default() -> Self {
        Self::new()
    }
}

fn bundled_fontdb() -> fontdb::Database {
    let mut db = fontdb::Database::new();
    db.load_font_data(epaint_default_fonts::UBUNTU_LIGHT.to_vec());
    db.load_font_data(epaint_default_fonts::HACK_REGULAR.to_vec());
    db.set_sans_serif_family("Ubuntu");
    db.set_serif_family("Ubuntu");
    db.set_cursive_family("Ubuntu");
    db.set_fantasy_family("Ubuntu");
    db.set_monospace_family("Hack");
    db
}

#[cfg(not(target_arch = "wasm32"))]
fn system_fontdb() -> Arc<fontdb::Database> {
    static DB: OnceLock<Arc<fontdb::Database>> = OnceLock::new();
    DB.get_or_init(|| {
        let mut db = bundled_fontdb();
        db.load_system_fonts();
        for fam in [
            "Helvetica",
            "Arial",
            "Segoe UI",
            "DejaVu Sans",
            "Noto Sans",
            "Liberation Sans",
        ] {
            if has_family(&db, fam) {
                db.set_sans_serif_family(fam);
                break;
            }
        }
        for fam in [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
            "Noto Serif",
            "Liberation Serif",
        ] {
            if has_family(&db, fam) {
                db.set_serif_family(fam);
                break;
            }
        }
        Arc::new(db)
    })
    .clone()
}

#[cfg(not(target_arch = "wasm32"))]
fn has_family(db: &fontdb::Database, fam: &str) -> bool {
    db.faces()
        .any(|f| f.families.iter().any(|(n, _)| n.eq_ignore_ascii_case(fam)))
}

#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
fn system_fontdb() -> Arc<fontdb::Database> {
    static DB: OnceLock<Arc<fontdb::Database>> = OnceLock::new();
    DB.get_or_init(|| Arc::new(bundled_fontdb())).clone()
}

fn to_ts(a: &Affine2) -> tiny_skia::Transform {
    let m = a.m;
    tiny_skia::Transform::from_row(m[0], m[3], m[1], m[4], m[2], m[5])
}

/// Natural size of the placed SVG in user units (falls back to the parsed size when unset).
fn layer_size(layer: &VectorLayer, tree: Option<&usvg::Tree>) -> [f32; 2] {
    if layer.size[0] > 0.0 && layer.size[1] > 0.0 {
        return layer.size;
    }
    match tree {
        Some(t) => [t.size().width(), t.size().height()],
        None => svg_size(&layer.svg).unwrap_or([0.0, 0.0]),
    }
}

fn transformed_aabb(t: &Affine2, size: [f32; 2]) -> [f32; 4] {
    let pts = [
        [0.0, 0.0],
        [size[0], 0.0],
        [size[0], size[1]],
        [0.0, size[1]],
    ]
    .map(|p| t.apply(p));
    let mut r = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    for p in pts {
        r[0] = r[0].min(p[0]);
        r[1] = r[1].min(p[1]);
        r[2] = r[2].max(p[0]);
        r[3] = r[3].max(p[1]);
    }
    r
}

/// sRGB-encoded 8-bit → linear lookup.
fn srgb_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| std::array::from_fn(|i| prisme_color::srgb_to_linear(i as f32 / 255.0)))
}

/// Convert a premultiplied sRGB8 pixmap to straight-alpha linear working RGBA f32.
fn pixmap_to_working(px: &[u8], out: &mut [f32]) {
    let lut = srgb_lut();
    let conv = prisme_color::srgb();
    for (s, d) in px.chunks_exact(4).zip(out.chunks_exact_mut(4)) {
        let a = s[3];
        if a == 0 {
            d.copy_from_slice(&[0.0; 4]);
            continue;
        }
        let lin = if a == 255 {
            [lut[s[0] as usize], lut[s[1] as usize], lut[s[2] as usize]]
        } else {
            let inv = 1.0 / a as f32;
            let f = |c: u8| prisme_color::srgb_to_linear((c as f32 * inv).min(1.0));
            [f(s[0]), f(s[1]), f(s[2])]
        };
        let w = conv.to_working(lin);
        d[0] = w[0];
        d[1] = w[1];
        d[2] = w[2];
        d[3] = a as f32 / 255.0;
    }
}

impl VectorRenderer {
    pub fn new() -> Self {
        Self {
            fonts: Mutex::new(Arc::new(bundled_fontdb())),
            system_loaded: Mutex::new(false),
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Replace the font database used for `<text>` elements (e.g. to share one that
    /// already holds system and user fonts). Clears the parse cache.
    pub fn set_fontdb(&self, db: Arc<fontdb::Database>) {
        *self.fonts.lock().unwrap() = db;
        *self.system_loaded.lock().unwrap() = true;
        self.cache.lock().unwrap().clear();
    }

    /// Ensure system fonts are available for SVG text (native only; no-op on wasm).
    fn ensure_system_fonts(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut loaded = self.system_loaded.lock().unwrap();
            if !*loaded {
                *self.fonts.lock().unwrap() = system_fontdb();
                *loaded = true;
            }
        }
    }

    fn tree(&self, svg: &Arc<String>) -> Option<Arc<usvg::Tree>> {
        let key = Arc::as_ptr(svg) as usize;
        {
            let cache = self.cache.lock().unwrap();
            if let Some(e) = cache.get(&key)
                && e.src.upgrade().is_some_and(|s| Arc::ptr_eq(&s, svg))
            {
                return Some(e.tree.clone());
            }
        }
        let has_text = svg.contains("<text") || svg.contains(":text");
        if has_text {
            self.ensure_system_fonts();
        }
        let opt = usvg::Options {
            fontdb: self.fonts.lock().unwrap().clone(),
            ..Default::default()
        };
        let tree = match usvg::Tree::from_str(svg, &opt) {
            Ok(t) => Arc::new(t),
            Err(e) => {
                log::warn!("SVG parse failed: {e}");
                return None;
            }
        };
        let mut cache = self.cache.lock().unwrap();
        cache.retain(|_, e| e.src.strong_count() > 0);
        if cache.len() >= CACHE_CAP {
            // Drop text-free entries first; otherwise everything (simple policy).
            cache.retain(|_, e| e.has_text);
            if cache.len() >= CACHE_CAP {
                cache.clear();
            }
        }
        cache.insert(
            key,
            CacheEntry {
                src: Arc::downgrade(svg),
                tree: tree.clone(),
                has_text,
            },
        );
        Some(tree)
    }

    /// Rasterize at mip level (level-space pixel coordinates, straight alpha, linear
    /// working space, F16 tiles). Returns `None` when the SVG fails to parse or is empty.
    pub fn rasterize(&self, layer: &VectorLayer, mip: u32) -> Option<TileGrid> {
        let tree = self.tree(&layer.svg)?;
        let size = layer_size(layer, Some(&tree));
        if size[0] <= 0.0 || size[1] <= 0.0 {
            return None;
        }
        let s = 0.5f32.powi(mip as i32);
        let m = Affine2::scale(s, s).mul(&layer.transform);
        let b = transformed_aabb(&m, size);
        if !b.iter().all(|v| v.is_finite()) {
            return None;
        }
        let lim = MAX_LEVEL_DIM as f32;
        let rect = IRect::from_f32(b[0].max(-lim), b[1].max(-lim), b[2].min(lim), b[3].min(lim));
        if rect.is_empty() {
            return None;
        }
        let chunks: Vec<IRect> = {
            let (cx0, cy0) = (rect.x0.div_euclid(CHUNK), rect.y0.div_euclid(CHUNK));
            let (cx1, cy1) = (
                (rect.x1 - 1).div_euclid(CHUNK),
                (rect.y1 - 1).div_euclid(CHUNK),
            );
            (cy0..=cy1)
                .flat_map(|cy| (cx0..=cx1).map(move |cx| (cx, cy)))
                .map(|(cx, cy)| {
                    IRect::new(cx * CHUNK, cy * CHUNK, (cx + 1) * CHUNK, (cy + 1) * CHUNK)
                        .intersect(&rect)
                })
                .filter(|r| !r.is_empty())
                .collect()
        };
        let render_chunk = |r: &IRect| -> Option<(IRect, Vec<f32>)> {
            let mut pm = tiny_skia::Pixmap::new(r.width() as u32, r.height() as u32)?;
            let t = Affine2::translate(-r.x0 as f32, -r.y0 as f32).mul(&m);
            resvg::render(&tree, to_ts(&t), &mut pm.as_mut());
            if pm.data().chunks_exact(4).all(|p| p[3] == 0) {
                return None;
            }
            let mut out = vec![0.0f32; r.area() as usize * 4];
            pixmap_to_working(pm.data(), &mut out);
            Some((*r, out))
        };
        #[cfg(not(target_arch = "wasm32"))]
        let results: Vec<(IRect, Vec<f32>)> = {
            use rayon::prelude::*;
            chunks.par_iter().filter_map(render_chunk).collect()
        };
        #[cfg(target_arch = "wasm32")]
        let results: Vec<(IRect, Vec<f32>)> = chunks.iter().filter_map(render_chunk).collect();

        let mut grid = TileGrid::rgba(PixelFormat::F16);
        for (r, data) in results {
            grid.write_rect(&r, &data);
        }
        if grid.tiles.is_empty() {
            None
        } else {
            Some(grid)
        }
    }

    /// Document-space bounds `[x0, y0, x1, y1]` of the placed SVG (for UI handles).
    pub fn bounds(layer: &VectorLayer) -> [f32; 4] {
        transformed_aabb(&layer.transform, layer_size(layer, None))
    }
}

/// Natural size `[width, height]` of an SVG document in user units.
pub fn svg_size(svg: &str) -> Option<[f32; 2]> {
    // Text is irrelevant for the size, so parse with an empty font database.
    let opt = usvg::Options {
        fontdb: Arc::new(fontdb::Database::new()),
        ..Default::default()
    };
    let tree = usvg::Tree::from_str(svg, &opt).ok()?;
    Some([tree.size().width(), tree.size().height()])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(svg: &str) -> VectorLayer {
        let size = svg_size(svg).unwrap();
        VectorLayer {
            svg: Arc::new(svg.to_string()),
            size,
            transform: Affine2::translate(10.0, 20.0),
        }
    }

    const RED: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><rect width="100" height="50" fill="#ff0000"/></svg>"##;

    #[test]
    fn size_parsing() {
        assert_eq!(svg_size(RED), Some([100.0, 50.0]));
        assert_eq!(svg_size("not svg"), None);
    }

    #[test]
    fn red_rect_mip0() {
        let r = VectorRenderer::new();
        let l = layer(RED);
        let g = r.rasterize(&l, 0).unwrap();
        assert_eq!(g.content_bounds(), IRect::new(10, 20, 110, 70));
        let exp = prisme_color::srgb().to_working([1.0, 0.0, 0.0]);
        let p = g.get_pixel(50, 40);
        for k in 0..3 {
            assert!((p[k] - exp[k]).abs() < 2e-3, "{p:?} vs {exp:?}");
        }
        assert!((p[3] - 1.0).abs() < 1e-3);
        assert_eq!(VectorRenderer::bounds(&l), [10.0, 20.0, 110.0, 70.0]);
    }

    #[test]
    fn red_rect_mip1_and_cache() {
        let r = VectorRenderer::new();
        let l = layer(RED);
        let g = r.rasterize(&l, 1).unwrap();
        assert_eq!(g.content_bounds(), IRect::new(5, 10, 55, 35));
        // Second call hits the parse cache.
        let g2 = r.rasterize(&l, 1).unwrap();
        assert_eq!(g2.content_bounds(), g.content_bounds());
        assert_eq!(r.cache.lock().unwrap().len(), 1);
    }

    #[test]
    fn huge_output_is_chunked() {
        let r = VectorRenderer::new();
        let mut l = layer(RED);
        l.transform = Affine2::scale(30.0, 30.0); // 3000 x 1500
        let g = r.rasterize(&l, 0).unwrap();
        assert_eq!(g.content_bounds(), IRect::new(0, 0, 3000, 1500));
        assert!(g.get_pixel(2999, 1499)[3] > 0.99);
    }

    #[test]
    fn half_transparent_unpremultiplies() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10" fill="#ffffff" fill-opacity="0.5"/></svg>"##;
        let r = VectorRenderer::new();
        let g = r.rasterize(&layer(svg), 0).unwrap();
        let p = g.get_pixel(15, 25);
        assert!((p[3] - 0.5).abs() < 0.01, "{p:?}");
        assert!((p[0] - 1.0).abs() < 0.02, "{p:?}");
    }
}
