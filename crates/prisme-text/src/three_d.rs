//! 3D extruded text wrapped onto a parametric sphere, rendered with a CPU z-buffer
//! triangle rasterizer (2×2 supersampling, flat colours, no lighting).

use crate::geom::{P2, signed_area};
use lyon_tessellation::math::point;
use lyon_tessellation::path::Path as LyonPath;
use lyon_tessellation::{BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers};
use prisme_core::geom::IRect;
use prisme_core::layer::Text3D;

type V3 = [f32; 3];

/// Flat paint: solid colour or a 256-entry LUT indexed by the interpolated parameter.
pub(crate) enum Paint {
    Solid([f32; 4]),
    Lut(Vec<[f32; 4]>),
}

impl Paint {
    #[inline]
    fn at(&self, t: f32) -> [f32; 4] {
        let c = match self {
            Paint::Solid(c) => *c,
            Paint::Lut(l) => l[((t.clamp(0.0, 1.0) * (l.len() - 1) as f32).round()) as usize],
        };
        let a = c[3].clamp(0.0, 1.0);
        [c[0] * a, c[1] * a, c[2] * a, a]
    }
}

#[derive(Clone, Copy)]
struct Vert {
    /// Level-space screen position.
    x: f32,
    y: f32,
    /// Depth key, larger = closer.
    key: f32,
    /// 1/w for perspective-correct attribute interpolation.
    invw: f32,
    t: f32,
}

struct Sphere {
    /// Level-space centre.
    c: P2,
    /// level px per document px
    s: f32,
    r: f32,
    persp: Option<(f32, f32)>,
    color: [f32; 4],
}

pub(crate) struct Mesh {
    verts: Vec<Vert>,
    tris: Vec<([u32; 3], u8)>,
    fill: Paint,
    side: Paint,
    sphere: Option<Sphere>,
    /// Level-space bounds of everything drawn.
    pub bbox: [f32; 4],
}

pub(crate) struct Params<'a> {
    pub cfg: &'a Text3D,
    /// Text block size (document px).
    pub block: [f32; 2],
    /// Text block top-left (document) and rotation (radians) around it.
    pub position: P2,
    pub rotation: f32,
    /// Level scale (2^-mip).
    pub scale: f32,
    pub fill: Paint,
    /// Text-space → fill gradient parameter (affine: t = a·x + b·y + c).
    pub fill_t: [f32; 3],
    pub side: Paint,
    /// Maximum segment length in text space (for sphere curvature).
    pub max_len: f32,
}

fn rot_matrix(r: [f32; 3]) -> [[f32; 3]; 3] {
    let (sy, cy) = r[0].sin_cos(); // yaw around Y
    let (sp, cp) = r[1].sin_cos(); // pitch around X
    let (sr, cr) = r[2].sin_cos(); // roll around Z
    let ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
    let rx = [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]];
    let rz = [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]];
    let mul = |a: [[f32; 3]; 3], b: [[f32; 3]; 3]| -> [[f32; 3]; 3] {
        std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
    };
    mul(rz, mul(rx, ry))
}

#[inline]
fn apply(m: &[[f32; 3]; 3], p: V3) -> V3 {
    std::array::from_fn(|i| m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2])
}

#[inline]
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

struct Mapper {
    spherical: bool,
    r: f32,
    lon0: f32,
    lat0: f32,
    rot: [[f32; 3]; 3],
    centre: P2,
    /// (camera distance f, k = f - z_ref) when perspective.
    persp: Option<(f32, f32)>,
    /// Document → level: level = s · (C + R2(rotation)·proj)
    c_doc: P2,
    cos2: f32,
    sin2: f32,
    s: f32,
}

impl Mapper {
    /// Text point (u, v relative to the block centre) at extrusion `e` → view space.
    fn view(&self, uv: P2, e: f32) -> V3 {
        let (u, v) = (uv[0] - self.centre[0], uv[1] - self.centre[1]);
        let p = if self.spherical {
            let lon = self.lon0 + u / self.r;
            let lat = self.lat0 - v / self.r;
            let (sl, cl) = lon.sin_cos();
            let (sa, ca) = lat.sin_cos();
            let rr = self.r + e;
            [ca * sl * rr, -sa * rr, ca * cl * rr]
        } else {
            [u, v, e]
        };
        apply(&self.rot, p)
    }

    fn project(&self, p: V3) -> Vert {
        let (xy, key, invw) = match self.persp {
            Some((f, k)) => {
                let w = (f - p[2]).max(f * 1e-3);
                ([p[0] * k / w, p[1] * k / w], 1.0 / w, 1.0 / w)
            }
            None => ([p[0], p[1]], p[2], 1.0),
        };
        let rx = self.cos2 * xy[0] - self.sin2 * xy[1];
        let ry = self.sin2 * xy[0] + self.cos2 * xy[1];
        Vert {
            x: self.s * (self.c_doc[0] + rx),
            y: self.s * (self.c_doc[1] + ry),
            key,
            invw,
            t: 0.0,
        }
    }

    fn eye_dir(&self, p: V3) -> V3 {
        match self.persp {
            Some((f, _)) => sub([0.0, 0.0, f], p),
            None => [0.0, 0.0, 1.0],
        }
    }
}

/// Tessellate a glyph's contours (non-zero fill) into triangles.
fn tessellate(contours: &[Vec<P2>], tol: f32) -> (Vec<P2>, Vec<u32>) {
    let mut b = LyonPath::builder();
    for c in contours {
        b.begin(point(c[0][0], c[0][1]));
        for p in &c[1..] {
            b.line_to(point(p[0], p[1]));
        }
        b.end(true);
    }
    let path = b.build();
    let mut geom: VertexBuffers<P2, u32> = VertexBuffers::new();
    let mut tess = FillTessellator::new();
    let opts = FillOptions::non_zero().with_tolerance(tol.max(1e-3));
    if tess
        .tessellate_path(
            &path,
            &opts,
            &mut BuffersBuilder::new(&mut geom, |v: FillVertex| v.position().to_array()),
        )
        .is_err()
    {
        return (Vec::new(), Vec::new());
    }
    (geom.vertices, geom.indices)
}

impl Mesh {
    /// Build the projected mesh from per-glyph text-space contours.
    pub(crate) fn build(glyphs: &[Vec<Vec<P2>>], p: Params) -> Mesh {
        let cfg = p.cfg;
        let spherical = cfg.radius > 1e-3;
        let r = cfg.radius.max(1e-3);
        let depth = cfg.depth.max(0.0);
        let centre = [p.block[0] * 0.5, p.block[1] * 0.5];
        let (sin2, cos2) = p.rotation.sin_cos();
        let c_doc = [
            p.position[0] + cos2 * centre[0] - sin2 * centre[1],
            p.position[1] + sin2 * centre[0] + cos2 * centre[1],
        ];
        let mut m = Mapper {
            spherical,
            r,
            lon0: cfg.longitude,
            lat0: cfg.latitude,
            rot: rot_matrix(cfg.rotation),
            centre,
            persp: None,
            c_doc,
            cos2,
            sin2,
            s: p.scale,
        };
        if cfg.perspective > 1e-4 {
            let ext = if spherical {
                r + depth
            } else {
                0.5 * (p.block[0].hypot(p.block[1])) + depth
            };
            let f = ext * (1.0 + 2.0 / cfg.perspective);
            let z_ref = m.view(centre, depth)[2];
            m.persp = Some((f, f - z_ref));
        }

        let mut verts: Vec<Vert> = Vec::new();
        let mut tris: Vec<([u32; 3], u8)> = Vec::new();
        let ft = p.fill_t;
        let fill_t = |q: P2| ft[0] * q[0] + ft[1] * q[1] + ft[2];
        let max_len = if spherical { p.max_len } else { f32::INFINITY };

        for contours in glyphs {
            if contours.is_empty() {
                continue;
            }
            let area: f32 = contours.iter().map(|c| signed_area(c)).sum();
            if area.abs() < 1e-9 {
                continue;
            }
            let sg = area.signum();

            // Caps.
            let (cv, ci) = tessellate(contours, 0.01);
            for tri in ci.chunks_exact(3) {
                let (a, b, c) = (
                    cv[tri[0] as usize],
                    cv[tri[1] as usize],
                    cv[tri[2] as usize],
                );
                let el = |x: P2, y: P2| (x[0] - y[0]).hypot(x[1] - y[1]);
                let longest = el(a, b).max(el(b, c)).max(el(c, a));
                let n = if max_len.is_finite() {
                    ((longest / max_len).ceil() as usize).clamp(1, 64)
                } else {
                    1
                };
                // Uniform n×n subdivision in barycentric space.
                let bary = |i: usize, j: usize| -> P2 {
                    let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
                    let w = 1.0 - u - v;
                    [
                        w * a[0] + u * b[0] + v * c[0],
                        w * a[1] + u * b[1] + v * c[1],
                    ]
                };
                for i in 0..n {
                    for j in 0..n - i {
                        let mut subs = vec![[bary(i, j), bary(i + 1, j), bary(i, j + 1)]];
                        if i + j + 1 < n {
                            subs.push([bary(i + 1, j), bary(i + 1, j + 1), bary(i, j + 1)]);
                        }
                        for st in subs {
                            let cen = [
                                (st[0][0] + st[1][0] + st[2][0]) / 3.0,
                                (st[0][1] + st[1][1] + st[2][1]) / 3.0,
                            ];
                            for (e, mat, sign) in [(depth, 0u8, 1.0f32), (0.0, 0u8, -1.0f32)] {
                                // Visibility: outward normal against the eye direction.
                                let pc = m.view(cen, e);
                                let normal = if spherical {
                                    sub(m.view(cen, e + 1.0), pc)
                                } else {
                                    apply(&m.rot, [0.0, 0.0, 1.0])
                                };
                                if sign * dot(normal, m.eye_dir(pc)) <= 0.0 {
                                    continue;
                                }
                                let base = verts.len() as u32;
                                for q in st {
                                    let mut v = m.project(m.view(q, e));
                                    v.t = fill_t(q);
                                    verts.push(v);
                                }
                                tris.push(([base, base + 1, base + 2], mat));
                            }
                        }
                    }
                }
            }

            // Side walls.
            if depth > 0.0 {
                for c in contours {
                    let nc = c.len();
                    for i in 0..nc {
                        let (a, b) = (c[i], c[(i + 1) % nc]);
                        let d = [b[0] - a[0], b[1] - a[1]];
                        let l = d[0].hypot(d[1]);
                        if l < 1e-6 {
                            continue;
                        }
                        let out2 = [sg * d[1] / l, -sg * d[0] / l];
                        let mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
                        let pm = m.view(mid, depth * 0.5);
                        let eps = 0.01 * l.max(0.1);
                        let normal = sub(
                            m.view(
                                [mid[0] + out2[0] * eps, mid[1] + out2[1] * eps],
                                depth * 0.5,
                            ),
                            pm,
                        );
                        if dot(normal, m.eye_dir(pm)) <= 0.0 {
                            continue;
                        }
                        let base = verts.len() as u32;
                        for (q, e, t) in [
                            (a, depth, 0.0),
                            (b, depth, 0.0),
                            (b, 0.0, 1.0),
                            (a, 0.0, 1.0),
                        ] {
                            let mut v = m.project(m.view(q, e));
                            v.t = t;
                            verts.push(v);
                        }
                        tris.push(([base, base + 1, base + 2], 1));
                        tris.push(([base, base + 2, base + 3], 1));
                    }
                }
            }
        }

        let mut bbox = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        for v in &verts {
            bbox[0] = bbox[0].min(v.x);
            bbox[1] = bbox[1].min(v.y);
            bbox[2] = bbox[2].max(v.x);
            bbox[3] = bbox[3].max(v.y);
        }
        let sphere = (cfg.show_sphere && spherical).then(|| {
            let c = [c_doc[0] * p.scale, c_doc[1] * p.scale];
            let rp = match m.persp {
                Some((f, k)) => k * r / (f * f - r * r).max(1e-6).sqrt(),
                None => r,
            } * p.scale;
            bbox = [
                bbox[0].min(c[0] - rp),
                bbox[1].min(c[1] - rp),
                bbox[2].max(c[0] + rp),
                bbox[3].max(c[1] + rp),
            ];
            Sphere {
                c,
                s: p.scale,
                r,
                persp: m.persp,
                color: cfg.sphere_color,
            }
        });
        Mesh {
            verts,
            tris,
            fill: p.fill,
            side: p.side,
            sphere,
            bbox,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.tris.is_empty() && self.sphere.is_none()
    }

    /// Render into region `e` (level px). Returns (coverage, premultiplied colour).
    pub(crate) fn render(&self, e: &IRect) -> (Vec<f32>, Vec<[f32; 4]>) {
        let (w, h) = (e.width() as usize, e.height() as usize);
        let (sw, sh) = (w * 2, h * 2);
        let mut depth = vec![f32::NEG_INFINITY; sw * sh];
        let mut color = vec![[0.0f32; 4]; sw * sh];
        // sample (i, j) sits at level (x0 + (i + 0.5)/2, y0 + (j + 0.5)/2)
        let to_s = |v: &Vert| -> [f32; 2] {
            [
                (v.x - e.x0 as f32) * 2.0 - 0.5,
                (v.y - e.y0 as f32) * 2.0 - 0.5,
            ]
        };
        const BAND: usize = 32;
        let nb = sh.div_ceil(BAND).max(1);
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); nb];
        for (ti, (idx, _)) in self.tris.iter().enumerate() {
            let ps = idx.map(|i| to_s(&self.verts[i as usize]));
            let (ymin, ymax) = (
                ps.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min),
                ps.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max),
            );
            let (xmin, xmax) = (
                ps.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min),
                ps.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max),
            );
            if xmax < 0.0 || ymax < 0.0 || xmin > (sw - 1) as f32 || ymin > (sh - 1) as f32 {
                continue;
            }
            let b0 = (ymin.max(0.0) as usize) / BAND;
            let b1 = ((ymax.min((sh - 1) as f32)) as usize) / BAND;
            for b in buckets.iter_mut().take(b1.min(nb - 1) + 1).skip(b0) {
                b.push(ti as u32);
            }
        }

        let band_job = |bi: usize, dep: &mut [f32], col: &mut [[f32; 4]]| {
            let y_base = bi * BAND;
            let rows = dep.len() / sw;
            if let Some(sp) = &self.sphere {
                let sc = sp.color;
                let a = sc[3].clamp(0.0, 1.0);
                let prem = [sc[0] * a, sc[1] * a, sc[2] * a, a];
                for ry in 0..rows {
                    let ly = e.y0 as f32 + ((y_base + ry) as f32 + 0.5) * 0.5;
                    for x in 0..sw {
                        let lx = e.x0 as f32 + (x as f32 + 0.5) * 0.5;
                        let (dx, dy) = ((lx - sp.c[0]) / sp.s, (ly - sp.c[1]) / sp.s);
                        let q2 = dx * dx + dy * dy;
                        let key = match sp.persp {
                            None => {
                                if q2 >= sp.r * sp.r {
                                    continue;
                                }
                                (sp.r * sp.r - q2).sqrt()
                            }
                            Some((f, k)) => {
                                let q = q2 / (k * k);
                                let disc = (q + 1.0) * sp.r * sp.r - q * f * f;
                                if disc < 0.0 {
                                    continue;
                                }
                                let z = (q * f + disc.sqrt()) / (q + 1.0);
                                1.0 / (f - z).max(1e-6)
                            }
                        };
                        let i = ry * sw + x;
                        dep[i] = key;
                        col[i] = prem;
                    }
                }
            }
            for &ti in &buckets[bi] {
                let (idx, mat) = self.tris[ti as usize];
                let vs = idx.map(|i| self.verts[i as usize]);
                let ps = vs.map(|v| to_s(&v));
                let [(x0, y0), (x1, y1), (x2, y2)] = ps.map(|p| (p[0], p[1]));
                let area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
                if area.abs() < 1e-9 {
                    continue;
                }
                let inv = 1.0 / area;
                let xa = x0.min(x1).min(x2).ceil().max(0.0) as usize;
                let xb = x0.max(x1).max(x2).floor().min((sw - 1) as f32);
                let ya = (y0.min(y1).min(y2).ceil().max(y_base as f32)) as usize;
                let yb = y0.max(y1).max(y2).floor().min((y_base + rows - 1) as f32);
                if xb < 0.0 || yb < ya as f32 {
                    continue;
                }
                let (xb, yb) = (xb as usize, yb as usize);
                let paint = if mat == 0 { &self.fill } else { &self.side };
                let solid = match paint {
                    Paint::Solid(_) => Some(paint.at(0.0)),
                    _ => None,
                };
                const EPS: f32 = -1e-5;
                for y in ya..=yb {
                    let py = y as f32;
                    let row = (y - y_base) * sw;
                    for x in xa..=xb {
                        let px = x as f32;
                        let w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) * inv;
                        let w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) * inv;
                        let w2 = 1.0 - w0 - w1;
                        if w0 < EPS || w1 < EPS || w2 < EPS {
                            continue;
                        }
                        let key = w0 * vs[0].key + w1 * vs[1].key + w2 * vs[2].key;
                        let i = row + x;
                        if key > dep[i] {
                            dep[i] = key;
                            col[i] = match solid {
                                Some(c) => c,
                                None => {
                                    let iw = w0 * vs[0].invw + w1 * vs[1].invw + w2 * vs[2].invw;
                                    let t = (w0 * vs[0].t * vs[0].invw
                                        + w1 * vs[1].t * vs[1].invw
                                        + w2 * vs[2].t * vs[2].invw)
                                        / iw.max(1e-12);
                                    paint.at(t)
                                }
                            };
                        }
                    }
                }
            }
        };

        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            depth
                .par_chunks_mut(BAND * sw)
                .zip(color.par_chunks_mut(BAND * sw))
                .enumerate()
                .for_each(|(bi, (d, c))| band_job(bi, d, c));
        }
        #[cfg(target_arch = "wasm32")]
        for (bi, (d, c)) in depth
            .chunks_mut(BAND * sw)
            .zip(color.chunks_mut(BAND * sw))
            .enumerate()
        {
            band_job(bi, d, c);
        }

        // Resolve 2×2 samples.
        let mut cov = vec![0.0f32; w * h];
        let mut fill = vec![[0.0f32; 4]; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0.0f32; 4];
                for (sx, sy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let c = color[(2 * y + sy) * sw + 2 * x + sx];
                    for k in 0..4 {
                        acc[k] += c[k];
                    }
                }
                let a = acc.map(|v| v * 0.25);
                cov[y * w + x] = a[3];
                fill[y * w + x] = a;
            }
        }
        (cov, fill)
    }
}
