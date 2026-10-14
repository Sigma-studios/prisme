//! Liquify: brush dabs paint a displacement field that warps a layer.
//!
//! The field is a *backward* map over the session's working area: output pixel `p` samples
//! the layer snapshot taken when the session started at `p + d(p)` (edge-extended inside the
//! working area), then mixes with the snapshot by the selection. Nodes sit every `step`
//! document px (at pixel centres when `step == 1`) and hold `(dx, dy, freeze, 0)`; the field
//! is sampled bilinearly. Painting the field is cheap and order dependent, so it runs here
//! on the CPU (which also makes per-stroke undo trivial); the host uploads changed nodes and
//! warps the layer on the GPU every frame (`prisme_gpu::stroke`, `k_liquify`), and bakes it
//! with a GPU render + readback. [`warp_rect`] is the CPU reference and fallback.

use crate::stroke::PaintTarget;
use prisme_core::tile::TileGrid;
use prisme_core::IRect;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Largest working area side (document px; GPU texture limits).
pub const MAX_SIDE: i32 = 8192;
/// Largest field side (nodes); bigger areas use coarser nodes.
pub const MAX_NODES: i32 = 2048;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum LiquifyMode {
    /// Push pixels along the drag.
    #[default]
    Forward,
    /// Gradually return the displacement towards zero.
    Reconstruct,
    /// Relax (blur) the displacement.
    Smooth,
    TwirlCw,
    TwirlCcw,
    Pucker,
    Bloat,
    /// Paint protected areas.
    Freeze,
    /// Erase the freeze mask.
    Thaw,
}

impl LiquifyMode {
    pub const ALL: [LiquifyMode; 9] = [
        LiquifyMode::Forward,
        LiquifyMode::Reconstruct,
        LiquifyMode::Smooth,
        LiquifyMode::TwirlCw,
        LiquifyMode::TwirlCcw,
        LiquifyMode::Pucker,
        LiquifyMode::Bloat,
        LiquifyMode::Freeze,
        LiquifyMode::Thaw,
    ];

    pub fn name(self) -> &'static str {
        match self {
            LiquifyMode::Forward => "Forward Warp",
            LiquifyMode::Reconstruct => "Reconstruct",
            LiquifyMode::Smooth => "Smooth",
            LiquifyMode::TwirlCw => "Twirl Clockwise",
            LiquifyMode::TwirlCcw => "Twirl Counter-clockwise",
            LiquifyMode::Pucker => "Pucker",
            LiquifyMode::Bloat => "Bloat",
            LiquifyMode::Freeze => "Freeze Mask",
            LiquifyMode::Thaw => "Thaw Mask",
        }
    }

    /// Script / settings id.
    pub fn id(self) -> &'static str {
        match self {
            LiquifyMode::Forward => "forward",
            LiquifyMode::Reconstruct => "reconstruct",
            LiquifyMode::Smooth => "smooth",
            LiquifyMode::TwirlCw => "twirl-cw",
            LiquifyMode::TwirlCcw => "twirl-ccw",
            LiquifyMode::Pucker => "pucker",
            LiquifyMode::Bloat => "bloat",
            LiquifyMode::Freeze => "freeze",
            LiquifyMode::Thaw => "thaw",
        }
    }

    /// Keeps acting while the pointer is held still (rate driven).
    pub fn stationary(self) -> bool {
        !matches!(self, LiquifyMode::Forward | LiquifyMode::Freeze | LiquifyMode::Thaw)
    }
}

/// Liquify tool settings (options bar).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LiquifyOptions {
    pub mode: LiquifyMode,
    /// Brush diameter (document px).
    pub size: f32,
    /// Strength in [0,1] (pen pressure scales it).
    pub pressure: f32,
    /// Edge falloff in [0,1]: 0 = very soft, 1 = firm.
    pub density: f32,
    /// Speed of the stationary modes (bloat, pucker, twirl, smooth, reconstruct) in [0,1].
    pub rate: f32,
    /// Tint frozen areas in the preview.
    pub show_mask: bool,
}

impl Default for LiquifyOptions {
    fn default() -> Self {
        Self { mode: LiquifyMode::Forward, size: 150.0, pressure: 0.8, density: 0.5, rate: 0.5, show_mask: true }
    }
}

/// Session start state handed to the GPU host.
#[derive(Clone, Debug)]
pub struct LiquifySpec {
    pub target: PaintTarget,
    /// Layer pixels at session start (the working copy keeps these until the session lands).
    pub base: Arc<TileGrid>,
    pub selection: Option<Arc<TileGrid>>,
    /// Working area (document px).
    pub rect: IRect,
    /// Document px per field node.
    pub step: i32,
    /// Field size in nodes.
    pub nodes: [i32; 2],
}

/// The displacement + freeze field (see the module docs).
#[derive(Clone, Debug)]
pub struct LiquifyField {
    pub rect: IRect,
    pub step: i32,
    pub w: i32,
    pub h: i32,
    /// Row-major `(dx, dy, freeze, 0)`.
    pub data: Vec<[f32; 4]>,
}

/// Working area for a layer: the document plus the layer's content, within the size limit.
pub fn working_rect(doc_bounds: IRect, content: IRect) -> IRect {
    let r = doc_bounds.union(&content);
    if r.width() <= MAX_SIDE && r.height() <= MAX_SIDE {
        return r;
    }
    let d = doc_bounds;
    IRect::new(d.x0, d.y0, d.x0 + d.width().min(MAX_SIDE), d.y0 + d.height().min(MAX_SIDE))
}

/// One brush application.
#[derive(Clone, Copy, Debug)]
pub struct LiquifyDab {
    pub mode: LiquifyMode,
    pub center: [f32; 2],
    pub radius: f32,
    /// Strength of this application (pressure × rate…), see [`LiquifyField::apply_dab`].
    pub amount: f32,
    pub density: f32,
    /// Forward warp: brush movement since the previous dab.
    pub motion: [f32; 2],
}

/// Brush weight at normalized distance `t` (1 at the centre, 0 at the rim).
pub fn falloff(t: f32, density: f32) -> f32 {
    if t >= 1.0 {
        return 0.0;
    }
    (1.0 - t * t).powf(1.0 + 3.0 * (1.0 - density.clamp(0.0, 1.0)))
}

fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t]
}

impl LiquifyField {
    pub fn new(rect: IRect) -> Self {
        let side = rect.width().max(rect.height()).max(1);
        let step = ((side + MAX_NODES - 1) / MAX_NODES).max(1);
        let (w, h) = ((rect.width() + step - 1) / step, (rect.height() + step - 1) / step);
        Self { rect, step, w: w.max(1), h: h.max(1), data: vec![[0.0; 4]; (w.max(1) * h.max(1)) as usize] }
    }

    pub fn nodes(&self) -> IRect {
        IRect::new(0, 0, self.w, self.h)
    }

    /// Document position of node `(i, j)`.
    pub fn node_pos(&self, i: i32, j: i32) -> [f32; 2] {
        let s = self.step as f32;
        [self.rect.x0 as f32 + (i as f32 + 0.5) * s, self.rect.y0 as f32 + (j as f32 + 0.5) * s]
    }

    fn at(&self, i: i32, j: i32) -> [f32; 4] {
        self.data[(j.clamp(0, self.h - 1) * self.w + i.clamp(0, self.w - 1)) as usize]
    }

    /// Bilinear field value at a document position (edge nodes extend outwards).
    pub fn sample(&self, x: f32, y: f32) -> [f32; 4] {
        let s = self.step as f32;
        let u = (x - self.rect.x0 as f32) / s - 0.5;
        let v = (y - self.rect.y0 as f32) / s - 0.5;
        let (i, j) = (u.floor(), v.floor());
        let (tx, ty) = (u - i, v - j);
        let (i, j) = (i as i32, j as i32);
        lerp4(lerp4(self.at(i, j), self.at(i + 1, j), tx), lerp4(self.at(i, j + 1), self.at(i + 1, j + 1), tx), ty)
    }

    /// Nodes whose position lies in the document rect `r` (conservative).
    pub fn nodes_in(&self, r: IRect) -> IRect {
        let s = self.step as f32;
        let f = |v: i32, o: i32| (v - o) as f32 / s - 0.5;
        IRect::new(
            f(r.x0, self.rect.x0).floor() as i32,
            f(r.y0, self.rect.y0).floor() as i32,
            f(r.x1, self.rect.x0).ceil() as i32 + 1,
            f(r.y1, self.rect.y0).ceil() as i32 + 1,
        )
        .intersect(&self.nodes())
    }

    /// Document pixels whose field value depends on `nodes` (inside the working area).
    pub fn doc_rect(&self, nodes: IRect) -> IRect {
        if nodes.is_empty() {
            return IRect::EMPTY;
        }
        let s = self.step as f32;
        let (ox, oy) = (self.rect.x0 as f32, self.rect.y0 as f32);
        IRect::from_f32(ox + (nodes.x0 as f32 - 0.5) * s, oy + (nodes.y0 as f32 - 0.5) * s, ox + (nodes.x1 as f32 + 0.5) * s, oy + (nodes.y1 as f32 + 0.5) * s)
            .expand(1)
            .intersect(&self.rect)
    }

    /// Document rect where the warp differs from identity.
    pub fn active_rect(&self) -> IRect {
        let mut r = IRect::EMPTY;
        for j in 0..self.h {
            let row = &self.data[(j * self.w) as usize..][..self.w as usize];
            let (mut x0, mut x1) = (i32::MAX, -1);
            for (i, d) in row.iter().enumerate() {
                if d[0].abs() > 1e-4 || d[1].abs() > 1e-4 {
                    x0 = x0.min(i as i32);
                    x1 = i as i32;
                }
            }
            if x1 >= 0 {
                r = r.union(&IRect::new(x0, j, x1 + 1, j + 1));
            }
        }
        self.doc_rect(r)
    }

    pub fn read(&self, nodes: IRect) -> Vec<[f32; 4]> {
        let mut out = Vec::with_capacity(nodes.area() as usize);
        for j in nodes.y0..nodes.y1 {
            out.extend_from_slice(&self.data[(j * self.w + nodes.x0) as usize..][..nodes.width() as usize]);
        }
        out
    }

    pub fn write(&mut self, nodes: IRect, v: &[[f32; 4]]) {
        let n = nodes.width() as usize;
        for (k, j) in (nodes.y0..nodes.y1).enumerate() {
            self.data[(j * self.w + nodes.x0) as usize..][..n].copy_from_slice(&v[k * n..][..n]);
        }
    }

    /// Apply one dab; returns the nodes it changed.
    ///
    /// `amount` is the forward-warp follow ratio, the bloat/pucker scale step, the twirl
    /// angle (radians), the reconstruct/smooth blend, or the freeze/thaw opacity, all times
    /// the brush weight. Displacement changes are held back by the freeze mask.
    pub fn apply_dab(&mut self, d: &LiquifyDab) -> IRect {
        let r = d.radius.max(0.5);
        let c = d.center;
        let nodes = self.nodes_in(IRect::from_f32(c[0] - r, c[1] - r, c[0] + r, c[1] + r));
        if nodes.is_empty() || d.amount <= 0.0 {
            return IRect::EMPTY;
        }
        let w = nodes.width() as usize;
        let mut out = self.read(nodes);
        let this = &*self;
        let mut rows: Vec<(i32, &mut [[f32; 4]])> = out.chunks_mut(w).enumerate().map(|(k, row)| (nodes.y0 + k as i32, row)).collect();
        crate::par_for_each(&mut rows, |(j, row)| {
            for (k, v) in row.iter_mut().enumerate() {
                let i = nodes.x0 + k as i32;
                let p = this.node_pos(i, *j);
                let o = [p[0] - c[0], p[1] - c[1]];
                let wt = falloff((o[0] * o[0] + o[1] * o[1]).sqrt() / r, d.density);
                if wt <= 0.0 {
                    continue;
                }
                let a = wt * d.amount;
                match d.mode {
                    LiquifyMode::Freeze => v[2] += (1.0 - v[2]) * a.min(1.0),
                    LiquifyMode::Thaw => v[2] *= 1.0 - a.min(1.0),
                    m => {
                        let keep = 1.0 - v[2];
                        if keep <= 0.0 {
                            continue;
                        }
                        // Backward map: the new output at p is the old output at q.
                        let resample = |q: [f32; 2]| {
                            let s = this.sample(q[0], q[1]);
                            [q[0] + s[0] - p[0], q[1] + s[1] - p[1]]
                        };
                        let n = match m {
                            LiquifyMode::Forward => resample([p[0] - a * d.motion[0], p[1] - a * d.motion[1]]),
                            LiquifyMode::Bloat => resample([c[0] + o[0] * (1.0 - a.min(0.9)), c[1] + o[1] * (1.0 - a.min(0.9))]),
                            LiquifyMode::Pucker => resample([c[0] + o[0] * (1.0 + a), c[1] + o[1] * (1.0 + a)]),
                            LiquifyMode::TwirlCw | LiquifyMode::TwirlCcw => {
                                // Content turns by +angle (clockwise on screen, y down) when sampling at -angle.
                                let ang = if m == LiquifyMode::TwirlCw { -a } else { a };
                                let (sn, cs) = ang.sin_cos();
                                resample([c[0] + o[0] * cs - o[1] * sn, c[1] + o[0] * sn + o[1] * cs])
                            }
                            LiquifyMode::Reconstruct => [v[0] * (1.0 - a.min(1.0)), v[1] * (1.0 - a.min(1.0))],
                            _ => {
                                // Smooth: blend towards the mean of a 3x3 neighbourhood.
                                let sp = (this.step as f32).max(r * 0.25);
                                let mut m = [0.0f32; 2];
                                for dy in -1..=1 {
                                    for dx in -1..=1 {
                                        let s = this.sample(p[0] + dx as f32 * sp, p[1] + dy as f32 * sp);
                                        m[0] += s[0];
                                        m[1] += s[1];
                                    }
                                }
                                let k = a.min(1.0);
                                [v[0] + (m[0] / 9.0 - v[0]) * k, v[1] + (m[1] / 9.0 - v[1]) * k]
                            }
                        };
                        v[0] += (n[0] - v[0]) * keep;
                        v[1] += (n[1] - v[1]) * keep;
                    }
                }
            }
        });
        drop(rows);
        self.write(nodes, &out);
        nodes
    }
}

fn cubic(t: f32) -> [f32; 4] {
    // Catmull-Rom (same as `transform`).
    let t2 = t * t;
    let t3 = t2 * t;
    [-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2]
}

/// Premultiplied source pixels over a rect, index-clamped (edge extend).
struct Src {
    r: IRect,
    px: Vec<[f32; 4]>,
}

impl Src {
    fn get(&self, x: i32, y: i32) -> [f32; 4] {
        let (x, y) = (x.clamp(self.r.x0, self.r.x1 - 1), y.clamp(self.r.y0, self.r.y1 - 1));
        self.px[((y - self.r.y0) * self.r.width() + x - self.r.x0) as usize]
    }

    /// Premultiplied sample at a continuous position (pixel centres at +0.5).
    fn sample(&self, x: f32, y: f32, bicubic: bool) -> [f32; 4] {
        let (fx, fy) = (x - 0.5, y - 0.5);
        let (ix, iy) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - ix, fy - iy);
        let (ix, iy) = (ix as i32, iy as i32);
        let n = [self.get(ix, iy), self.get(ix + 1, iy), self.get(ix, iy + 1), self.get(ix + 1, iy + 1)];
        if !bicubic {
            return lerp4(lerp4(n[0], n[1], tx), lerp4(n[2], n[3], tx), ty);
        }
        let (wx, wy) = (cubic(tx), cubic(ty));
        let mut acc = [0.0f32; 4];
        for (j, wyj) in wy.iter().enumerate() {
            for (i, wxi) in wx.iter().enumerate() {
                let p = self.get(ix - 1 + i as i32, iy - 1 + j as i32);
                let w = wxi * wyj;
                for c in 0..4 {
                    acc[c] += p[c] * w;
                }
            }
        }
        // No ringing: stay within the four nearest pixels.
        for c in 0..4 {
            let lo = n.iter().fold(f32::MAX, |m, p| m.min(p[c]));
            let hi = n.iter().fold(f32::MIN, |m, p| m.max(p[c]));
            acc[c] = acc[c].clamp(lo, hi);
        }
        acc
    }
}

fn premul(p: &[f32]) -> [f32; 4] {
    [p[0] * p[3], p[1] * p[3], p[2] * p[3], p[3]]
}

/// CPU reference warp of `rect` (inside `field.rect`): straight RGBA, row-major.
/// Mirrors `k_liquify` at level 0 (bilinear preview / bicubic bake).
pub fn warp_rect(base: &TileGrid, field: &LiquifyField, sel: Option<&TileGrid>, rect: IRect, bicubic: bool) -> Vec<f32> {
    let rect = rect.intersect(&field.rect);
    let (w, h) = (rect.width() as usize, rect.height() as usize);
    if w == 0 || h == 0 {
        return vec![];
    }
    let fr = field.rect;
    let clampx = |v: f32| v.clamp(fr.x0 as f32 + 0.5, fr.x1 as f32 - 0.5);
    let clampy = |v: f32| v.clamp(fr.y0 as f32 + 0.5, fr.y1 as f32 - 0.5);
    // Sample positions, then the source rect they need.
    let mut pos = vec![[0.0f32; 2]; w * h];
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for (k, s) in pos.iter_mut().enumerate() {
        let p = [(rect.x0 + (k % w) as i32) as f32 + 0.5, (rect.y0 + (k / w) as i32) as f32 + 0.5];
        let d = field.sample(p[0], p[1]);
        *s = [clampx(p[0] + d[0]), clampy(p[1] + d[1])];
        x0 = x0.min(s[0]);
        y0 = y0.min(s[1]);
        x1 = x1.max(s[0]);
        y1 = y1.max(s[1]);
    }
    let sr = IRect::from_f32(x0, y0, x1, y1).expand(3).union(&rect).intersect(&fr);
    let src = Src { r: sr, px: base.read_rect(&sr).chunks_exact(4).map(premul).collect() };
    let selv = sel.map(|s| s.read_rect(&rect));
    let mut out = vec![0.0f32; w * h * 4];
    let mut rows: Vec<(usize, &mut [f32])> = out.chunks_mut(w * 4).enumerate().collect();
    crate::par_for_each(&mut rows, |(y, row)| {
        for x in 0..w {
            let k = *y * w + x;
            let mut v = src.sample(pos[k][0], pos[k][1], bicubic);
            if let Some(sv) = &selv {
                let b = src.get(rect.x0 + x as i32, rect.y0 + *y as i32);
                v = lerp4(b, v, sv[k].clamp(0.0, 1.0));
            }
            let o = &mut row[x * 4..x * 4 + 4];
            if v[3] > 1e-6 {
                o.copy_from_slice(&[v[0] / v[3], v[1] / v[3], v[2] / v[3], v[3].min(1.0)]);
            } else {
                o.fill(0.0);
            }
        }
    });
    drop(rows);
    out
}

/// Bake a whole session on the CPU: the target's pixels = its base warped by `field`
/// (bicubic). Fallback when no GPU readback is available.
pub fn apply_cpu(doc: &mut prisme_core::Document, spec: &LiquifySpec, field: &LiquifyField) {
    let Some(g) = crate::stroke::target_grid(doc, spec.target) else { return };
    *g = (*spec.base).clone();
    let r = field.active_rect();
    if r.is_empty() {
        return;
    }
    let px = warp_rect(&spec.base, field, spec.selection.as_deref(), r, true);
    g.write_rect(&r, &px);
}

#[cfg(test)]
mod tests {
    use super::*;
    use prisme_core::PixelFormat;

    fn dab(mode: LiquifyMode, center: [f32; 2], amount: f32, motion: [f32; 2]) -> LiquifyDab {
        LiquifyDab { mode, center, radius: 30.0, amount, density: 0.5, motion }
    }

    #[test]
    fn identity_warp_is_exact() {
        let mut g = TileGrid::rgba(PixelFormat::F32);
        g.map_rect(&IRect::new(0, 0, 64, 64), |x, y, p| p.copy_from_slice(&[x as f32 / 64.0, y as f32 / 64.0, 0.5, 1.0]));
        let f = LiquifyField::new(IRect::new(0, 0, 64, 64));
        for cubic in [false, true] {
            let px = warp_rect(&g, &f, None, IRect::new(0, 0, 64, 64), cubic);
            let want = g.read_rect(&IRect::new(0, 0, 64, 64));
            assert!(px.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-6));
        }
    }

    #[test]
    fn forward_follows_reconstruct_returns_freeze_blocks() {
        let mut f = LiquifyField::new(IRect::new(0, 0, 200, 200));
        f.apply_dab(&dab(LiquifyMode::Forward, [100.0, 100.0], 1.0, [10.0, 0.0]));
        // Centre output now samples 10 px to the left: content moved right.
        let d = f.sample(100.5, 100.5);
        assert!((d[0] + 10.0).abs() < 0.6 && d[1].abs() < 1e-3, "{d:?}");
        for _ in 0..40 {
            f.apply_dab(&dab(LiquifyMode::Reconstruct, [100.0, 100.0], 0.5, [0.0; 2]));
        }
        assert!(f.sample(100.5, 100.5)[0].abs() < 0.01);
        f.apply_dab(&dab(LiquifyMode::Freeze, [100.0, 100.0], 1.0, [0.0; 2]));
        f.apply_dab(&dab(LiquifyMode::Freeze, [100.0, 100.0], 1.0, [0.0; 2]));
        f.apply_dab(&dab(LiquifyMode::Forward, [100.0, 100.0], 1.0, [10.0, 0.0]));
        assert!(f.sample(100.5, 100.5)[0].abs() < 0.01, "frozen");
        f.apply_dab(&dab(LiquifyMode::Thaw, [100.0, 100.0], 1.0, [0.0; 2]));
        assert!(f.sample(100.5, 100.5)[2] < 0.01);
    }

    #[test]
    fn bloat_magnifies_pucker_shrinks_twirl_turns() {
        let c = [100.0, 100.0];
        let mut f = LiquifyField::new(IRect::new(0, 0, 200, 200));
        f.apply_dab(&dab(LiquifyMode::Bloat, c, 0.3, [0.0; 2]));
        // A point right of centre samples closer to the centre.
        assert!(f.sample(110.0, 100.0)[0] < -0.5);
        let mut f = LiquifyField::new(IRect::new(0, 0, 200, 200));
        f.apply_dab(&dab(LiquifyMode::Pucker, c, 0.3, [0.0; 2]));
        assert!(f.sample(110.0, 100.0)[0] > 0.5);
        let mut f = LiquifyField::new(IRect::new(0, 0, 200, 200));
        f.apply_dab(&dab(LiquifyMode::TwirlCw, c, 0.5, [0.0; 2]));
        // Clockwise (y down): the point right of centre shows content from above (smaller y).
        assert!(f.sample(110.0, 100.0)[1] < -0.5);
    }

    #[test]
    fn coarse_fields_cover_big_areas() {
        let f = LiquifyField::new(IRect::new(-10, 0, 5000, 3000));
        assert_eq!(f.step, 3);
        assert_eq!(f.doc_rect(f.nodes()), f.rect);
        let n = f.nodes_in(IRect::new(100, 100, 130, 130));
        let p = f.node_pos(n.x0, n.y0);
        assert!(p[0] <= 100.0 && p[1] <= 100.0);
    }
}
