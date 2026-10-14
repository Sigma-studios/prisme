use crate::dab::Dab;
use crate::{BrushMode, BrushSettings, InputSample, InputSource};
use prisme_core::tile::{Tile, TileCoord, TileGrid, TILE_SIZE};
use prisme_core::IRect;
use std::collections::HashMap;
use std::sync::Arc;

const TS: i32 = TILE_SIZE as i32;

/// Converts input samples into spaced dabs.
pub struct Stroke {
    pub settings: BrushSettings,
    last: Option<InputSample>,
    smoothed: [f32; 2],
    /// Distance travelled since the last dab.
    residual: f32,
    rng: u64,
    dir: f32,
    last_time: f64,
}

impl Stroke {
    pub fn new(settings: BrushSettings, seed: u64) -> Self {
        Self { settings, last: None, smoothed: [0.0; 2], residual: 0.0, rng: seed | 1, dir: 0.0, last_time: 0.0 }
    }

    fn rand(&mut self) -> f32 {
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        ((self.rng.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32) / (1u64 << 24) as f32
    }

    fn input_value(&mut self, src: InputSource, s: &InputSample, velocity: f32) -> f32 {
        match src {
            InputSource::None => 1.0,
            InputSource::Pressure => s.pressure,
            InputSource::Velocity => 1.0 - velocity,
            InputSource::Tilt => s.tilt,
            InputSource::Random => self.rand(),
            InputSource::Direction => (self.dir / std::f32::consts::TAU).rem_euclid(1.0),
        }
    }

    fn make_dab(&mut self, pos: [f32; 2], s: &InputSample, velocity: f32) -> Dab {
        let st = self.settings.clone();
        let size_in = self.input_value(st.size_dynamic.source, s, velocity);
        let op_in = self.input_value(st.opacity_dynamic.source, s, velocity);
        let flow_in = self.input_value(st.flow_dynamic.source, s, velocity);
        let mut radius = st.size * 0.5 * st.size_dynamic.eval(size_in);
        if st.jitter_size > 0.0 {
            radius *= 1.0 - st.jitter_size * self.rand();
        }
        let mut alpha = st.flow * st.flow_dynamic.eval(flow_in) * st.opacity_dynamic.eval(op_in);
        if st.jitter_opacity > 0.0 {
            alpha *= 1.0 - st.jitter_opacity * self.rand();
        }
        let mut angle = st.angle;
        match st.angle_source {
            InputSource::Direction => angle += self.dir,
            InputSource::Tilt => angle += s.tilt_dir,
            InputSource::Random => angle += self.rand() * std::f32::consts::TAU,
            _ => {}
        }
        if st.jitter_angle > 0.0 {
            angle += (self.rand() - 0.5) * st.jitter_angle * std::f32::consts::TAU;
        }
        let mut center = pos;
        if st.scatter > 0.0 {
            let a = self.rand() * std::f32::consts::TAU;
            let d = self.rand() * st.scatter * st.size;
            center[0] += a.cos() * d;
            center[1] += a.sin() * d;
        }
        if st.pixel_perfect {
            radius = radius.max(0.5);
            // Snap so odd diameters centre on a pixel and even ones on a pixel corner.
            let diam = (radius * 2.0).round().max(1.0);
            radius = diam * 0.5;
            let off = if (diam as i32) % 2 == 1 { 0.5 } else { 0.0 };
            center = [(center[0] - off).round() + off, (center[1] - off).round() + off];
        }
        let seed = (self.rand() * 1e6) as u32;
        Dab {
            center,
            radius,
            angle,
            roundness: st.roundness,
            hardness: st.hardness,
            alpha: alpha.clamp(0.0, 1.0),
            tip: st.tip,
            pixel: st.pixel_perfect,
            grain: st.grain,
            seed,
        }
    }

    fn step(&self) -> f32 {
        let s = &self.settings;
        if s.pixel_perfect {
            return 1.0_f32.max(s.size * s.spacing).min(s.size.max(1.0));
        }
        (s.size * s.spacing).max(0.5)
    }

    /// Feed a sample; returns the dabs to stamp.
    pub fn add(&mut self, s: InputSample) -> Vec<Dab> {
        let mut out = vec![];
        let Some(prev) = self.last else {
            self.smoothed = s.pos;
            self.last = Some(s);
            self.last_time = s.time;
            out.push(self.make_dab(s.pos, &s, 0.0));
            return out;
        };
        // Lazy-mouse smoothing.
        let k = 1.0 - self.settings.smoothing.clamp(0.0, 0.95);
        let target = [self.smoothed[0] + (s.pos[0] - self.smoothed[0]) * k, self.smoothed[1] + (s.pos[1] - self.smoothed[1]) * k];
        let from = self.smoothed;
        let dx = target[0] - from[0];
        let dy = target[1] - from[1];
        let dist = (dx * dx + dy * dy).sqrt();
        if dist > 1e-4 {
            self.dir = dy.atan2(dx);
        }
        let dt = (s.time - prev.time).max(1e-4) as f32;
        let velocity = ((dist / dt) / 3000.0).clamp(0.0, 1.0);
        let step = self.step();
        let mut t = step - self.residual;
        while t <= dist {
            let f = t / dist.max(1e-6);
            let pos = [from[0] + dx * f, from[1] + dy * f];
            let interp = InputSample {
                pos,
                pressure: prev.pressure + (s.pressure - prev.pressure) * f,
                tilt: prev.tilt + (s.tilt - prev.tilt) * f,
                tilt_dir: s.tilt_dir,
                time: prev.time + (s.time - prev.time) * f as f64,
            };
            out.push(self.make_dab(pos, &interp, velocity));
            t += step;
        }
        self.residual = dist - (t - step);
        self.smoothed = target;
        self.last = Some(s);
        out
    }

    /// Airbrush: emit dabs over time while the pointer is stationary.
    pub fn tick(&mut self, time: f64) -> Vec<Dab> {
        if !self.settings.airbrush {
            return vec![];
        }
        let Some(last) = self.last else { return vec![] };
        let rate = 30.0; // dabs per second
        let n = ((time - self.last_time) * rate) as usize;
        if n == 0 {
            return vec![];
        }
        self.last_time += n as f64 / rate;
        let pos = self.smoothed;
        (0..n.min(8)).map(|_| self.make_dab(pos, &last, 0.0)).collect()
    }
}

/// What a stroke writes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlendTarget {
    /// Paint a straight-alpha linear colour.
    Color([f32; 4]),
    Erase,
    /// Paint a value into a single-channel grid (layer masks).
    Value(f32),
}

impl BlendTarget {
    pub fn from_mode(mode: BrushMode, color: [f32; 4]) -> Self {
        match mode {
            BrushMode::Paint => BlendTarget::Color(color),
            BrushMode::Erase => BlendTarget::Erase,
        }
    }
}

/// Accumulates dab coverage for one stroke and composites it onto the base snapshot.
pub struct StrokeCanvas {
    base: Arc<TileGrid>,
    coverage: TileGrid,
    pub target: BlendTarget,
    pub opacity: f32,
    selection: Option<Arc<TileGrid>>,
    lock_alpha: bool,
    /// Value of missing tiles in the base (1.0 for reveal-all masks).
    base_default: f32,
    pub dirty: IRect,
}

impl StrokeCanvas {
    pub fn new(
        base: Arc<TileGrid>,
        target: BlendTarget,
        opacity: f32,
        selection: Option<Arc<TileGrid>>,
        lock_alpha: bool,
        base_default: f32,
    ) -> Self {
        Self {
            base,
            coverage: TileGrid::new(prisme_core::PixelFormat::F32, 1),
            target,
            opacity,
            selection,
            lock_alpha,
            base_default,
            dirty: IRect::EMPTY,
        }
    }

    pub fn base(&self) -> &Arc<TileGrid> {
        &self.base
    }

    /// Stamp dabs and recomposite the affected pixels into `out`. Returns the changed rect.
    pub fn apply(&mut self, dabs: &[Dab], out: &mut TileGrid) -> IRect {
        if dabs.is_empty() {
            return IRect::EMPTY;
        }
        // Group dabs by tile.
        let mut per_tile: HashMap<TileCoord, Vec<usize>> = HashMap::new();
        let mut changed = IRect::EMPTY;
        for (i, d) in dabs.iter().enumerate() {
            let b = d.bounds();
            let r = IRect::new(b[0], b[1], b[2], b[3]);
            changed = changed.union(&r);
            for c in TileGrid::coords_in(&r) {
                per_tile.entry(c).or_default().push(i);
            }
        }
        // Make sure tiles exist (so we can process them independently).
        let coords: Vec<TileCoord> = per_tile.keys().copied().collect();
        let channels = out.channels;
        let format = out.format;
        let base_default = self.base_default;
        let mut jobs: Vec<(TileCoord, Tile, Tile)> = coords
            .iter()
            .map(|c| {
                let cov = self.coverage.tiles.remove(c).map(|t| Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone()));
                let cov = cov.unwrap_or_else(|| Tile::new(prisme_core::PixelFormat::F32, 1));
                let o = out.tiles.remove(c).map(|t| Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone()));
                let o = o.unwrap_or_else(|| {
                    let mut t = Tile::new(format, channels);
                    if base_default != 0.0 {
                        t.fill([base_default; 4]);
                    }
                    t
                });
                (*c, cov, o)
            })
            .collect();
        let target = self.target;
        let opacity = self.opacity;
        let lock_alpha = self.lock_alpha;
        let base = self.base.clone();
        let selection = self.selection.clone();
        let process = |(c, cov, out_t): &mut (TileCoord, Tile, Tile)| {
            let tile_rect = c.rect();
            let mut touched = IRect::EMPTY;
            cov.with_f32_mut(|cv| {
                for &i in &per_tile[c] {
                    let d = &dabs[i];
                    let b = d.bounds();
                    let r = IRect::new(b[0], b[1], b[2], b[3]).intersect(&tile_rect);
                    if r.is_empty() {
                        continue;
                    }
                    touched = touched.union(&r);
                    for y in r.y0..r.y1 {
                        let row = ((y - tile_rect.y0) * TS) as usize;
                        for x in r.x0..r.x1 {
                            let k = d.coverage(x as f32 + 0.5, y as f32 + 0.5) * d.alpha;
                            if k > 0.0 {
                                let idx = row + (x - tile_rect.x0) as usize;
                                cv[idx] += (1.0 - cv[idx]) * k;
                            }
                        }
                    }
                }
            });
            if touched.is_empty() {
                return;
            }
            let cv = cov.f32_view();
            let base_tile = base.get_tile(*c);
            let bv = base_tile.map(|t| t.f32_view());
            let sel_tile = selection.as_ref().and_then(|s| s.get_tile(*c).cloned());
            let sv = sel_tile.as_ref().map(|t| t.f32_view());
            let has_sel = selection.is_some();
            let ch = channels as usize;
            out_t.with_f32_mut(|o| {
                for y in touched.y0..touched.y1 {
                    let row = ((y - tile_rect.y0) * TS) as usize;
                    for x in touched.x0..touched.x1 {
                        let idx = row + (x - tile_rect.x0) as usize;
                        let mut a = cv[idx] * opacity;
                        if has_sel {
                            a *= sv.as_ref().map(|s| s[idx]).unwrap_or(0.0);
                        }
                        let p = idx * ch;
                        let b: [f32; 4] = match &bv {
                            Some(bv) if ch == 4 => [bv[p], bv[p + 1], bv[p + 2], bv[p + 3]],
                            Some(bv) => [bv[p]; 4],
                            None => [base_default; 4],
                        };
                        match target {
                            BlendTarget::Value(v) => {
                                o[p] = b[0] + (v - b[0]) * a;
                            }
                            BlendTarget::Erase => {
                                if ch == 4 {
                                    o[p..p + 3].copy_from_slice(&b[..3]);
                                    o[p + 3] = b[3] * (1.0 - a);
                                } else {
                                    o[p] = b[0] * (1.0 - a);
                                }
                            }
                            BlendTarget::Color(col) => {
                                let a = a * col[3];
                                if ch == 1 {
                                    o[p] = b[0] + (1.0 - b[0]) * a;
                                } else if lock_alpha {
                                    for k in 0..3 {
                                        o[p + k] = b[k] + (col[k] - b[k]) * a;
                                    }
                                    o[p + 3] = b[3];
                                } else {
                                    let ao = a + b[3] * (1.0 - a);
                                    if ao > 1e-7 {
                                        for k in 0..3 {
                                            o[p + k] = (col[k] * a + b[k] * b[3] * (1.0 - a)) / ao;
                                        }
                                    }
                                    o[p + 3] = ao;
                                }
                            }
                        }
                    }
                }
            });
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            jobs.par_iter_mut().for_each(process);
        }
        #[cfg(target_arch = "wasm32")]
        jobs.iter_mut().for_each(process);

        for (c, cov, o) in jobs {
            self.coverage.tiles.insert(c, Arc::new(cov));
            out.tiles.insert(c, Arc::new(o));
        }
        self.dirty = self.dirty.union(&changed);
        changed
    }

    /// Drop transparent tiles created by the stroke (call when the stroke ends).
    pub fn finish(&self, out: &mut TileGrid) {
        if self.base_default == 0.0 {
            let dirty = self.dirty;
            let coords: Vec<_> = TileGrid::coords_in(&dirty).collect();
            for c in coords {
                if out.tiles.get(&c).is_some_and(|t| t.is_clear()) {
                    out.tiles.remove(&c);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BrushSettings;
    use prisme_core::PixelFormat;

    #[test]
    fn spacing_produces_even_dabs() {
        let mut s = Stroke::new(BrushSettings { size: 10.0, spacing: 0.5, smoothing: 0.0, ..Default::default() }, 1);
        let mut dabs = s.add(InputSample::mouse([0.0, 0.0], 0.0));
        dabs.extend(s.add(InputSample::mouse([100.0, 0.0], 0.1)));
        assert_eq!(dabs.len(), 21);
        for w in dabs.windows(2) {
            assert!((w[1].center[0] - w[0].center[0] - 5.0).abs() < 1e-3);
        }
    }

    #[test]
    fn opacity_caps_stroke() {
        let base = Arc::new(TileGrid::rgba(PixelFormat::F32));
        let mut out = (*base).clone();
        let mut canvas = StrokeCanvas::new(base, BlendTarget::Color([1.0, 0.0, 0.0, 1.0]), 0.5, None, false, 0.0);
        let mut s = Stroke::new(BrushSettings { size: 20.0, hardness: 1.0, spacing: 0.05, ..Default::default() }, 1);
        let mut dabs = s.add(InputSample::mouse([50.0, 50.0], 0.0));
        for i in 0..20 {
            dabs.extend(s.add(InputSample::mouse([50.0 + (i % 2) as f32, 50.0], 0.01 * i as f64)));
        }
        canvas.apply(&dabs, &mut out);
        let p = out.get_pixel(50, 50);
        assert!((p[3] - 0.5).abs() < 1e-3, "{p:?}");
        assert!((p[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn erase_and_selection() {
        let mut g = TileGrid::rgba(PixelFormat::F32);
        g.map_rect(&IRect::new(0, 0, 100, 100), |_, _, p| p.copy_from_slice(&[0.5, 0.5, 0.5, 1.0]));
        let base = Arc::new(g);
        let mut out = (*base).clone();
        let sel = Arc::new(prisme_core::mask::rect([0.0, 0.0, 50.0, 100.0], false));
        let mut canvas = StrokeCanvas::new(base, BlendTarget::Erase, 1.0, Some(sel), false, 0.0);
        let mut s = Stroke::new(BrushSettings { size: 200.0, hardness: 1.0, ..Default::default() }, 1);
        let dabs = s.add(InputSample::mouse([50.0, 50.0], 0.0));
        canvas.apply(&dabs, &mut out);
        assert_eq!(out.get_pixel(10, 50)[3], 0.0);
        assert_eq!(out.get_pixel(80, 50)[3], 1.0);
    }

    #[test]
    fn pixel_brush_single_pixel() {
        let base = Arc::new(TileGrid::rgba(PixelFormat::F16));
        let mut out = (*base).clone();
        let mut canvas = StrokeCanvas::new(base, BlendTarget::Color([1.0; 4]), 1.0, None, false, 0.0);
        let mut s = Stroke::new(BrushSettings { size: 1.0, pixel_perfect: true, ..Default::default() }, 1);
        let dabs = s.add(InputSample::mouse([10.3, 10.7], 0.0));
        canvas.apply(&dabs, &mut out);
        let b = out.content_bounds();
        assert_eq!(b, IRect::new(10, 10, 11, 11));
        assert_eq!(out.get_pixel(10, 10)[3], 1.0);
    }
}
