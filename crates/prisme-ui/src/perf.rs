//! Frame timing history for the Performance panel and slow-frame warnings.
//!
//! Every frame is recorded ([`prisme_core::perf`] spans and counters, GPU timings when the
//! device has timestamp queries) and kept for the last few hundred frames, so a stutter can be
//! inspected after it happened. A frame's duration runs from its start to the next frame's: its
//! own UI work, egui's tessellation and painting, then the wait for the GPU / vsync (or for
//! input when idle, which is why idle frames don't count as slow).

use crate::app::AppState;
use prisme_core::perf::{self, FrameRec};
use std::collections::VecDeque;

/// Frames kept.
const KEEP: usize = 600;
/// Slow frames kept.
const KEEP_SLOW: usize = 60;

pub struct Frame {
    pub rec: FrameRec,
    /// Start to next frame's start.
    pub duration_ms: f32,
    /// Inside `PrismeApp::ui`.
    pub ui_ms: f32,
    /// All CPU work of the frame (UI, tessellation, painting), without the vsync / GPU wait.
    pub cpu_ms: Option<f32>,
    /// GPU timings of work issued during the frame.
    pub gpu: Vec<(&'static str, f32)>,
    /// Painting, a tool interaction, a stroke landing or a job running: frames follow each
    /// other, so a long duration is a stutter rather than idling.
    pub active: bool,
    pub tool: &'static str,
}

impl Frame {
    pub fn wait_ms(&self) -> f32 {
        (self.duration_ms - self.cpu_ms.unwrap_or(self.ui_ms)).max(0.0)
    }

    pub fn paint_ms(&self) -> f32 {
        (self.cpu_ms.unwrap_or(self.ui_ms) - self.ui_ms).max(0.0)
    }

    pub fn gpu_ms(&self) -> Option<f32> {
        (!self.gpu.is_empty()).then(|| self.gpu.iter().map(|g| g.1).sum())
    }

    /// Slow: the CPU work alone, or (while active) the whole frame, took over `limit`.
    pub fn slow(&self, limit: f32) -> bool {
        self.cpu_ms.unwrap_or(self.ui_ms) > limit || (self.active && self.duration_ms > limit)
    }

    /// The deepest span that explains most of the frame (a phase with one dominant child
    /// gives that child).
    pub fn culprit(&self) -> Option<(&'static str, f32)> {
        let spans = &self.rec.spans;
        let mut best = spans.iter().filter(|s| s.depth == 0).max_by(|a, b| a.ms.total_cmp(&b.ms))?;
        loop {
            let end = best.start_ms + best.ms;
            let child = spans
                .iter()
                .filter(|s| s.depth == best.depth + 1 && s.start_ms >= best.start_ms && s.start_ms + s.ms <= end + 0.01)
                .max_by(|a, b| a.ms.total_cmp(&b.ms));
            match child {
                Some(c) if c.ms > best.ms * 0.5 => best = c,
                _ => return Some((best.name, best.ms)),
            }
        }
    }

    /// One-line reason for a slow frame.
    pub fn diagnosis(&self) -> String {
        let cpu = self.cpu_ms.unwrap_or(self.ui_ms);
        let gpu = self.gpu_ms();
        if self.wait_ms() > cpu {
            return match gpu {
                Some(g) => format!("waiting for the GPU (engine work {g:.1} ms)"),
                None => "waiting for the GPU / display".into(),
            };
        }
        if self.paint_ms() > self.ui_ms {
            return format!("UI painting {:.1} ms", self.paint_ms());
        }
        match self.culprit() {
            Some((name, ms)) => format!("{name} {ms:.1} ms"),
            None => format!("UI {:.1} ms", self.ui_ms),
        }
    }
}

#[derive(Default)]
pub struct PerfState {
    pub frames: VecDeque<Frame>,
    /// Indices (`rec.index`) of recent slow frames, oldest first.
    pub slow: VecDeque<u64>,
    /// The frame being recorded, finished when the next one starts.
    current: Option<(Frame, f64)>,
    next_index: u64,
    pub paused: bool,
    pub selected: Option<u64>,
    last_warning: f64,
    /// Whether the device times GPU work.
    pub gpu_timing: bool,
}

impl PerfState {
    pub fn get(&self, index: u64) -> Option<&Frame> {
        let first = self.frames.front()?.rec.index;
        self.frames.get(index.checked_sub(first)? as usize).filter(|f| f.rec.index == index)
    }

    fn get_mut(&mut self, index: u64) -> Option<&mut Frame> {
        let first = self.frames.front()?.rec.index;
        self.frames.get_mut(index.checked_sub(first)? as usize).filter(|f| f.rec.index == index)
    }

    pub fn clear(&mut self) {
        self.frames.clear();
        self.slow.clear();
        self.selected = None;
    }
}

/// Start of `PrismeApp::ui`: finish the previous frame (its duration and full CPU time are
/// known now) and start recording this one.
pub fn begin(s: &mut AppState, cpu_prev: Option<f32>) {
    let now = perf::now();
    let p = &mut s.perf;
    if let Some((mut f, start)) = p.current.take() {
        f.duration_ms = ((now - start) * 1e3) as f32;
        // eframe reports the last painted frame's CPU time; a frame egui ran twice (or a
        // screenshot frame) doesn't line up with it, so implausible values are dropped.
        f.cpu_ms = cpu_prev.map(|c| c * 1e3).filter(|c| *c >= f.ui_ms * 0.9 && *c <= f.duration_ms + 1.0);
        if !p.paused {
            let limit = s.settings.perf_slow_ms;
            if f.slow(limit) {
                if s.settings.perf_warn && now - p.last_warning > 3.0 {
                    p.last_warning = now;
                    let msg = format!("{} Slow frame: {:.0} ms — {}", egui_phosphor::regular::WARNING, f.duration_ms, f.diagnosis());
                    s.status = Some((msg, prisme_platform::now_seconds()));
                }
                p.slow.push_back(f.rec.index);
                if p.slow.len() > KEEP_SLOW {
                    p.slow.pop_front();
                }
            }
            p.frames.push_back(f);
            if p.frames.len() > KEEP {
                p.frames.pop_front();
            }
        }
    }
    if let Some(g) = &mut s.gpu {
        g.engine.poll_timer();
        p.gpu_timing = g.engine.timer.is_some();
    }
    for (i, label, ms) in perf::take_gpu_times() {
        if let Some(f) = p.get_mut(i) {
            match f.gpu.iter_mut().find(|g| g.0 == label) {
                Some(g) => g.1 += ms,
                None => f.gpu.push((label, ms)),
            }
        }
    }
    p.next_index += 1;
    perf::begin_frame(p.next_index);
}

/// End of `PrismeApp::ui`.
pub fn end(s: &mut AppState, ctx: &egui::Context, ui_ms: f32) {
    let rec = perf::end_frame();
    let start = perf::now() - ui_ms as f64 / 1e3;
    let busy = s.tools.get(&s.tool).is_some_and(|t| t.busy());
    let active = busy || ctx.input(|i| i.pointer.any_down()) || s.docs.iter().any(|d| d.stroke.is_some()) || !s.jobs.is_empty();
    let f = Frame { rec, duration_ms: 0.0, ui_ms, cpu_ms: None, gpu: vec![], active, tool: s.tool.name() };
    s.perf.current = Some((f, start));
}

/// Plain-text report of the slow frames and the selected one (Copy Report, `perf-report`).
pub fn report(s: &AppState) -> String {
    use std::fmt::Write;
    let p = &s.perf;
    let mut out = String::new();
    let _ = writeln!(out, "Prisme performance report");
    if let Some(g) = &s.gpu {
        let info = g.rs.adapter.get_info();
        let (_, _, vram) = g.engine.tile_stats();
        let _ = writeln!(out, "GPU: {} ({:?}), GPU timing: {}, resident tiles {} MB, texture pool {} MB", info.name, info.backend, if p.gpu_timing { "on" } else { "unavailable" }, vram >> 20, g.engine.pool_bytes() >> 20);
    }
    if let Some(d) = s.doc() {
        let doc = d.doc();
        let _ = writeln!(out, "Document: {} × {}, {} layers, {} nodes, history {} entries, zoom {:.0}%", doc.info.width, doc.info.height, doc.all_layers().len(), doc.graph.nodes.len(), d.history.entries().len(), d.view.zoom * 100.0);
    }
    let active: Vec<&Frame> = p.frames.iter().filter(|f| f.active).collect();
    if !active.is_empty() {
        let mut d: Vec<f32> = active.iter().map(|f| f.duration_ms).collect();
        d.sort_by(f32::total_cmp);
        let q = |x: f32| d[((d.len() - 1) as f32 * x) as usize];
        let _ = writeln!(out, "Active frames: {} — median {:.1} ms, 95th {:.1} ms, worst {:.1} ms", d.len(), q(0.5), q(0.95), q(1.0));
    }
    let _ = writeln!(out, "\nSlow frames (> {:.0} ms), oldest first:", s.settings.perf_slow_ms);
    for i in &p.slow {
        if let Some(f) = p.get(*i) {
            let _ = writeln!(out, "  #{i} {:.1} ms ({}) — {}", f.duration_ms, f.tool, f.diagnosis());
        }
    }
    // The selected frame, then the latest slow ones.
    let mut pick: Vec<u64> = p.selected.into_iter().collect();
    pick.extend(p.slow.iter().rev().take(3).filter(|i| p.selected != Some(**i)));
    for f in pick.into_iter().filter_map(|i| p.get(i)) {
        let _ = writeln!(out, "\nFrame #{}:", f.rec.index);
        frame_text(&mut out, f);
    }
    out
}

pub fn frame_text(out: &mut String, f: &Frame) {
    use std::fmt::Write;
    let _ = writeln!(
        out,
        "  duration {:.1} ms = UI {:.1} + painting {:.1} + wait {:.1}{}",
        f.duration_ms,
        f.ui_ms,
        f.paint_ms(),
        f.wait_ms(),
        if f.active { "" } else { " (idle)" }
    );
    for (label, ms) in &f.gpu {
        let _ = writeln!(out, "  GPU {label}: {ms:.2} ms");
    }
    for s in sorted_spans(f) {
        let _ = writeln!(out, "  {}{} {:.2} ms", "  ".repeat(s.depth as usize), s.name, s.ms);
    }
    for (k, v) in &f.rec.counters {
        let _ = writeln!(out, "  {k}: {v}");
    }
}

/// Spans in start order (a parent before its children), repeated siblings merged.
pub fn sorted_spans(f: &Frame) -> Vec<perf::SpanRec> {
    let mut v = f.rec.spans.clone();
    v.sort_by(|a, b| a.start_ms.total_cmp(&b.start_ms).then(a.depth.cmp(&b.depth)));
    let mut out: Vec<perf::SpanRec> = vec![];
    // Merge consecutive siblings with the same name (one per pointer event, per dab batch…).
    for s in v {
        if let Some(prev) = out.iter_mut().rev().find(|p| p.depth <= s.depth)
            && prev.depth == s.depth
            && prev.name == s.name
        {
            prev.ms += s.ms;
            continue;
        }
        out.push(s);
    }
    out
}
