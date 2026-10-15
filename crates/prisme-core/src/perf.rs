//! Frame profiler: named, nested time spans and counters recorded on the UI thread.
//!
//! Always on and cheap (a thread-local push per span): the app reads each frame's record
//! with [`end_frame`] and keeps a history for the Performance panel, so a hitch can be looked
//! at after it happened. Other threads record nothing. GPU timings arrive frames later and are
//! attached to the frame that issued the work ([`gpu_time`], keyed by [`frame`]).

use std::cell::RefCell;

/// One closed span: `depth` 0 is a top-level phase of the frame.
#[derive(Clone, Debug)]
pub struct SpanRec {
    pub name: &'static str,
    pub depth: u8,
    /// Start, ms since the frame began.
    pub start_ms: f32,
    pub ms: f32,
}

#[derive(Clone, Debug, Default)]
pub struct FrameRec {
    pub index: u64,
    /// Seconds (`now_seconds` clock) when the frame began.
    pub start: f64,
    /// Spans in closing order (children before their parent).
    pub spans: Vec<SpanRec>,
    pub counters: Vec<(&'static str, u64)>,
}

#[derive(Default)]
struct Recorder {
    frame: FrameRec,
    depth: u8,
    /// GPU results that arrived: (frame index, label, ms).
    gpu: Vec<(u64, &'static str, f32)>,
}

thread_local! {
    static REC: RefCell<Recorder> = RefCell::new(Recorder::default());
}

/// Seconds since an arbitrary origin (works on wasm).
pub fn now() -> f64 {
    use std::sync::OnceLock;
    static START: OnceLock<web_time::Instant> = OnceLock::new();
    START.get_or_init(web_time::Instant::now).elapsed().as_secs_f64()
}

/// Start a new frame record (the previous one should have been taken by [`end_frame`]).
pub fn begin_frame(index: u64) {
    REC.with(|r| {
        let mut r = r.borrow_mut();
        r.frame = FrameRec { index, start: now(), spans: Vec::with_capacity(64), counters: vec![] };
        r.depth = 0;
    });
}

/// Take the current frame's record.
pub fn end_frame() -> FrameRec {
    REC.with(|r| std::mem::take(&mut r.borrow_mut().frame))
}

/// Index of the frame being recorded.
pub fn frame() -> u64 {
    REC.with(|r| r.borrow().frame.index)
}

/// Time a scope: the span closes when the guard drops.
#[must_use = "the span ends when the guard is dropped"]
pub fn span(name: &'static str) -> Span {
    let depth = REC.with(|r| {
        let mut r = r.borrow_mut();
        r.depth += 1;
        r.depth - 1
    });
    Span { name, depth, start: now() }
}

pub struct Span {
    name: &'static str,
    depth: u8,
    start: f64,
}

impl Drop for Span {
    fn drop(&mut self) {
        let end = now();
        REC.with(|r| {
            let mut r = r.borrow_mut();
            r.depth = r.depth.saturating_sub(1);
            let f0 = r.frame.start;
            r.frame.spans.push(SpanRec { name: self.name, depth: self.depth, start_ms: ((self.start - f0) * 1e3) as f32, ms: ((end - self.start) * 1e3) as f32 });
        });
    }
}

/// Add `n` to a per-frame counter.
pub fn count(name: &'static str, n: u64) {
    if n == 0 {
        return;
    }
    REC.with(|r| {
        let c = &mut r.borrow_mut().frame.counters;
        match c.iter_mut().find(|(k, _)| *k == name) {
            Some((_, v)) => *v += n,
            None => c.push((name, n)),
        }
    });
}

/// A GPU timing for work issued during frame `index` (delivered when the query resolves).
pub fn gpu_time(index: u64, label: &'static str, ms: f32) {
    REC.with(|r| r.borrow_mut().gpu.push((index, label, ms)));
}

/// GPU timings delivered since the last call.
pub fn take_gpu_times() -> Vec<(u64, &'static str, f32)> {
    REC.with(|r| std::mem::take(&mut r.borrow_mut().gpu))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_nest_and_counters_add() {
        begin_frame(7);
        {
            let _a = span("outer");
            let _b = span("inner");
            count("tiles", 2);
            count("tiles", 3);
            count("none", 0);
        }
        let f = end_frame();
        assert_eq!(f.index, 7);
        let names: Vec<_> = f.spans.iter().map(|s| (s.name, s.depth)).collect();
        assert_eq!(names, [("inner", 1), ("outer", 0)]);
        assert!(f.spans[1].ms >= f.spans[0].ms);
        assert_eq!(f.counters, [("tiles", 5)]);
    }
}
