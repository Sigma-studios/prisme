//! GPU time of the engine's work, for the Performance panel.
//!
//! A scope writes a timestamp at its start and end (empty compute passes carrying pass
//! timestamps: the only kind every backend with `TIMESTAMP_QUERY` supports). Each UI frame's
//! scopes share a query set; when the frame changes they're resolved, read back asynchronously
//! and handed to [`prisme_core::perf::gpu_time`] under the frame that issued them. Without the
//! feature (most browsers, some drivers) nothing is recorded.

use prisme_core::perf;
use std::sync::{Arc, Mutex};

/// Scopes per frame (two queries each).
const SCOPES: u32 = 32;
/// Frames in flight before the oldest batch's results are dropped.
const BATCHES: usize = 4;

struct Batch {
    frame: u64,
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    labels: Vec<&'static str>,
    /// None: recording or free; Some(ok) once the read buffer is mapped.
    mapped: Arc<Mutex<Option<bool>>>,
    state: State,
}

#[derive(PartialEq, Eq)]
enum State {
    Free,
    Recording,
    Mapping,
}

pub struct GpuTimer {
    batches: Vec<Batch>,
    /// A one-thread no-op: some backends (Metal) skip the timestamps of empty passes.
    noop: wgpu::ComputePipeline,
    current: Option<usize>,
    period_ns: f32,
}

/// (batch, scope index) of a started scope; None when untimed.
pub struct Scope(Option<(usize, u32)>);

impl Scope {
    pub const NONE: Scope = Scope(None);
}

impl GpuTimer {
    /// None when the device can't time (no `TIMESTAMP_QUERY`).
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let size = SCOPES as u64 * 2 * 8;
        let batches = (0..BATCHES)
            .map(|_| Batch {
                frame: 0,
                set: device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("perf"), ty: wgpu::QueryType::Timestamp, count: SCOPES * 2 }),
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("perf-resolve"),
                    size,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                read: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("perf-read"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                labels: vec![],
                mapped: Arc::new(Mutex::new(None)),
                state: State::Free,
            })
            .collect();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("perf-noop"),
            source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
        });
        let noop = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("perf-noop"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Some(GpuTimer { batches, noop, current: None, period_ns: queue.get_timestamp_period() })
    }

    fn stamp(noop: &wgpu::ComputePipeline, enc: &mut wgpu::CommandEncoder, set: &wgpu::QuerySet, index: u32) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("perf-stamp"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites { query_set: set, beginning_of_pass_write_index: Some(index), end_of_pass_write_index: None }),
        });
        pass.set_pipeline(noop);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Start timing work recorded into `enc` after this call (labelled under the current frame).
    pub fn begin(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, label: &'static str) -> Scope {
        let frame = perf::frame();
        if self.current.is_some_and(|i| self.batches[i].frame != frame) {
            self.finish(device, queue);
        }
        if self.current.is_none() {
            // A free batch, or none: this frame goes untimed.
            let Some(i) = self.batches.iter().position(|b| b.state == State::Free) else { return Scope(None) };
            let b = &mut self.batches[i];
            b.frame = frame;
            b.labels.clear();
            b.state = State::Recording;
            self.current = Some(i);
        }
        let c = self.current.unwrap();
        let b = &mut self.batches[c];
        if b.labels.len() as u32 >= SCOPES {
            return Scope(None);
        }
        let i = b.labels.len() as u32;
        b.labels.push(label);
        Self::stamp(&self.noop, enc, &b.set, i * 2);
        Scope(Some((c, i)))
    }

    /// End a scope (record into the same encoder, or one submitted after it).
    pub fn end(&mut self, enc: &mut wgpu::CommandEncoder, scope: Scope) {
        if let Some((c, i)) = scope.0
            && self.current == Some(c) {
                Self::stamp(&self.noop, enc, &self.batches[c].set, i * 2 + 1);
            }
    }

    /// Resolve the recording batch (its scopes' work must have been submitted).
    fn finish(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Some(i) = self.current.take() else { return };
        let b = &mut self.batches[i];
        let n = b.labels.len() as u32 * 2;
        if n == 0 {
            b.state = State::Free;
            return;
        }
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("perf-resolve") });
        enc.resolve_query_set(&b.set, 0..n, &b.resolve, 0);
        enc.copy_buffer_to_buffer(&b.resolve, 0, &b.read, 0, n as u64 * 8);
        queue.submit([enc.finish()]);
        *b.mapped.lock().unwrap() = None;
        let m = b.mapped.clone();
        b.read.slice(..n as u64 * 8).map_async(wgpu::MapMode::Read, move |r| *m.lock().unwrap() = Some(r.is_ok()));
        b.state = State::Mapping;
    }

    /// Per UI frame (before recording new work): resolve the previous frames' scopes and
    /// deliver the ones that came back.
    pub fn poll(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.current.is_some_and(|i| self.batches[i].frame != perf::frame()) {
            self.finish(device, queue);
        }
        let _ = device.poll(wgpu::PollType::Poll);
        for b in &mut self.batches {
            if b.state != State::Mapping {
                continue;
            }
            let Some(ok) = *b.mapped.lock().unwrap() else { continue };
            if ok {
                let n = b.labels.len() * 2;
                let ticks: Vec<u64> = match b.read.slice(..n as u64 * 8).get_mapped_range() {
                    Ok(d) => bytemuck::cast_slice(&d).to_vec(),
                    Err(_) => vec![0; n],
                };
                for (i, label) in b.labels.iter().enumerate() {
                    let (t0, t1) = (ticks[i * 2], ticks[i * 2 + 1]);
                    // An unended scope's end query reads 0.
                    if t1 > t0 && t0 != 0 {
                        perf::gpu_time(b.frame, label, (t1 - t0) as f32 * self.period_ns / 1e6);
                    }
                }
                b.read.unmap();
            }
            b.state = State::Free;
        }
    }
}
