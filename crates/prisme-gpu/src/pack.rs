//! f16 landing of GPU tiles: stroke tiles (f32 textures) converted to their f16 bits on the
//! GPU (`shaders/pack.wgsl`, exactly the `half` crate's rounding) into storage buffers, then
//! read back as ready-made [`Tile`]s: half the bytes of an f32 read-back and no conversion on
//! the CPU. The buffers stay alive so the same bits can be handed to the tile store
//! ([`crate::Engine::adopt_tiles`]) instead of uploading the landed tiles again.
//!
//! The buffers go back to a small pool when the landing is done: fresh ones cost more than the
//! conversion itself (wgpu zero-fills them and their pages are touched for the first time —
//! tens of ms of GPU time for a long stroke, felt as a stall right after pointer-up).

use crate::image::GpuImage;
use prisme_core::tile::{PixelFormat, Tile, TileCoord, TILE_PIXELS, TILE_SIZE};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Tiles per buffer (32 MiB for RGBA).
const CHUNK: usize = 64;
/// Buffer pairs (bits + read) the pool keeps: up to `POOL_KEEP * CHUNK` tiles land without
/// allocating.
const POOL_KEEP: usize = 4;

/// Spare (channels, bits, read) buffers, `CHUNK` tiles each.
#[derive(Clone, Default)]
struct BufferPool(Arc<Mutex<Vec<(u8, wgpu::Buffer, wgpu::Buffer)>>>);

pub(crate) struct PackPipelines {
    layout: wgpu::BindGroupLayout,
    rgba: wgpu::ComputePipeline,
    mask: wgpu::ComputePipeline,
    pool: BufferPool,
}

impl PackPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let src = include_str!("shaders/pack.wgsl").replace("TILE_SIZE_U", &format!("{TILE_SIZE}u"));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("pack"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pack"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("pack"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pack"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self { rgba: pipeline("k_pack4"), mask: pipeline("k_pack1"), layout, pool: BufferPool::default() }
    }
}

struct Chunk {
    /// The f16 bits (kept for the tile store).
    bits: wgpu::Buffer,
    read: wgpu::Buffer,
    first: usize,
    count: usize,
    state: Arc<Mutex<Option<bool>>>,
    /// The read buffer was unmapped after its tiles were taken: the pair can be reused.
    done: Arc<AtomicBool>,
}

/// A set of tiles converted to f16 on the GPU, being read back.
pub struct PackedTiles {
    pub coords: Vec<TileCoord>,
    channels: u8,
    chunks: Vec<Chunk>,
    pool: BufferPool,
}

impl Drop for PackedTiles {
    fn drop(&mut self) {
        let mut pool = self.pool.0.lock().unwrap();
        for c in self.chunks.drain(..) {
            if c.done.load(Ordering::Acquire) && pool.len() < POOL_KEEP {
                pool.push((self.channels, c.bits, c.read));
            }
        }
    }
}

/// Bytes of one f16 tile.
pub fn tile_bytes(channels: u8) -> u64 {
    TILE_PIXELS as u64 * channels as u64 * 2
}

impl PackedTiles {
    /// Convert `tiles` (f32 textures, one tile each, `channels` 1 or 4) and start reading them.
    pub(crate) fn start(device: &wgpu::Device, queue: &wgpu::Queue, p: &PackPipelines, tiles: &[(TileCoord, GpuImage)], channels: u8) -> PackedTiles {
        let per = tile_bytes(channels);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pack") });
        let mut chunks = vec![];
        for (n, part) in tiles.chunks(CHUNK).enumerate() {
            let size = per * part.len() as u64;
            // Full-size buffers (any chunk can reuse them), from the pool when it has some.
            let spare = {
                let mut pool = p.pool.0.lock().unwrap();
                pool.iter().position(|b| b.0 == channels).map(|i| pool.swap_remove(i))
            };
            let (bits, read) = match spare {
                Some((_, bits, read)) => (bits, read),
                None => {
                    let full = per * CHUNK as u64;
                    let bits = device.create_buffer(&wgpu::BufferDescriptor { label: Some("pack-bits"), size: full, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
                    let read = device.create_buffer(&wgpu::BufferDescriptor { label: Some("pack-read"), size: full, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                    (bits, read)
                }
            };
            let groups: Vec<wgpu::BindGroup> = part
                .iter()
                .enumerate()
                .map(|(i, (_, img))| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("pack"),
                        layout: &p.layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(img.view()) },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &bits, offset: per * i as u64, size: wgpu::BufferSize::new(per) }),
                            },
                        ],
                    })
                })
                .collect();
            {
                // One pass for the whole chunk.
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pack"), timestamp_writes: None });
                pass.set_pipeline(if channels == 1 { &p.mask } else { &p.rgba });
                let words = (TILE_PIXELS as u32 * channels as u32 / 2).div_ceil(64);
                for g in &groups {
                    pass.set_bind_group(0, g, &[]);
                    pass.dispatch_workgroups(words, 1, 1);
                }
            }
            enc.copy_buffer_to_buffer(&bits, 0, &read, 0, size);
            chunks.push(Chunk { bits, read, first: n * CHUNK, count: part.len(), state: Arc::new(Mutex::new(None)), done: Arc::new(AtomicBool::new(false)) });
        }
        queue.submit([enc.finish()]);
        for c in &chunks {
            let s = c.state.clone();
            c.read.slice(..per * c.count as u64).map_async(wgpu::MapMode::Read, move |r| *s.lock().unwrap() = Some(r.is_ok()));
        }
        PackedTiles { coords: tiles.iter().map(|(c, _)| *c).collect(), channels, chunks, pool: p.pool.clone() }
    }

    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// Non-blocking: whether [`PackedTiles::take`] would return the tiles.
    pub fn ready(&self, device: &wgpu::Device) -> bool {
        let _ = device.poll(wgpu::PollType::Poll);
        self.chunks.iter().all(|c| c.state.lock().unwrap().is_some())
    }

    /// Block until read back (native only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait(&self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        while !self.ready(device) {
            std::thread::yield_now();
        }
    }

    /// The f16 tiles in `coords` order once read back (None if not ready or a map failed).
    pub fn take(&self, device: &wgpu::Device) -> Option<Vec<Tile>> {
        if !self.mapped(device) {
            return None;
        }
        build_tiles(self.chunks.iter().map(|c| (&c.read, c.count, &*c.done)), self.channels)
    }

    /// Every chunk read back successfully.
    fn mapped(&self, device: &wgpu::Device) -> bool {
        self.ready(device) && self.chunks.iter().all(|c| *c.state.lock().unwrap() == Some(true))
    }

    /// [`PackedTiles::take`] on a worker thread: building the tiles copies every byte into new
    /// buffers (~140 MB for a long stroke, tens of ms of page faults), which would stall the UI.
    /// None if not ready yet (or a map failed: then `take` gives None too).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn take_in_background(&self, device: &wgpu::Device) -> Option<TakingTiles> {
        if !self.mapped(device) {
            return None;
        }
        let reads: Vec<(wgpu::Buffer, usize, Arc<AtomicBool>)> = self.chunks.iter().map(|c| (c.read.clone(), c.count, c.done.clone())).collect();
        let channels = self.channels;
        let (tx, rx) = std::sync::mpsc::channel();
        rayon::spawn(move || {
            let _ = tx.send(build_tiles(reads.iter().map(|(b, n, d)| (b, *n, &**d)), channels));
        });
        Some(TakingTiles(rx))
    }

    /// Buffer and byte offset holding tile `i`'s bits.
    pub(crate) fn bits(&self, i: usize) -> Option<(&wgpu::Buffer, u64)> {
        let c = self.chunks.iter().find(|c| i >= c.first && i < c.first + c.count)?;
        Some((&c.bits, (i - c.first) as u64 * tile_bytes(self.channels)))
    }
}

/// Tiles being built by [`PackedTiles::take_in_background`].
pub struct TakingTiles(std::sync::mpsc::Receiver<Option<Vec<Tile>>>);

impl TakingTiles {
    /// Non-blocking: Some once done (inner None: the read-back failed).
    pub fn try_take(&self) -> Option<Option<Vec<Tile>>> {
        match self.0.try_recv() {
            Ok(t) => Some(t),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Some(None),
        }
    }

    pub fn wait(&self) -> Option<Vec<Tile>> {
        self.0.recv().ok().flatten()
    }
}

/// Copy mapped read buffers (`count` tiles each) into tiles, then unmap them (and flag them
/// reusable).
fn build_tiles<'a>(reads: impl Iterator<Item = (&'a wgpu::Buffer, usize, &'a AtomicBool)>, channels: u8) -> Option<Vec<Tile>> {
    let per = tile_bytes(channels) as usize;
    let mut out = vec![];
    for (read, count, done) in reads {
        {
            let data = read.slice(..(per * count) as u64).get_mapped_range().ok()?;
            // The copy is bound by first-touch page faults on the new tile buffers: spread
            // it over the cores on native.
            let one = |i: usize| Tile::from_bytes(PixelFormat::F16, channels, &data[i * per..(i + 1) * per]);
            #[cfg(not(target_arch = "wasm32"))]
            let tiles: Option<Vec<Tile>> = {
                use rayon::prelude::*;
                (0..count).into_par_iter().map(one).collect()
            };
            #[cfg(target_arch = "wasm32")]
            let tiles: Option<Vec<Tile>> = (0..count).map(one).collect();
            out.extend(tiles?);
        }
        read.unmap();
        done.store(true, Ordering::Release);
    }
    Some(out)
}
